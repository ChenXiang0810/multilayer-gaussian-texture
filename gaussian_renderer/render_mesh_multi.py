
# This is script for 3D Gaussian Splatting rendering

import math
import torch
import torch.nn.functional as F
from arguments import OptimizationParams
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh, render_irrandiance_sh_sum
from utils.loss_utils import ssim, first_order_edge_aware_loss, second_order_edge_aware_loss, \
    bilateral_smooth_loss, tv_loss
from utils.image_utils import psnr
from .r3dg_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.general_utils import build_rotation
from gs_to_tri._C import matchGStoTri


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(*batch_dim, 9), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    # pyre-ignore [16]: `torch.Tensor` has no attribute `new_tensor`.
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(q_abs.new_tensor(0.1)))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
           torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
           ].reshape(*batch_dim, 4)


def translate_gaussians(pos, translate):
    new_pos = pos + translate
    return new_pos


def scale_gaussians(pos, scales, scale):
    # scale gaussians potsition
    new_pos = pos * scale

    # scale gaussians scale
    new_scales = scales * scale

    return new_pos, new_scales


def rotate_xyz(pos, rotmat):
    new_pos = pos @ rotmat.T
    return new_pos


def rotate_rot(quat, rotmat):
    new_rotation = build_rotation(quat)
    new_rotation = rotmat @ new_rotation
    new_quat = matrix_to_quaternion(new_rotation)
    # new_quat[:, [0, 1, 2, 3]] = new_quat[:, [3, 0, 1, 2]]  # xyzw -> wxyz
    # qua[..., 0:4] = torch.from_numpy(new_quat).to(qua.device).float()
    return new_quat


def point_in_triangle(pos, vert0, vert1, vert2):
    p0 = vert0 - pos
    p1 = vert1 - pos
    p2 = vert2 - pos

    t0 = p0[..., 0] * p1[..., 1] - p0[..., 1] * p1[..., 0]
    t1 = p1[..., 0] * p2[..., 1] - p1[..., 1] * p2[..., 0]
    t2 = p2[..., 0] * p0[..., 1] - p2[..., 1] * p0[..., 0]

    return (t0 * t1 >= 0) * (t0 * t2 >= 0) * (t1 * t2 >= 0)


def render_mesh_view(camera: Camera, pc, pipe, bg_color: torch.Tensor,
                tri_id, tri_pos, tri_normal, tri_uv, mat_LtoW, scale_size,
                scaling_modifier, override_color, computer_pseudo_normal=True):

    # Set up rasterization configuration
    tanfovx = math.tan(camera.FoVx * 0.5)
    tanfovy = math.tan(camera.FoVy * 0.5)
    intrinsic = camera.intrinsics
    raster_settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=float(intrinsic[0, 2]),
        cy=float(intrinsic[1, 2]),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=pc[0].active_sh_degree,
        campos=camera.camera_center,
        prefiltered=False,
        backward_geometry=True,
        computer_pseudo_normal=computer_pseudo_normal,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D_multi = []
    means2D_multi = []
    shs_multi = []
    colors_precomp_multi = []
    opacities_multi = []
    scales_multi = []
    rotations_multi = []
    cov3D_precomp_multi = []
    features_multi = []

    for mesh_id in range(len(scale_size)):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(pc[mesh_id].get_xyz, dtype=pc[mesh_id].get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        means3D = pc[mesh_id].get_xyz
        means2D = screenspace_points
        opacity = pc[mesh_id].get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc[mesh_id].get_covariance(scaling_modifier)
        else:
            scales = pc[mesh_id].get_scaling
            rotations = pc[mesh_id].get_rotation

        normals = pc[mesh_id].get_normal

        # For mesh
        scale_norm = torch.tensor([0.5, 0.5, 0.5]).cuda()
        means3D, scales = scale_gaussians(means3D, scales, scale_norm[None, :])

        translate_norm = torch.tensor([0.5, 0.5, 0]).cuda()
        means3D = translate_gaussians(means3D, translate_norm[None, :])

        gs_to_tri = torch.zeros(means3D.shape[0], dtype=torch.int32).cuda() - 1
        matchGStoTri(means3D[..., 0:2].float(), tri_uv[mesh_id][:, 0, :].float(), tri_uv[mesh_id][:, 1, :].float(), tri_uv[mesh_id][:, 2, :].float(), gs_to_tri)

        mask_valid = torch.where(gs_to_tri != -1)[0]
        # print(mask_valid.shape)

        gs_to_tri = gs_to_tri[mask_valid]
        means3D = means3D[mask_valid]
        means2D = means2D[mask_valid]
        opacity = opacity[mask_valid]
        scales = scales[mask_valid]
        rotations = rotations[mask_valid]
        normals = normals[mask_valid]

        tri_uv3 = tri_uv[mesh_id][gs_to_tri]
        tri_uv3 = torch.cat([tri_uv3, torch.ones([tri_uv3.shape[0], 3, 1]).cuda()], dim=-1)
        tri_gs3 = torch.cat([means3D[..., 0:2], torch.ones([means3D.shape[0], 1]).cuda()], dim=-1).unsqueeze(-1)
        mat_inv = torch.inverse(tri_uv3.permute(0, 2, 1)).to(dtype=torch.float32)
        bary = torch.bmm(mat_inv, tri_gs3).squeeze(-1)
        # print(bary.shape)
        world_pos = bary[..., 0:1] * tri_pos[mesh_id][gs_to_tri, 0] + bary[..., 1:2] * tri_pos[mesh_id][gs_to_tri, 1] + bary[..., 2:3] * \
                    tri_pos[mesh_id][gs_to_tri, 2]
        world_normal = bary[..., 0:1] * tri_normal[mesh_id][gs_to_tri, 0] + bary[..., 1:2] * tri_normal[mesh_id][gs_to_tri, 1] + bary[...,
                                                                                                               2:3] * \
                       tri_normal[mesh_id][gs_to_tri, 2]
        # print(world_pos.shape)
        # print(world_normal.shape)

        mat_LtoW_cur = mat_LtoW[mesh_id][gs_to_tri]
        scale_size_cur = scale_size[mesh_id][gs_to_tri].unsqueeze(-1)

        means3D, scales = scale_gaussians(means3D, scales, scale_size_cur.repeat(1, 3))
        means3D = world_pos + means3D[..., 2:3] * world_normal
        rotations = rotate_rot(rotations, mat_LtoW_cur)
        normals = mat_LtoW_cur @ normals.unsqueeze(-1)
        normals = normals.squeeze(-1)
        means3D = means3D.to(torch.float32)
        scales = scales.to(torch.float32)

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if pipe.compute_SHs_python:
                if not pc[mesh_id].use_brdf:
                    shs_view = pc[mesh_id].get_shs[mask_valid].transpose(1, 2).view(-1, 3, (pc[mesh_id].max_sh_degree + 1) ** 2)
                    dir_pp = (means3D - camera.camera_center.repeat(means3D.shape[0], 1))
                    dir_pp = (mat_LtoW_cur.permute(0, 2, 1) @ dir_pp.unsqueeze(-1)).squeeze(-1)
                    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0).to(torch.float32)
                elif not pc[mesh_id].use_neural:
                    envSHs = pc[mesh_id].get_light[None]
                    albedo = pc[mesh_id].get_albedo[mask_valid]
                    specular = pc[mesh_id].get_specular[mask_valid]
                    s = pc[mesh_id].get_glossiness[mask_valid]
                    normal = normals.detach()

                    # Diffusion Color: from the first two levels of SH only
                    diffuse_rgb = render_irrandiance_sh_sum(envSHs[:, :, :9], normal)
                    diffuse_rgb = torch.clamp_min(diffuse_rgb, 0.0)
                    diffuse = albedo * diffuse_rgb  # N_rays*N_samples, 3

                    # reflection formula: w_i = 2 * |w_o * n| * n - w_o
                    rays_d = pc[mesh_id].get_xyz[mask_valid] - camera.camera_center.repeat(means3D.shape[0], 1)
                    rays_d = rays_d / rays_d.norm(dim=1, keepdim=True)
                    cos_theta = -(rays_d * normal).sum(dim=-1, keepdim=True)
                    reflect_d = 2 * cos_theta * normal + rays_d  # N_rays*N_samples, 3
                    reflect_d = reflect_d / reflect_d.norm(dim=1, keepdim=True)
                    # mask_back = torch.where(cos_theta > 0, torch.ones_like(cos_theta), torch.zeros_like(cos_theta)).detach()

                    order_coeff = torch.arange(0, envSHs.shape[0], device=envSHs.device)[:, None]  # 1, N_sh, 1
                    order_coeff = torch.pow(order_coeff, 0.5).floor()
                    sh_coeff = torch.exp(-order_coeff * order_coeff / 2 / s)[..., None] * envSHs[:, :, :9]  # 1, N_sh, 3

                    specular_rgb = render_irrandiance_sh_sum(sh_coeff, reflect_d)
                    specular_rgb = torch.clamp_min(specular_rgb, 0.0)
                    specular_color = specular * specular_rgb

                    colors_precomp = torch.clamp_min(diffuse + specular_color, 0.0)  # * mask_back
                    # colors_precomp = torch.clamp_min(diffuse, 0.0) * mask_back
                else:
                    if pc[mesh_id].stage == 3:
                        # For encoder-decoder stage3
                        latent = pc[mesh_id].get_latent[mask_valid]
                        # latent = torch.sigmoid(pc[mesh_id].get_latent[mask_valid])  # modified!!!
                        brdf = pc[mesh_id].get_decoder(latent)
                    elif pc[mesh_id].stage == 2:
                        # For encoder-decoder stage2
                        trained_normal = (pc[mesh_id].get_normal[mask_valid].detach() + 1) * 0.5
                        net_input = torch.cat([pc[mesh_id].get_shs[mask_valid].reshape(-1, 48).detach(), trained_normal], dim=-1)
                        brdf = pc[mesh_id].get_decoder(pc[mesh_id].get_encoder(pc[mesh_id].get_encoding(net_input)))

                    albedo = torch.sigmoid(brdf[..., 0:3])
                    specular = torch.sigmoid(brdf[..., 3:4])
                    s = torch.nn.functional.softplus(brdf[..., 4:5]) + 1

                    normal = brdf[..., 5:8]
                    normal = normal / (normal.norm(dim=1, keepdim=True) + 1e-4)
                    # flag_sign = torch.where(normal[..., 2:3] > 0, torch.ones_like(normal[..., 2:3]), -torch.ones_like(normal[..., 2:3])).detach()
                    # normal = normal * flag_sign
                    normal = mat_LtoW_cur @ normal.unsqueeze(-1).to(torch.float32)
                    normal = normal.squeeze(-1).to(torch.float16)
                    normals = normal
                    # normal = normals  # For mesh rendering

                    if not pipe.compute_SHs_python_defer:
                        envSHs = pc[mesh_id].get_light[None]

                        # Diffusion Color: from the first two levels of SH only
                        diffuse_rgb = render_irrandiance_sh_sum(envSHs[:, :, :9], normal)
                        diffuse_rgb = torch.clamp_min(diffuse_rgb, 0.0)
                        diffuse = albedo * diffuse_rgb  # N_rays*N_samples, 3

                        # reflection formula: w_i = 2 * |w_o * n| * n - w_o
                        rays_d = pc[mesh_id].get_xyz[mask_valid] - camera.camera_center.repeat(means3D.shape[0], 1)
                        rays_d = rays_d / rays_d.norm(dim=1, keepdim=True)
                        cos_theta = -(rays_d * normal).sum(dim=-1, keepdim=True)
                        reflect_d = 2 * cos_theta * normal + rays_d  # N_rays*N_samples, 3
                        reflect_d = reflect_d / reflect_d.norm(dim=1, keepdim=True)
                        # mask_back = torch.where(cos_theta > 0, torch.ones_like(cos_theta), torch.zeros_like(cos_theta)).detach()
                        # normals = normal * mask_back

                        order_coeff = torch.arange(0, envSHs.shape[0], device=envSHs.device)[:, None]  # 1, N_sh, 1
                        order_coeff = torch.pow(order_coeff, 0.5).floor()
                        sh_coeff = torch.exp(-order_coeff * order_coeff / 2 / s)[..., None] * envSHs[:, :, :9]  # 1, N_sh, 3

                        specular_rgb = render_irrandiance_sh_sum(sh_coeff, reflect_d)
                        specular_rgb = torch.clamp_min(specular_rgb, 0.0)
                        specular_color = specular * specular_rgb

                        colors_precomp = torch.clamp_min(diffuse + specular_color, 0.0)  # * mask_back
                        # if mesh_id == 1:
                        #     colors_precomp[..., 0] = 0.66
                        #     colors_precomp[..., 1] = 0.63
                        #     colors_precomp[..., 2] = 0.56
                    else:
                        shs = pc[mesh_id].get_shs[mask_valid]
            else:
                shs = pc[mesh_id].get_shs[mask_valid]
        else:
            colors_precomp = override_color

        dir_pp = (pc[mesh_id].get_xyz[mask_valid] - camera.camera_center.repeat(means3D.shape[0], 1))
        dir_pp_normalized = F.normalize(dir_pp, dim=-1)

        xyz_homo = torch.cat([means3D, torch.ones_like(means3D[:, :1])], dim=-1)
        depths = (xyz_homo @ camera.world_view_transform)[:, 2:3]
        depths2 = depths.square()
        features = torch.cat([normals, depths, depths2], dim=-1)

        if pc[mesh_id].use_brdf:
            if pc[mesh_id].use_neural:
                features = torch.cat([features, albedo, specular], dim=-1)

        means3D_multi.append(means3D)
        means2D_multi.append(means2D)
        # shs_multi.append(shs)
        colors_precomp_multi.append(colors_precomp)
        opacities_multi.append(opacity)
        scales_multi.append(scales)
        rotations_multi.append(rotations)
        # cov3D_precomp_multi.append(cov3D_precomp)
        features_multi.append(features)

    means3D_multi = torch.cat(means3D_multi, dim=0)
    means2D_multi = torch.cat(means2D_multi, dim=0)
    # shs_multi = torch.cat(shs_multi, dim=0)
    shs_multi = None
    colors_precomp_multi = torch.cat(colors_precomp_multi, dim=0)
    opacities_multi = torch.cat(opacities_multi, dim=0)
    scales_multi = torch.cat(scales_multi, dim=0)
    rotations_multi = torch.cat(rotations_multi, dim=0)
    # cov3D_precomp_multi = torch.cat(cov3D_precomp_multi, dim=0)
    cov3D_precomp_multi = None
    features_multi = torch.cat(features_multi, dim=0)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    (num_rendered, num_contrib, rendered_image, rendered_opacity, rendered_depth,
     rendered_feature, rendered_pseudo_normal, rendered_surface_xyz, weights, radii) = rasterizer(
        means3D=means3D_multi,
        means2D=means2D_multi,
        shs=shs_multi,
        colors_precomp=colors_precomp_multi,
        opacities=opacities_multi,
        scales=scales_multi,
        rotations=rotations_multi,
        cov3D_precomp=cov3D_precomp_multi,
        features=features_multi,
    )
     
    mask = num_contrib > 0
    rendered_feature = rendered_feature / rendered_opacity.clamp_min(1e-5) * mask

    if not pc[mesh_id].use_brdf:
        rendered_normal, rendered_depth, rendered_depth2 = torch.split(rendered_feature, [3, 1, 1], dim=0)
    elif pc[mesh_id].use_neural:
        rendered_normal, rendered_depth, rendered_depth2, rendered_albedo, rendered_specular = torch.split(rendered_feature, [3, 1, 1, 3, 1], dim=0)
    
    rendered_var = rendered_depth2 - rendered_depth.square()

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    results = {"render": rendered_image,
               "opacity": rendered_opacity,
               "depth": rendered_depth,
               "depth_var": rendered_var,
               "normal": rendered_normal,
               "pseudo_normal": rendered_pseudo_normal,
               "surface_xyz": rendered_surface_xyz,
               "viewspace_points": screenspace_points,
               "visibility_filter": radii > 0,
               "radii": radii,
               "num_rendered": num_rendered,
               "num_contrib": num_contrib,
               "opacities": opacity,
               "normals": normals,
               "directions": dir_pp_normalized,
               "weights": weights}

    if pc[mesh_id].use_brdf:
        results["albedo"] = rendered_albedo
        results["specular"] = rendered_specular

    return results


def calculate_loss(viewpoint_camera, pc, render_pkg, opt, iteration):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }
    
    rendered_image = render_pkg["render"]
    rendered_opacity = render_pkg["opacity"]
    rendered_depth = render_pkg["depth"]
    rendered_normal = render_pkg["normal"]
    visibility_filter = render_pkg["visibility_filter"]
    gt_image = viewpoint_camera.original_image.cuda()
    image_mask = viewpoint_camera.image_mask.cuda()

    Ll1 = F.l1_loss(rendered_image, gt_image)
    ssim_val = ssim(rendered_image, gt_image)
    tb_dict["loss_l1"] = Ll1.item()
    tb_dict["psnr"] = psnr(rendered_image, gt_image).mean().item()
    tb_dict["ssim"] = ssim_val.item()
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)

    if opt.lambda_mask_entropy > 0:
        o = rendered_opacity.clamp(1e-6, 1 - 1e-6)
        loss_mask_entropy = -(image_mask * torch.log(o) + (1-image_mask) * torch.log(1 - o)).mean()
        tb_dict["loss_mask_entropy"] = loss_mask_entropy.item()
        loss = loss + opt.lambda_mask_entropy * loss_mask_entropy

    if opt.lambda_normal_render_depth > 0:
        normal_pseudo = render_pkg['pseudo_normal']
        loss_normal_render_depth = F.mse_loss(
            rendered_normal * image_mask, normal_pseudo.detach() * image_mask)
        tb_dict["loss_normal_render_depth"] = loss_normal_render_depth.item()
        loss = loss + opt.lambda_normal_render_depth * loss_normal_render_depth

    if opt.lambda_normal_smooth > 0:
        loss_normal_smooth = first_order_edge_aware_loss(rendered_normal, gt_image)
        tb_dict["loss_normal_smooth"] = loss_normal_smooth.item()
        lambda_normal_smooth = opt.lambda_normal_smooth
        loss = loss + lambda_normal_smooth * loss_normal_smooth
    
    if opt.lambda_depth_smooth > 0:
        loss_depth_smooth = first_order_edge_aware_loss(rendered_depth, gt_image)
        tb_dict["loss_depth_smooth"] = loss_depth_smooth.item()
        lambda_depth_smooth = opt.lambda_depth_smooth
        loss = loss + lambda_depth_smooth * loss_depth_smooth
        
    if opt.lambda_point_entropy > 0:
        ws = render_pkg["weights"]
        vis_opacities = render_pkg["opacities"]
        loss_point_entropy = (ws * (
                        - vis_opacities * torch.log(vis_opacities + 1e-10)
                        - (1 - vis_opacities) * torch.log(1 - vis_opacities + 1e-10)
                        )).mean()
        tb_dict["loss_normal_smooth"] = loss_point_entropy.item()
        loss = loss + opt.lambda_point_entropy * loss_point_entropy
        
    if opt.lambda_orientation > 0 and iteration > opt.lambda_orientation_from_iter:
        ws = render_pkg["weights"].clamp_max(1)
        normals = render_pkg["normals"]
        directions = render_pkg["directions"]
        loss_orientation = (ws * (normals * directions).sum(-1, keepdim=True).clamp_min(0.0)).mean()
        tb_dict["loss_orientation"] = loss_orientation.item()
        loss = loss + opt.lambda_orientation * loss_orientation
    
    if opt.lambda_depth_var > 0:
        depth_var = render_pkg["depth_var"]
        loss_depth_var = depth_var.clamp_min(1e-6).sqrt().mean()
        tb_dict["loss_depth_var"] = loss_depth_var.item()
        lambda_depth_var = opt.lambda_depth_var * min(math.pow(10, iteration / 5000), 100)
        # lambda_depth_var = opt.lambda_depth_var
        loss = loss + lambda_depth_var * loss_depth_var
    
    
    if opt.lambda_surface > 0:
        center, _ = torch.median(pc.get_xyz, dim=0)
        loss_surface = torch.exp(-(pc.get_xyz - center[None, ...]).abs().mean())
        
        tb_dict["loss_surface"] = loss_surface.item()
        loss = loss + opt.lambda_surface * loss_surface
        
    if opt.lambda_scaling > 0:
        scaling = pc.get_scaling
        scaling_loss = (scaling - scaling.mean(dim=-1, keepdim=True)).abs().sum(-1).mean()
        lambda_scaling = opt.lambda_scaling - 0.99 * opt.lambda_scaling * min(1, 4 * iteration / opt.iterations)
        loss = loss + lambda_scaling * scaling_loss

    if pc.use_brdf:
        rendered_albedo = render_pkg["albedo"]
        rendered_specular = render_pkg["specular"]
        # rendered_latent = render_pkg["latent"]

        if opt.lambda_albedo_smooth > 0:
            # image_mask = viewpoint_camera.image_mask.cuda()
            # loss_albedo_smooth = first_order_edge_aware_loss(rendered_albedo * image_mask, gt_image)
            loss_albedo_smooth = first_order_edge_aware_loss(rendered_albedo, gt_image)
            tb_dict["loss_albedo_smooth"] = loss_albedo_smooth.item()
            loss = loss + opt.lambda_albedo_smooth * loss_albedo_smooth

        if opt.lambda_specular_smooth > 0:
            # image_mask = viewpoint_camera.image_mask.cuda()
            # loss_specular_smooth = first_order_edge_aware_loss(rendered_specular * image_mask, gt_image)
            loss_specular_smooth = first_order_edge_aware_loss(rendered_specular, gt_image)
            tb_dict["loss_specular_smooth"] = loss_specular_smooth.item()
            loss = loss + opt.lambda_specular_smooth * loss_specular_smooth

        # if opt.lambda_latent_smooth > 0:
        #     # image_mask = viewpoint_camera.image_mask.cuda()
        #     # loss_specular_smooth = first_order_edge_aware_loss(rendered_latent * image_mask, gt_image)
        #     loss_latent_smooth = first_order_edge_aware_loss(rendered_latent, gt_image)
        #     tb_dict["loss_latent_smooth"] = loss_latent_smooth.item()
        #     loss = loss + opt.lambda_latent_smooth * loss_latent_smooth
    
    tb_dict["loss"] = loss.item()
    
    return loss, tb_dict

def render_mesh_multi(viewpoint_camera: Camera, pc, pipe, bg_color: torch.Tensor,
            tri_id, tri_pos, tri_normal, tri_uv, mat_LtoW, scale_size,
           scaling_modifier=1.0,override_color=None, opt: OptimizationParams = None, 
           is_training=False, dict_params=None, iteration=0):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """
    results = render_mesh_view(viewpoint_camera, pc, pipe, bg_color, tri_id, tri_pos, tri_normal, tri_uv, mat_LtoW,
                            scale_size, scaling_modifier, override_color,
                          computer_pseudo_normal=True if opt is not None and opt.lambda_normal_render_depth>0 else False)

    if is_training:
        loss, tb_dict = calculate_loss(viewpoint_camera, pc, results, opt, iteration)
        results["tb_dict"] = tb_dict
        results["loss"] = loss
    
    return results
