import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from collections import defaultdict
from random import randint
from utils.loss_utils import ssim, tv_loss, first_order_loss
from gaussian_renderer import render_fn_dict
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr, visualize_depth
from utils.system_utils import prepare_output_and_logger
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from gui import GUI
from scene.direct_light_map import DirectLightMap
from utils.graphics_utils import rgb_to_srgb
from torchvision.utils import save_image, make_grid
from lpipsPyTorch import lpips
from scene.utils import save_render_orb, save_depth_orb, save_normal_orb, save_albedo_orb, save_roughness_orb

import open3d as o3d

# Command
# python train.py --eval --lambda_normal_render_depth 0.1 --lambda_normal_smooth 0.01 --lambda_mask_entropy 0 --save_training_vis --lambda_depth_var 1e-2 --densify_until_iter 0 -s blender_dataset\grass -m output/grass/3dgs --mesh --compute_SHs_python --iterations 10000
# python train.py --eval --lambda_normal_render_depth 0.1 --lambda_normal_smooth 0.01 --lambda_mask_entropy 0 --save_training_vis --lambda_depth_var 1e-2 --densify_until_iter 0 --compute_SHs_python --lambda_albedo_smooth 0.01 --lambda_specular_smooth 0.01 --neural -s blender_dataset\grass -m output/grass/3dgs --mesh --load_iter 10000 --iterations 20000 --stage 2
# python train.py --eval --lambda_normal_render_depth 0.1 --lambda_normal_smooth 0.01 --lambda_mask_entropy 0 --save_training_vis --lambda_depth_var 1e-2 --densify_until_iter 0 --compute_SHs_python --lambda_albedo_smooth 0.01 --lambda_specular_smooth 0.01 --neural -s blender_dataset\grass -m output/grass/3dgs --mesh --load_iter 20000 --iterations 30000 --stage 3


def training(dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams, is_pbr=False, is_decompose=False, is_neural=False, load_iter=None, stage=1):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    """
    Setup Gaussians
    """
    gaussians = GaussianModel(dataset.sh_degree, render_type=args.type, use_brdf=is_decompose, use_neural=is_neural, stage=stage)
    scene = Scene(dataset, gaussians, load_iteration=load_iter)
    if args.checkpoint:
        print("Create Gaussians from checkpoint {}".format(args.checkpoint))
        first_iter = gaussians.create_from_ckpt(args.checkpoint, restore_optimizer=True)

    elif scene.loaded_iter:
        first_iter = load_iter
        gaussians.load_ply(os.path.join(dataset.model_path,
                                        "point_cloud",
                                        "iteration_" + str(scene.loaded_iter),
                                        "point_cloud.ply"))
        if is_decompose:
            light_path = os.path.join(dataset.model_path, "point_cloud", "iteration_" + str(scene.loaded_iter), "light.npy")
            if os.path.exists(light_path):
                gaussians.set_light_sh(light_path)
            else:
                gaussians.create_light_sh(3, 1)

            if is_neural:
                decoder_path = os.path.join(dataset.model_path, "point_cloud", "iteration_" + str(scene.loaded_iter), "decoder_params.pth")
                if os.path.exists(decoder_path):
                    gaussians.set_decoder(decoder_path)
                else:
                    gaussians.create_decoder(32, 2)

                encoder_path = os.path.join(dataset.model_path, "point_cloud", "iteration_" + str(scene.loaded_iter), "encoder_params.pth")
                if os.path.exists(encoder_path):
                    gaussians.set_encoder(encoder_path)
                else:
                    gaussians.create_encoder(128, 2)
    else:
        gaussians.create_from_pcd(scene.scene_info.point_cloud, scene.cameras_extent)

    gaussians.training_setup(opt)

    """
    Setup PBR components
    """
    pbr_kwargs = dict()
    if is_pbr:
        
        # first update visibility
        gaussians.update_visibility(pipe.sample_num)
        
        pbr_kwargs['sample_num'] = pipe.sample_num
        print("Using global incident light for regularization.")
        direct_env_light = DirectLightMap(dataset.env_resolution, opt.light_init)
        
        if args.checkpoint:
            env_checkpoint = os.path.dirname(args.checkpoint) + "/env_light_" + os.path.basename(args.checkpoint)
            print("Trying to load global incident light from ", env_checkpoint)
            if os.path.exists(env_checkpoint):
                direct_env_light.create_from_ckpt(env_checkpoint, restore_optimizer=True)
                print("Successfully loaded!")
            else:
                print("Failed to load!")

            direct_env_light.training_setup(opt)
            pbr_kwargs["env_light"] = direct_env_light

    """ Prepare render function and bg"""
    render_fn = render_fn_dict[args.type]
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    """ GUI """
    windows = None
    if args.gui:
        cam = scene.getTrainCameras()[0]
        c2w = cam.c2w.detach().cpu().numpy()
        center = gaussians.get_xyz.mean(dim=0).detach().cpu().numpy()

        render_kwargs = {"pc": gaussians, "pipe": pipe, "bg_color": background, "opt": opt, "is_training": False,
                         "dict_params": pbr_kwargs}

        windows = GUI(cam.image_height, cam.image_width, cam.FoVy,
                      c2w=c2w, center=center,
                      render_fn=render_fn, render_kwargs=render_kwargs,
                      mode='pbr')

    """ Training """
    viewpoint_stack = None
    ema_dict_for_log = defaultdict(int)
    progress_bar = tqdm(range(first_iter + 1, opt.iterations + 1), desc="Training progress",
                        initial=first_iter, total=opt.iterations)

    # My modification
    res = 512
    num_z = 4
    h_z = 2.0
    min_z = -1.0
    uv_id = np.linspace(0, res - 1, num=res)
    uv_val = (uv_id + 0.5) * 2 / res - 1.0
    u_val, v_val = np.meshgrid(uv_val, uv_val)
    uv = np.concatenate([u_val[..., None], v_val[..., None]], axis=-1)
    uv = uv.reshape((-1, 2))
    uv = np.repeat(uv, num_z, axis=0)
    uv = torch.tensor(uv).cuda()
    z_id = np.linspace(0, num_z - 1, num=num_z)
    z_val = (z_id + 0.5) * h_z / num_z + min_z
    z_val = np.repeat(z_val[None, :], res * res, axis=0)
    z_val = z_val.reshape((-1, 1))
    z_val = torch.tensor(z_val).cuda()

    # Load Mesh
    mesh = o3d.io.read_triangle_mesh('model/SphereMidPoly.obj')

    tri_vert = torch.tensor(np.asarray(mesh.triangles)).cuda()
    vert_pos = torch.tensor(np.asarray(mesh.vertices)).cuda()
    vert_uv = torch.tensor(np.asarray(mesh.triangle_uvs)).cuda()
    vert_normal = torch.tensor(np.asarray(mesh.vertex_normals)).cuda()

    tri_id = torch.linspace(0, tri_vert.shape[0] - 1, steps=tri_vert.shape[0], dtype=torch.int32).cuda()

    vert0_pos = vert_pos[tri_vert[tri_id, 0]]
    vert1_pos = vert_pos[tri_vert[tri_id, 1]]
    vert2_pos = vert_pos[tri_vert[tri_id, 2]]
    tri_pos = torch.cat([vert0_pos[:, None, :], vert1_pos[:, None, :], vert2_pos[:, None, :]], dim=1)

    vert0_normal = vert_normal[tri_vert[tri_id, 0]]
    vert1_normal = vert_normal[tri_vert[tri_id, 1]]
    vert2_normal = vert_normal[tri_vert[tri_id, 2]]
    tri_normal = torch.cat([vert0_normal[:, None, :], vert1_normal[:, None, :], vert2_normal[:, None, :]], dim=1)

    vert0_uv = vert_uv[3 * tri_id + 0]
    vert1_uv = vert_uv[3 * tri_id + 1]
    vert2_uv = vert_uv[3 * tri_id + 2]
    tri_uv = torch.cat([vert0_uv[:, None, :], vert1_uv[:, None, :], vert2_uv[:, None, :]], dim=1)

    # Precomputation
    local_n = (torch.tensor([0, 0, 1]).cuda()).to(dtype=torch.float32)[None, :].repeat(tri_vert.shape[0], 1)
    local_t = torch.cat([tri_uv[:, 1] - tri_uv[:, 0], torch.zeros([tri_uv.shape[0], 1]).cuda()], dim=-1)
    local_t = (local_t / torch.norm(local_t, dim=-1, keepdim=True)).to(dtype=torch.float32)
    local_b = torch.cross(local_t, local_n, dim=-1)
    local_frame = torch.cat([local_t[:, None, :], local_b[:, None, :], local_n[:, None, :]], dim=1)
    # This n may not be the same as face_n, so consider change this later (n should point positive)
    world_n = (tri_normal[:, 0] / torch.norm(tri_normal[:, 0], dim=-1, keepdim=True)).to(dtype=torch.float32)
    world_t = tri_pos[:, 1] - tri_pos[:, 0]
    world_t = (world_t / torch.norm(world_t, dim=-1, keepdim=True)).to(dtype=torch.float32)
    world_b = torch.cross(world_t, world_n, dim=-1)
    world_frame = torch.cat([world_t[:, None, :], world_b[:, None, :], world_n[:, None, :]], dim=1)
    mat_LtoW = torch.bmm(world_frame.permute(0, 2, 1), torch.inverse(local_frame.permute(0, 2, 1)))

    pos_size = torch.norm(tri_pos[:, 0] - tri_pos[:, 1], dim=-1) + torch.norm(tri_pos[:, 0] - tri_pos[:, 2], dim=-1) + torch.norm(tri_pos[:, 1] - tri_pos[:, 2], dim=-1)
    uv_size = torch.norm(tri_uv[:, 0] - tri_uv[:, 1], dim=-1) + torch.norm(tri_uv[:, 0] - tri_uv[:, 2], dim=-1) + torch.norm(tri_uv[:, 1] - tri_uv[:, 2], dim=-1)
    scale_size = pos_size / uv_size
    
    for iteration in progress_bar:
        gaussians.update_learning_rate(iteration)

        if windows is not None:
            windows.render()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        loss = 0
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == args.debug_from:
            pipe.debug = True

        pbr_kwargs["iteration"] = iteration - first_iter
        if not args.mesh:
            render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background,
                                   opt=opt, is_training=True, dict_params=pbr_kwargs, iteration=iteration)
        else:
            render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background, tri_id, tri_pos, tri_normal, tri_uv, mat_LtoW, scale_size,
                                   opt=opt, is_training=True, dict_params=pbr_kwargs, iteration=iteration)

        viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        tb_dict = render_pkg["tb_dict"]
        loss += render_pkg["loss"]

        # My modification
        xyz_points = gaussians.get_xyz
        zero_tensor = torch.zeros_like(xyz_points[..., 0]).cuda()
        dis_to_center_x = torch.maximum(torch.abs(xyz_points[..., 0] - uv[..., 0]) * res - 1, zero_tensor)
        dis_to_center_y = torch.maximum(torch.abs(xyz_points[..., 1] - uv[..., 1]) * res - 1, zero_tensor)
        dis_to_center_z = torch.maximum(torch.abs(xyz_points[..., 2] - z_val[..., 0]) * num_z - h_z * 0.5, zero_tensor)
        loss += torch.mean(dis_to_center_x) + torch.mean(dis_to_center_y) + torch.mean(dis_to_center_z)

        scale_points = gaussians.get_scaling
        scale_x = torch.maximum(scale_points[..., 0] * res - 3, zero_tensor)
        scale_y = torch.maximum(scale_points[..., 1] * res - 3, zero_tensor)
        scale_z = torch.maximum(scale_points[..., 2] * res - 3, zero_tensor)
        loss += torch.mean(scale_x) + torch.mean(scale_y) + torch.mean(scale_z)

        loss.backward()

        with torch.no_grad():
            if pipe.save_training_vis:
                if not args.mesh:
                    save_training_vis(viewpoint_cam, gaussians, background, render_fn,
                                      pipe, opt, first_iter, iteration, pbr_kwargs)
                else:
                    save_training_vis(viewpoint_cam, gaussians, background, render_fn,
                                      pipe, opt, first_iter, iteration, pbr_kwargs,
                                      tri_id, tri_pos, tri_normal, tri_uv, mat_LtoW, scale_size)
            # Progress bar
            pbar_dict = {"num": gaussians.get_xyz.shape[0]}
            if is_pbr:
                pbar_dict["light_mean"] = direct_env_light.get_env.mean().item()
                pbar_dict["env"] = direct_env_light.H
            for k in tb_dict:
                if k in ["psnr", "psnr_pbr"]:
                    ema_dict_for_log[k] = 0.4 * tb_dict[k] + 0.6 * ema_dict_for_log[k]
                    pbar_dict[k] = f"{ema_dict_for_log[k]:.{7}f}"
            progress_bar.set_postfix(pbar_dict)

            # Log and save
            if not args.mesh:
                training_report(tb_writer, iteration, tb_dict,
                                scene, render_fn, pipe=pipe,
                                bg_color=background, dict_params=pbr_kwargs)
            else:
                training_report(tb_writer, iteration, tb_dict,
                                scene, render_fn, pipe=pipe,
                                bg_color=background, tri_id=tri_id, tri_pos=tri_pos, tri_normal=tri_normal, tri_uv=tri_uv,
                                mat_LtoW=mat_LtoW, scale_size=scale_size, dict_params=pbr_kwargs)

            # densification
            
            if iteration < opt.densify_until_iter:
                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    
            # Optimizer step
            gaussians.step()
            for component in pbr_kwargs.values():
                try:
                    component.step()
                except:
                    pass
            
            # save checkpoints
            if iteration % args.save_interval == 0 or iteration == args.iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration % args.checkpoint_interval == 0 or iteration == args.iterations:
                
                torch.save((gaussians.capture(), iteration),
                           os.path.join(scene.model_path, "chkpnt" + str(iteration) + ".pth"))

                for com_name, component in pbr_kwargs.items():
                    try:
                        torch.save((component.capture(), iteration),
                                   os.path.join(scene.model_path, f"{com_name}_chkpnt" + str(iteration) + ".pth"))
                        print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    except:
                        pass

                    print("[ITER {}] Saving {} Checkpoint".format(iteration, com_name))

    if dataset.eval:
        if not args.mesh:
            eval_render(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs)
        else:
            eval_render(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs,
                        tri_id, tri_pos, tri_normal, tri_uv, mat_LtoW, scale_size)


def training_report(tb_writer, iteration, tb_dict, scene: Scene, renderFunc, pipe,
                    bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None,
                    opt: OptimizationParams = None, is_training=False,
                    tri_id=None, tri_pos=None, tri_normal=None, tri_uv=None, mat_LtoW=None, scale_size=None, **kwargs):
    if tb_writer:
        for key in tb_dict:
            tb_writer.add_scalar(f'train_loss_patches/{key}', tb_dict[key], iteration)

    # Report test and samples of training set
    if iteration % args.test_interval == 0:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                psnr_pbr_test = 0.0
                for idx, viewpoint in enumerate(
                        tqdm(config['cameras'], desc="Evaluating " + config['name'], leave=False)):
                    if tri_id is None:
                        render_pkg = renderFunc(viewpoint, scene.gaussians, pipe, bg_color,
                                                scaling_modifier, override_color, opt, is_training,
                                                **kwargs)
                    else:
                        render_pkg = renderFunc(viewpoint, scene.gaussians, pipe, bg_color,
                                                tri_id, tri_pos, tri_normal, tri_uv, mat_LtoW, scale_size,
                                                scaling_modifier, override_color, opt, is_training,
                                                **kwargs)

                    image = render_pkg["render"]
                    gt_image = viewpoint.original_image.cuda()

                    opacity = torch.clamp(render_pkg["opacity"], 0.0, 1.0)
                    depth = render_pkg["depth"]
                    depth = (depth - depth.min()) / (depth.max() - depth.min())
                    normal = torch.clamp(
                        render_pkg.get("normal", torch.zeros_like(image)) / 2 + 0.5 * opacity, 0.0, 1.0)

                    # BRDF
                    base_color = torch.clamp(render_pkg.get("base_color", torch.zeros_like(image)), 0.0, 1.0)
                    roughness = torch.clamp(render_pkg.get("roughness", torch.zeros_like(depth)), 0.0, 1.0)
                    image_pbr = render_pkg.get("pbr", torch.zeros_like(image))

                    grid = torchvision.utils.make_grid(
                        torch.stack([image, image_pbr, gt_image,
                                     opacity.repeat(3, 1, 1), depth.repeat(3, 1, 1), normal,
                                     base_color, roughness.repeat(3, 1, 1)], dim=0), nrow=3)

                    if tb_writer and (idx < 2):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             grid[None], global_step=iteration)

                    l1_test += F.l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    psnr_pbr_test += psnr(image_pbr, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                psnr_pbr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} PSNR_PBR {}".format(iteration, config['name'], l1_test,
                                                                                    psnr_test, psnr_pbr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr_pbr', psnr_pbr_test, iteration)
                if iteration == args.iterations:
                    with open(os.path.join(args.model_path, config['name'] + "_loss.txt"), 'w') as f:
                        f.write("L1 {} PSNR {} PSNR_PBR {}".format(l1_test, psnr_test, psnr_pbr_test))

        torch.cuda.empty_cache()


# def save_training_vis(viewpoint_cam, gaussians, background, render_fn, pipe, opt, first_iter, iteration, pbr_kwargs):
def save_training_vis(viewpoint_cam, gaussians, background, render_fn, pipe, opt, first_iter, iteration, pbr_kwargs,
                      tri_id=None, tri_pos=None, tri_normal=None, tri_uv=None, mat_LtoW=None, scale_size=None):
    os.makedirs(os.path.join(args.model_path, "visualize"), exist_ok=True)
    with torch.no_grad():
        if iteration % pipe.save_training_vis_iteration == 0 or iteration == first_iter + 1:
            if tri_id is None:
                render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background,
                                       opt=opt, is_training=False, dict_params=pbr_kwargs)
            else:
                render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background,
                                       tri_id, tri_pos, tri_normal, tri_uv, mat_LtoW, scale_size,
                                       opt=opt, is_training=False, dict_params=pbr_kwargs)

            visualization_list = [
                render_pkg["render"],
                viewpoint_cam.original_image.cuda(),
                visualize_depth(render_pkg["depth"]),
                (render_pkg["depth_var"] / 0.001).clamp_max(1).repeat(3, 1, 1),
                render_pkg["opacity"].repeat(3, 1, 1),
                render_pkg["normal"] * 0.5 + 0.5,
                render_pkg["pseudo_normal"] * 0.5 + 0.5,
            ]

            if is_pbr:
                
                H, W = render_pkg["pbr"].shape[1:]
                env = F.interpolate(render_pkg['env'].permute(0, 3, 1, 2), (H, 2*W))
                env_0 = env[0, :, :, :W]
                env_1 = env[0, :, :, W:]
                visualization_list.extend([
                    render_pkg["base_color"],
                    render_pkg["roughness"].repeat(3, 1, 1),
                    render_pkg["visibility"].repeat(3, 1, 1),
                    render_pkg["diffuse"],
                    # render_pkg["lights"],
                    render_pkg["specular"],
                    # render_pkg["local_lights"],
                    render_pkg["global_lights"],
                    render_pkg["pbr"],
                    rgb_to_srgb(env_0),
                    rgb_to_srgb(env_1),
                ])

            if gaussians.use_brdf:
                visualization_list.extend([
                    render_pkg["albedo"],
                    render_pkg["specular"].repeat(3, 1, 1),
                    # render_pkg["latent"][0:3],
                ])

            grid = torch.stack(visualization_list, dim=0)
            # grid = make_grid(grid, nrow=4)
            grid = make_grid(grid, nrow=5)
            scale = grid.shape[-2] / 800
            grid = F.interpolate(grid[None], (int(grid.shape[-2]/scale), int(grid.shape[-1]/scale)))[0]
            save_image(grid, os.path.join(args.model_path, "visualize", f"{iteration:06d}.png"))

def eval_render(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs,
                tri_id=None, tri_pos=None, tri_normal=None, tri_uv=None, mat_LtoW=None, scale_size=None):
    psnr_test = 0.0
    ssim_test = 0.0
    lpips_test = 0.0
    test_cameras = scene.getTestCameras()
    os.makedirs(os.path.join(args.model_path, 'eval', 'render'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'eval', 'gt'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'eval', 'normal'), exist_ok=True)
    if gaussians.use_pbr:
        os.makedirs(os.path.join(args.model_path, 'eval', 'base_color'), exist_ok=True)
        os.makedirs(os.path.join(args.model_path, 'eval', 'roughness'), exist_ok=True)
        os.makedirs(os.path.join(args.model_path, 'eval', 'lights'), exist_ok=True)
        os.makedirs(os.path.join(args.model_path, 'eval', 'local'), exist_ok=True)
        os.makedirs(os.path.join(args.model_path, 'eval', 'global'), exist_ok=True)
        os.makedirs(os.path.join(args.model_path, 'eval', 'visibility'), exist_ok=True)
    if gaussians.use_neural:
        os.makedirs(os.path.join(args.model_path, 'eval', 'albedo'), exist_ok=True)
        os.makedirs(os.path.join(args.model_path, 'eval', 'specular'), exist_ok=True)

    progress_bar = tqdm(range(0, len(test_cameras)), desc="Evaluating",
                        initial=0, total=len(test_cameras))

    with torch.no_grad():
        for idx in progress_bar:
            viewpoint = test_cameras[idx]
            if tri_id is None:
                results = render_fn(viewpoint, gaussians, pipe, background, opt=opt, is_training=False,
                                    dict_params=pbr_kwargs)
            else:
                results = render_fn(viewpoint, gaussians, pipe, background,
                                    tri_id, tri_pos, tri_normal, tri_uv, mat_LtoW, scale_size, opt=opt, is_training=False,
                                    dict_params=pbr_kwargs)
            if gaussians.use_pbr:
                image = results["pbr"]
            else:
                image = results["render"]

            image = torch.clamp(image, 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            psnr_test += psnr(image, gt_image).mean().double()
            ssim_test += ssim(image, gt_image).mean().double()
            lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()

            save_image(image, os.path.join(args.model_path, 'eval', "render", f"{viewpoint.image_name}.png"))
            save_image(gt_image, os.path.join(args.model_path, 'eval', "gt", f"{viewpoint.image_name}.png"))
            save_image(results["normal"] * 0.5 + 0.5,
                       os.path.join(args.model_path, 'eval', "normal", f"{viewpoint.image_name}.png"))
            if gaussians.use_pbr:
                save_image(results["base_color"],
                           os.path.join(args.model_path, 'eval', "base_color", f"{viewpoint.image_name}.png"))
                save_image(results["roughness"],
                           os.path.join(args.model_path, 'eval', "roughness", f"{viewpoint.image_name}.png"))
                save_image(results["lights"],
                           os.path.join(args.model_path, 'eval', "lights", f"{viewpoint.image_name}.png"))
                save_image(results["local_lights"],
                           os.path.join(args.model_path, 'eval', "local", f"{viewpoint.image_name}.png"))
                save_image(results["global_lights"],
                           os.path.join(args.model_path, 'eval', "global", f"{viewpoint.image_name}.png"))
                save_image(results["visibility"],
                           os.path.join(args.model_path, 'eval', "visibility", f"{viewpoint.image_name}.png"))
            if gaussians.use_neural:
                save_image(results["albedo"], os.path.join(args.model_path, 'eval', "albedo", f"{viewpoint.image_name}.png"))
                save_image(results["specular"], os.path.join(args.model_path, 'eval', "specular", f"{viewpoint.image_name}.png"))

    psnr_test /= len(test_cameras)
    ssim_test /= len(test_cameras)
    lpips_test /= len(test_cameras)
    with open(os.path.join(args.model_path, 'eval', "eval.txt"), "w") as f:
        f.write(f"psnr: {psnr_test}\n")
        f.write(f"ssim: {ssim_test}\n")
        f.write(f"lpips: {lpips_test}\n")
    print("\n[ITER {}] Evaluating {}: PSNR {} SSIM {} LPIPS {}".format(args.iterations, "test", psnr_test, ssim_test,
                                                                       lpips_test))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--gui', action='store_true', default=False, help="use gui")
    parser.add_argument('-t', '--type', choices=['render', 'normal', 'neilf'], default='render')
    parser.add_argument("--test_interval", type=int, default=2500)
    parser.add_argument("--save_interval", type=int, default=10000)  # 5000
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_interval", type=int, default=10000)  # 5000
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument("--neural", action="store_true", default=False)
    parser.add_argument("--decompose", action="store_true", default=False)
    parser.add_argument("--load_iter", type=int, default=None)
    parser.add_argument("--mesh", action="store_true", default=False)
    parser.add_argument("--stage", type=int, default=1)
    args = parser.parse_args(sys.argv[1:])
    print(f"Current model path: {args.model_path}")
    print(f"Current rendering type:  {args.type}")
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    is_pbr = args.type in ['neilf']
    is_decompose = True if args.neural else args.decompose
    training(lp.extract(args), op.extract(args), pp.extract(args), is_pbr=is_pbr, is_decompose=is_decompose, is_neural=args.neural, load_iter=args.load_iter, stage=args.stage)

    # All done
    print("\nTraining complete.")
