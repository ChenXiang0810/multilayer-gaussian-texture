
from scene import GaussianModel
import torch
import numpy as np
from plyfile import PlyData, PlyElement


def construct_list_of_attributes():
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(3):
        l.append('f_dc_{}'.format(i))
    for i in range(45):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(3):
        l.append('scale_{}'.format(i))
    for i in range(4):
        l.append('rot_{}'.format(i))
    return l


def save_ply(attributes):
    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]
    elements = np.empty(attributes.shape[0], dtype=dtype_full)

    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write("./point_cloud.ply")


def load_model(model_path, sh_degree=3):
    with torch.no_grad():  # temp
        gaussians = GaussianModel(sh_degree, use_neural=False, use_brdf=False)
        gaussians.load_ply(model_path)
        print(f"gaussain model has been load from {model_path}")
        return gaussians

# Select the source Gaussian
gaussians = load_model("D:/LabProject/GSTexture/Relightable3DGaussian/output/Grass/3dgs/point_cloud/iteration_10000/point_cloud_raw.ply")

src_res = 512  # The source resolution
tex_uv_id = np.linspace(0, src_res-1, num=src_res)
tex_uv_val = (tex_uv_id + 0.5) * 2 / src_res - 1.0
tex_u_val, tex_v_val = np.meshgrid(tex_uv_val, tex_uv_val)
tex_uv = np.concatenate([tex_u_val[..., None], tex_v_val[..., None]], axis=-1)

num_z = 2  # The source layers
gs_dim = 62
tex_gs = np.zeros((src_res * src_res * num_z, gs_dim))
tex_gs[..., 0:3] = gaussians.get_xyz.detach().cpu().numpy()
tex_gs[..., 3:6] = gaussians.get_normal.detach().cpu().numpy()
tex_gs[..., 6:9] = gaussians.get_shs.detach()[:, 0:1].transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
tex_gs[..., 9:54] = gaussians.get_shs.detach()[:, 1:].transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
tex_gs[..., 54:55] = gaussians.get_opacity.detach().cpu().numpy()
tex_gs[..., 55:58] = gaussians.get_scaling.detach().cpu().numpy()
tex_gs[..., 58:62] = gaussians.get_rotation.detach().cpu().numpy()


tex_gs = tex_gs.reshape((src_res, src_res, num_z, gs_dim))

# Select the source area
start_x = 8
start_y = 8
length = 496 

tex_gs = tex_gs[start_x:start_x+length, start_y:start_y+length]
scale_val = src_res / length
center_id_x = (start_x*2 + length) // 2
center_id_y = (start_y*2 + length) // 2
trans_val = tex_uv[center_id_x, center_id_y] - 1 / src_res
print(scale_val)
print(trans_val)

tex_gs[..., 0:2] = tex_gs[..., 0:2] - trans_val
tex_gs[..., 0:3] = tex_gs[..., 0:3] * scale_val
tex_gs[..., 55:58] = tex_gs[..., 55:58] * scale_val

tex_gs[..., 54] = np.log(tex_gs[..., 54] / (1 - tex_gs[..., 54]))
tex_gs[..., 55:58] = np.log(tex_gs[..., 55:58])

tex_gs = tex_gs.reshape(-1, gs_dim)
save_ply(tex_gs)






