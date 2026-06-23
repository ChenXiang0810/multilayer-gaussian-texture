from gaussian_renderer.render import render
from gaussian_renderer.neilf import render_neilf
from gaussian_renderer.render_mesh import render_mesh
from gaussian_renderer.render_mesh_multi import render_mesh_multi


render_fn_dict = {
    "render": render_mesh,  # render render_mesh render_mesh_multi
    "neilf": render_neilf,
}