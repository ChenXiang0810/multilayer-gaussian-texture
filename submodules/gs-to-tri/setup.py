#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

cxx_compiler_flags = []

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")

setup(
    name="gs_to_tri",
    ext_modules=[
        CUDAExtension(
            name="gs_to_tri._C",
            sources=[
            "match.cu",
            "gs_to_tri.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": [], "cxx": cxx_compiler_flags})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
