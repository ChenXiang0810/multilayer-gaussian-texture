/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>

void matchGStoTri(const torch::Tensor& points, const torch::Tensor& uv0s, const torch::Tensor& uv1s, const torch::Tensor& uv2s, torch::Tensor& ids);