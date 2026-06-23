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

#include "match.h"
#include "gs_to_tri.h"

void matchGStoTri(const torch::Tensor& points, const torch::Tensor& uv0s, const torch::Tensor& uv1s, const torch::Tensor& uv2s, torch::Tensor& ids)
{
  const int P = points.size(0);
  const int T = uv0s.size(0);

  GStoTri::find_mapping(P, T, (float2*)points.contiguous().data<float>(), (float2*)uv0s.contiguous().data<float>(), (float2*)uv1s.contiguous().data<float>(), (float2*)uv2s.contiguous().data<float>(), (int*)ids.contiguous().data<int>());

}