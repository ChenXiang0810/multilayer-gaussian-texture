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

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gs_to_tri.h"
#include <cfloat>
#include <vector>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#define __CUDACC__
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;


__global__ void mapping(const int start_t, const float2* points, const float2* uv0s, const float2* uv1s, const float2* uv2s, int* ids)
{
    auto block = cg::this_thread_block();
    int cur_p = block.group_index().x;
    int cur_t = block.thread_index().x + start_t;

    float2 point = points[cur_p];
    float2 uv0 = uv0s[cur_t];
    float2 uv1 = uv1s[cur_t];
    float2 uv2 = uv2s[cur_t];

	float2 p0 = { uv0.x - point.x, uv0.y - point.y };
	float2 p1 = { uv1.x - point.x, uv1.y - point.y };
	float2 p2 = { uv2.x - point.x, uv2.y - point.y };

    float t0 = p0.x * p1.y - p0.y * p1.x;
    float t1 = p1.x * p2.y - p1.y * p2.x;
    float t2 = p2.x * p0.y - p2.y * p0.x;

    if((t0 * t1 >= 0) && (t0 * t2 >= 0))
    {
        ids[cur_p] = cur_t;
    }
}

void GStoTri::find_mapping(int P, int T, float2* points, float2* uv0s, float2* uv1s, float2* uv2s, int* ids)
{
    int size_batch = 64;
    int num_batch = T / size_batch;
    int start_t = 0;
    for(int batch = 0; batch < num_batch; batch++)
    {
	    mapping << <P, size_batch >> > (start_t, points, uv0s, uv1s, uv2s, ids);
	    start_t += size_batch;
    }
    int left_num = T - start_t;
    if(left_num > 0)
    {
        mapping << <P, left_num >> > (start_t, points, uv0s, uv1s, uv2s, ids);
    }
}