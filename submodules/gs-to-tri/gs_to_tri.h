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

#ifndef GSTOTRI_H_INCLUDED
#define GSTOTRI_H_INCLUDED

class GStoTri
{
public:
	static void find_mapping(int P, int T, float2* points, float2* uv0s, float2* uv1s, float2* uv2s, int* ids);
};

#endif