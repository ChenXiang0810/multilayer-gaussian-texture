/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr
 */

#include <projects/gaussianviewer/renderer/GaussianView.hpp>
#include <core/graphics/GUI.hpp>
#include <thread>
#include <boost/asio.hpp>
#include <rasterizer.h>
#include <imgui_internal.h>

#include <unsupported/Eigen/MatrixFunctions>
Eigen::Matrix<float, 3, 3> so3_l1[3];
Eigen::Matrix<float, 5, 5> so3_l2[3];
Eigen::Matrix<float, 7, 7> so3_l3[3];
Eigen::Matrix<float, 3, 3> mat_P;
Eigen::Matrix<float, 3, 3> inv_P;

// Define the types and sizes that make up the contents of each Gaussian 
// in the trained model.
typedef sibr::Vector3f Pos;
template<int D>
struct SHs
{
	float shs[(D+1)*(D+1)*3];
};
struct Scale
{
	float scale[3];
};
struct Rot
{
	float rot[4];
};
template<int D>
struct RichPoint
{
	Pos pos;
	float n[3];
	SHs<D> shs;
	float opacity;
	Scale scale;
	Rot rot;
};

float sigmoid(const float m1)
{
	return 1.0f / (1.0f + exp(-m1));
}

float inverse_sigmoid(const float m1)
{
	return log(m1 / (1.0f - m1));
}

# define CUDA_SAFE_CALL_ALWAYS(A) \
A; \
cudaDeviceSynchronize(); \
if (cudaPeekAtLastError() != cudaSuccess) \
SIBR_ERR << cudaGetErrorString(cudaGetLastError());

#if DEBUG || _DEBUG
# define CUDA_SAFE_CALL(A) CUDA_SAFE_CALL_ALWAYS(A)
#else
# define CUDA_SAFE_CALL(A) A
#endif

void computeMatrixRT(Eigen::Matrix3f pSrc, Eigen::Matrix3f pDst, Eigen::Matrix3f& matR, Eigen::Vector3f& vecT)
{
	Eigen::Vector3f pSrcMean = pSrc.colwise().mean();
	Eigen::Vector3f pDstMean = pDst.colwise().mean();

	for (int i = 0; i < 3; i++)
	{
		pSrc.row(i) = pSrc.row(i) - pSrcMean.transpose();
		pDst.row(i) = pDst.row(i) - pDstMean.transpose();
	}

	Eigen::Matrix3f matS = pSrc.transpose() * pDst;
	Eigen::JacobiSVD<Eigen::Matrix3f> svd(matS, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix3f matU = svd.matrixU();
	Eigen::Matrix3f matV = svd.matrixV();
	float det = (matU * matV.transpose()).determinant();
	Eigen::Matrix3f matM;
	matM << 1, 0, 0,
		0, 1, 0,
		0, 0, det;

	matR = matV * matM * matU.transpose();
	vecT(0) = pDstMean(0) - pSrcMean.dot(matR.row(0).transpose());
	vecT(1) = pDstMean(1) - pSrcMean.dot(matR.row(1).transpose());
	vecT(2) = pDstMean(2) - pSrcMean.dot(matR.row(2).transpose());
}

bool pointInTriangle(Eigen::Vector2f pos, Eigen::Vector2f v0, Eigen::Vector2f v1, Eigen::Vector2f v2)
{
	Eigen::Vector2f p0 = v0 - pos;
	Eigen::Vector2f p1 = v1 - pos;
	Eigen::Vector2f p2 = v2 - pos;

	float t0 = p0(0) * p1(1) - p0(1) * p1(0);
	float t1 = p1(0) * p2(1) - p1(1) * p2(0);
	float t2 = p2(0) * p0(1) - p2(1) * p0(0);

	return (t0 * t1 >= 0) && (t0 * t2 >= 0) && (t1 * t2 >= 0);
}

void scaleGS(float val_scale, Pos& gs_pos, Scale& gs_scale)
{
	gs_scale.scale[0] *= val_scale;
	gs_scale.scale[1] *= val_scale;
	gs_scale.scale[2] *= val_scale;
	gs_pos *= val_scale;
}

void translateGS(sibr::Vector3f vec_trans, Pos& gs_pos)
{
	gs_pos += vec_trans;
}

void rotateGS(sibr::Matrix3f mat_rot, Pos& gs_pos, Rot& gs_rot, SHs<3>& gs_shs)
{
	gs_pos = mat_rot * gs_pos;

	sibr::Quaternionf q_cur(gs_rot.rot[0], gs_rot.rot[1], gs_rot.rot[2], gs_rot.rot[3]);
	sibr::Matrix3f mat_rot_new = q_cur.toRotationMatrix();
	mat_rot_new = mat_rot * mat_rot_new;
	sibr::Quaternionf q_new(mat_rot_new);
	gs_rot.rot[0] = q_new.w();
	gs_rot.rot[1] = q_new.x();
	gs_rot.rot[2] = q_new.y();
	gs_rot.rot[3] = q_new.z();

	// // Rotate SH
	// Eigen::Matrix3f permute_mat_rot = inv_P * mat_rot * mat_P;
	// Eigen::Vector3f angles_rot = permute_mat_rot.eulerAngles(1, 0, 1);
	// angles_rot(0) = std::fmod(angles_rot(0), 2 * M_PI);
	// angles_rot(1) = std::fmod(-angles_rot(1), 2 * M_PI);
	// angles_rot(2) = std::fmod(angles_rot(2), 2 * M_PI);
	// angles_rot(0) = angles_rot(0) < 0 ? angles_rot(0) + 2 * M_PI : angles_rot(0);
	// angles_rot(1) = angles_rot(1) < 0 ? angles_rot(1) + 2 * M_PI : angles_rot(1);
	// angles_rot(2) = angles_rot(2) < 0 ? angles_rot(2) + 2 * M_PI : angles_rot(2);
	// Eigen::Matrix<float, 3, 3> D_1 = (angles_rot(0) * so3_l1[1]).exp() * (angles_rot(1) * so3_l1[0]).exp() * (angles_rot(2) * so3_l1[1]).exp();
	// Eigen::Matrix<float, 5, 5> D_2 = (angles_rot(0) * so3_l2[1]).exp() * (angles_rot(1) * so3_l2[0]).exp() * (angles_rot(2) * so3_l2[1]).exp();
	// Eigen::Matrix<float, 7, 7> D_3 = (angles_rot(0) * so3_l3[1]).exp() * (angles_rot(1) * so3_l3[0]).exp() * (angles_rot(2) * so3_l3[1]).exp();
	// Eigen::Vector3f D_1_vec[3];
	// Eigen::Vector3f D_2_vec[5];
	// Eigen::Vector3f D_3_vec[7];
	// for (uint shs_n = 0; shs_n < 3; shs_n++)
	// {
	// 	for (uint col_n = 0; col_n < 3; col_n++)
	// 	{
	// 		D_1_vec[shs_n](col_n) = D_1(shs_n, 0) * gs_shs.shs[3 + col_n] +
	// 			D_1(shs_n, 1) * gs_shs.shs[6 + col_n] +
	// 			D_1(shs_n, 2) * gs_shs.shs[9 + col_n];
	// 	}
	// }
	// for (uint shs_n = 0; shs_n < 5; shs_n++)
	// {
	// 	for (uint col_n = 0; col_n < 3; col_n++)
	// 	{
	// 		D_2_vec[shs_n](col_n) = D_2(shs_n, 0) * gs_shs.shs[12 + col_n] +
	// 			D_2(shs_n, 1) * gs_shs.shs[15 + col_n] +
	// 			D_2(shs_n, 2) * gs_shs.shs[18 + col_n] +
	// 			D_2(shs_n, 3) * gs_shs.shs[21 + col_n] +
	// 			D_2(shs_n, 4) * gs_shs.shs[24 + col_n];
	// 	}
	// }
	// for (uint shs_n = 0; shs_n < 7; shs_n++)
	// {
	// 	for (uint col_n = 0; col_n < 3; col_n++)
	// 	{
	// 		D_3_vec[shs_n](col_n) = D_3(shs_n, 0) * gs_shs.shs[27 + col_n] +
	// 			D_3(shs_n, 1) * gs_shs.shs[30 + col_n] +
	// 			D_3(shs_n, 2) * gs_shs.shs[33 + col_n] +
	// 			D_3(shs_n, 3) * gs_shs.shs[36 + col_n] +
	// 			D_3(shs_n, 4) * gs_shs.shs[39 + col_n] + 
	// 			D_3(shs_n, 5) * gs_shs.shs[42 + col_n] + 
	// 			D_3(shs_n, 6) * gs_shs.shs[45 + col_n];
	// 	}
	// }
	// for (uint id = 0; id < 9; id++)
	// {
	// 	gs_shs.shs[3 + id] = D_1_vec[id / 3u](id % 3);
	// }
	// for (uint id = 0; id < 15; id++)
	// {
	// 	gs_shs.shs[12 + id] = D_2_vec[id / 3u](id % 3);
	// }
	// for (uint id = 0; id < 21; id++)
	// {
	// 	gs_shs.shs[27 + id] = D_3_vec[id / 3u](id % 3);
	// }
}

float B0(sibr::Vector2f uv)
{
	return std::min(std::min(uv.x() - floor(uv.x()), uv.y() - floor(uv.y())), std::min(ceil(uv.x()) - uv.x(), ceil(uv.y()) - uv.y()));
}

float B1(sibr::Vector2f uv)
{
	uv += sibr::Vector2f(0.5, 0.5);
	return std::min(std::min(uv.x() - floor(uv.x()), uv.y() - floor(uv.y())), std::min(ceil(uv.x()) - uv.x(), ceil(uv.y()) - uv.y()));
}

sibr::Vector2f T0(sibr::Vector2f uv)
{
	return sibr::Vector2f(floor(uv.x()) * 2, floor(uv.y()) * 2);
}

sibr::Vector2f T1(sibr::Vector2f uv)
{
	uv += sibr::Vector2f(0.5, 0.5);
	return sibr::Vector2f(floor(uv.x()) * 2 + 1, floor(uv.y()) * 2 + 1);
}

sibr::Vector2f rnd22(sibr::Vector2f p)
{
	sibr::Vector2f r(127.1 * p.x() + 311.7 * p.y(), 269.5 * p.x() + 183.3 * p.y());
	sibr::Vector2f temp(sin(r.x()) * 43758.5453, sin(r.y()) * 43758.5453);
	return sibr::Vector2f(temp.x() - floor(temp.x()), temp.y() - floor(temp.y()));
}

// Load the Gaussians from the given file.
template<int D>
int loadPly(const char* filename,
	std::vector<Pos>& pos,
	std::vector<SHs<3>>& shs,
	std::vector<float>& opacities,
	std::vector<Scale>& scales,
	std::vector<Rot>& rot,
	sibr::Vector3f& minn,
	sibr::Vector3f& maxx,
	std::vector<Rot>& rot_toLocal,
	bool synthesis = false,
	float scale_uv = 1.0f,
	std::string file_mesh = "")
{
	std::ifstream infile(filename, std::ios_base::binary);

	if (!infile.good())
		SIBR_ERR << "Unable to find model's PLY file, attempted:\n" << filename << std::endl;

	// "Parse" header (it has to be a specific format anyway)
	std::string buff;
	std::getline(infile, buff);
	std::getline(infile, buff);

	std::string dummy;
	std::getline(infile, buff);
	std::stringstream ss(buff);
	int count;
	ss >> dummy >> dummy >> count;

	// Output number of Gaussians contained
	SIBR_LOG << "Loading " << count << " Gaussian splats" << std::endl;

	while (std::getline(infile, buff))
		if (buff.compare("end_header") == 0)
			break;

	// Read all Gaussians at once (AoS)
	std::vector<RichPoint<D>> points(count);
	infile.read((char*)points.data(), count * sizeof(RichPoint<D>));

	// Resize our SoA data
	pos.resize(count);
	shs.resize(count);
	scales.resize(count);
	rot.resize(count);
	opacities.resize(count);

	// Gaussians are done training, they won't move anymore. Arrange
	// them according to 3D Morton order. This means better cache
	// behavior for reading Gaussians that end up in the same tile 
	// (close in 3D --> close in 2D).
	minn = sibr::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
	maxx = -minn;
	for (int i = 0; i < count; i++)
	{
		maxx = maxx.cwiseMax(points[i].pos);
		minn = minn.cwiseMin(points[i].pos);
	}
	std::vector<std::pair<uint64_t, int>> mapp(count);
	for (int i = 0; i < count; i++)
	{
		sibr::Vector3f rel = (points[i].pos - minn).array() / (maxx - minn).array();
		sibr::Vector3f scaled = ((float((1 << 21) - 1)) * rel);
		sibr::Vector3i xyz = scaled.cast<int>();

		uint64_t code = 0;
		for (int i = 0; i < 21; i++) {
			code |= ((uint64_t(xyz.x() & (1 << i))) << (2 * i + 0));
			code |= ((uint64_t(xyz.y() & (1 << i))) << (2 * i + 1));
			code |= ((uint64_t(xyz.z() & (1 << i))) << (2 * i + 2));
		}

		mapp[i].first = code;
		mapp[i].second = i;
	}
	auto sorter = [](const std::pair < uint64_t, int>& a, const std::pair < uint64_t, int>& b) {
		return a.first < b.first;
	};
	std::sort(mapp.begin(), mapp.end(), sorter);

	uint tex_res = 256;
	uint dst_res = floor(tex_res * scale_uv);
	uint z_res = 4;

	rot_toLocal.reserve(count);
	Rot q_identity;
	q_identity.rot[0] = 1;
	q_identity.rot[1] = 0;
	q_identity.rot[2] = 0;
	q_identity.rot[3] = 0;

	SHs<3> mean_shs;
	for (int shs_cp = 0; shs_cp < 48; shs_cp++)
	{
		mean_shs.shs[shs_cp] = 0;
	}

	// Move data from AoS to SoA
	int SH_N = (D + 1) * (D + 1);
	for (int k = 0; k < count; k++)
	{
		// int i = mapp[k].second;
		int i = k;
		pos[k] = points[i].pos;

		// Normalize quaternion
		float length2 = 0;
		for (int j = 0; j < 4; j++)
			length2 += points[i].rot.rot[j] * points[i].rot.rot[j];
		float length = sqrt(length2);
		for (int j = 0; j < 4; j++)
			rot[k].rot[j] = points[i].rot.rot[j] / length;

		// Exponentiate scale
		for(int j = 0; j < 3; j++)
			scales[k].scale[j] = exp(points[i].scale.scale[j]);

		// Activate alpha
		opacities[k] = sigmoid(points[i].opacity);

		shs[k].shs[0] = points[i].shs.shs[0];
		shs[k].shs[1] = points[i].shs.shs[1];
		shs[k].shs[2] = points[i].shs.shs[2];
		mean_shs.shs[0] += shs[k].shs[0];
		mean_shs.shs[1] += shs[k].shs[1];
		mean_shs.shs[2] += shs[k].shs[2];
		for (int j = 1; j < SH_N; j++)
		{
			shs[k].shs[j * 3 + 0] = points[i].shs.shs[(j - 1) + 3];
			shs[k].shs[j * 3 + 1] = points[i].shs.shs[(j - 1) + SH_N + 2];
			shs[k].shs[j * 3 + 2] = points[i].shs.shs[(j - 1) + 2 * SH_N + 1];
			mean_shs.shs[j * 3 + 0] += shs[k].shs[j * 3 + 0];
			mean_shs.shs[j * 3 + 1] += shs[k].shs[j * 3 + 1];
			mean_shs.shs[j * 3 + 2] += shs[k].shs[j * 3 + 2];
		}

		// Normalized Texture
		scaleGS(0.5, pos[k], scales[k]);
		sibr::Vector3f vec_trans = sibr::Vector3f(0.5, 0.5, 0);
		translateGS(vec_trans, pos[k]);
		
		if (synthesis)
		{
			uint x = (k / z_res) % tex_res;
			uint y = (k / z_res) / tex_res;
			sibr::Vector3f texel_center((x + 0.5) / tex_res, (y + 0.5) / tex_res, 0);
			pos[k] = pos[k] - texel_center;
		}
		
		rot_toLocal.push_back(q_identity);
	}

	for (int shs_cp = 0; shs_cp < 48; shs_cp++)
	{
		mean_shs.shs[shs_cp] /= float(count);
	}

	if (synthesis) // Synthesis
	{
		for (uint y = 0; y < dst_res; y++)
		{
			for (uint x = 0; x < dst_res; x++)
			{
				sibr::Vector2f cur_uv((x + 0.5) / tex_res, (y + 0.5) / tex_res);
				sibr::Vector3f b(B0(cur_uv), B1(cur_uv), 0);
				b /= b.sum();

				sibr::Vector2f t0 = T0(cur_uv);
				sibr::Vector2f t1 = T1(cur_uv);

				sibr::Vector2f uv0 = cur_uv - rnd22(t0);
				sibr::Vector2f uv1 = cur_uv - rnd22(t1);

				sibr::Vector2u tex_id0(floor((uv0.x() - floor(uv0.x())) * tex_res), floor((uv0.y() - floor(uv0.y())) * tex_res));
				sibr::Vector2u tex_id1(floor((uv1.x() - floor(uv1.x())) * tex_res), floor((uv1.y() - floor(uv1.y())) * tex_res));

				for (uint z = 0; z < z_res; z++)
				{
					uint gs_id0 = (tex_id0.x() + tex_id0.y() * tex_res) * z_res + z;
					uint gs_id1 = (tex_id1.x() + tex_id1.y() * tex_res) * z_res + z;

					Pos new_pos;
					Scale new_scale;
					Rot new_rot;
					SHs<3> new_shs;
					float new_opa;

					if (opacities[gs_id0] < 0.1 || opacities[gs_id1] < 0.1)
					{
						if (b.x() < b.y())
						{
							new_pos = pos[gs_id1];
							new_scale = scales[gs_id1];
							new_rot = rot[gs_id1];
							new_shs = shs[gs_id1];
							new_opa = opacities[gs_id1];
						}
						else
						{
							new_pos = pos[gs_id0];
							new_scale = scales[gs_id0];
							new_rot = rot[gs_id0];
							new_shs = shs[gs_id0];
							new_opa = opacities[gs_id0];
						}
					}
					else
					{
						new_pos = b.x() * pos[gs_id0] + b.y() * pos[gs_id1];
						for (int j = 0; j < 3; j++)
						{
							new_scale.scale[j] = b.x() * scales[gs_id0].scale[j] + b.y() * scales[gs_id1].scale[j];
						}
						for (int j = 0; j < 4; j++)
						{
							new_rot.rot[j] = b.x() * rot[gs_id0].rot[j] + b.y() * rot[gs_id1].rot[j];
						}
						for (int j = 0; j < 48; j++)
						{
							//new_shs.shs[j] = b.x() * shs[gs_id0].shs[j] + b.y() * shs[gs_id1].shs[j];
							new_shs.shs[j] = (b.x() * shs[gs_id0].shs[j] + b.y() * shs[gs_id1].shs[j] - mean_shs.shs[j]) / b.norm() + mean_shs.shs[j];
						}
						new_opa = b.x() * opacities[gs_id0] + b.y() * opacities[gs_id1];
					}
					new_pos += sibr::Vector3f(cur_uv.x(), cur_uv.y(), 0);

					scaleGS(float(tex_res) / float(dst_res), new_pos, new_scale);

					pos.push_back(new_pos);
					scales.push_back(new_scale);
					rot.push_back(new_rot);
					shs.push_back(new_shs);
					opacities.push_back(new_opa);
					rot_toLocal.push_back(q_identity);
				}
			}
		}
		pos.erase(pos.begin(), pos.begin() + count);
		scales.erase(scales.begin(), scales.begin() + count);
		rot.erase(rot.begin(), rot.begin() + count);
		shs.erase(shs.begin(), shs.begin() + count);
		opacities.erase(opacities.begin(), opacities.begin() + count);
		rot_toLocal.erase(rot_toLocal.begin(), rot_toLocal.begin() + count);
		count = pos.size();
		tex_res = dst_res;
	}

	if (file_mesh != "")
	{
		sibr::Mesh mesh_shape;
		bool flag = mesh_shape.load(file_mesh);

		if (flag)
		{
			sibr::Mesh::Triangles tri_vert = mesh_shape.triangles();
			sibr::Mesh::Vertices vert_pos = mesh_shape.vertices();
			sibr::Mesh::UVs vert_uv = mesh_shape.texCoords();
			sibr::Mesh::Normals vert_normal = mesh_shape.normals();

			for (int tri_id = 0; tri_id < tri_vert.size(); tri_id++)
			{
				sibr::Vector3f vert0_pos = vert_pos[tri_vert[tri_id](0)];
				sibr::Vector3f vert1_pos = vert_pos[tri_vert[tri_id](1)];
				sibr::Vector3f vert2_pos = vert_pos[tri_vert[tri_id](2)];

				sibr::Vector2f vert0_uv = vert_uv[tri_vert[tri_id](0)];
				sibr::Vector2f vert1_uv = vert_uv[tri_vert[tri_id](1)];
				sibr::Vector2f vert2_uv = vert_uv[tri_vert[tri_id](2)];

				sibr::Vector3f vert0_normal = vert_normal[tri_vert[tri_id](0)];
				sibr::Vector3f vert1_normal = vert_normal[tri_vert[tri_id](1)];
				sibr::Vector3f vert2_normal = vert_normal[tri_vert[tri_id](2)];

				sibr::Matrix3f tri_pos;
				tri_pos << vert0_pos(0), vert0_pos(1), vert0_pos(2),
					vert1_pos(0), vert1_pos(1), vert1_pos(2),
					vert2_pos(0), vert2_pos(1), vert2_pos(2);

				sibr::Matrix3f tri_uv;
				tri_uv << vert0_uv(0), vert0_uv(1), 0,
					vert1_uv(0), vert1_uv(1), 0,
					vert2_uv(0), vert2_uv(1), 0;

				sibr::Matrix3f tri_uv_bary;
				tri_uv_bary << vert0_uv(0), vert1_uv(0), vert2_uv(0),
					vert0_uv(1), vert1_uv(1), vert2_uv(1),
					1, 1, 1;

				//float pos_size = (vert0_pos - vert1_pos).norm();
				//float uv_size = (vert0_uv - vert1_uv).norm();
				float pos_size = (vert0_pos - vert1_pos).norm() +(vert0_pos - vert2_pos).norm() + (vert1_pos - vert2_pos).norm();
				float uv_size = (vert0_uv - vert1_uv).norm() +(vert0_uv - vert2_uv).norm() + (vert1_uv - vert2_uv).norm();
				float scale_size = pos_size / uv_size;
				//float pos_size = ((vert1_pos - vert0_pos).cross(vert2_pos - vert0_pos)).norm();
				//float uv_size = ((tri_uv.row(1) - tri_uv.row(0)).cross(tri_uv.row(2) - tri_uv.row(0))).norm();
				//float scale_size = sqrt(pos_size / uv_size);

				// sibr::Matrix3f tri_uv_scale = tri_uv * scale_size;
				// Eigen::Matrix3f matrix_rotate;
				// Eigen::Vector3f vector_trans;
				// computeMatrixRT(tri_uv_scale, tri_pos, matrix_rotate, vector_trans);

				sibr::Vector3f local_n(0, 0, 1);
				sibr::Vector3f local_t = (tri_uv.row(1) - tri_uv.row(0)).normalized().transpose();
				sibr::Vector3f local_b = local_t.cross(local_n);
				sibr::Matrix3f local_frame;
				local_frame << local_t(0), local_b(0), local_n(0),
					local_t(1), local_b(1), local_n(1),
					local_t(2), local_b(2), local_n(2);
				sibr::Vector3f world_n = vert0_normal.normalized();
				sibr::Vector3f world_t = (vert1_pos - vert0_pos).normalized();
				sibr::Vector3f world_b = world_t.cross(world_n);
				sibr::Matrix3f world_frame;
				world_frame << world_t(0), world_b(0), world_n(0),
					world_t(1), world_b(1), world_n(1),
					world_t(2), world_b(2), world_n(2);
				sibr::Matrix3f matrix_rotate = world_frame * local_frame.inverse();

				// sibr::Quaternionf q_toLocal(matrix_rotate.transpose()); // inverse == transpose
				sibr::Quaternionf q_toLocal(matrix_rotate); // inverse == transpose & for glm

				float max_u = std::max(std::max(tri_uv(0, 0), tri_uv(1, 0)), tri_uv(2, 0));
				float max_v = std::max(std::max(tri_uv(0, 1), tri_uv(1, 1)), tri_uv(2, 1));
				float min_u = std::min(std::min(tri_uv(0, 0), tri_uv(1, 0)), tri_uv(2, 0));
				float min_v = std::min(std::min(tri_uv(0, 1), tri_uv(1, 1)), tri_uv(2, 1));
				uint max_u_id = ceil(max_u * tex_res);
				uint max_v_id = ceil(max_v * tex_res);
				uint min_u_id = floor(min_u * tex_res);
				uint min_v_id = floor(min_v * tex_res);
				for (uint y = min_v_id; y < max_v_id; y++)
				{
					for (uint x = min_u_id; x < max_u_id; x++)
					{
						for (uint z = 0; z < z_res; z++)
						{
							uint cur_id = (y * tex_res + x) * z_res + z;
							sibr::Vector2f cur_pos = pos[cur_id].xy();
							if (pointInTriangle(cur_pos, vert0_uv, vert1_uv, vert2_uv))
							{
								sibr::Vector3f pos_bary(cur_pos.x(), cur_pos.y(), 1);
								sibr::Vector3f bary_centrics = tri_uv_bary.inverse() * pos_bary;
								sibr::Vector3f pos_world = tri_pos.transpose() * bary_centrics;
								sibr::Vector3f normal_world = bary_centrics(0) * vert0_normal + bary_centrics(1) * vert1_normal + bary_centrics(2) * vert2_normal;

								Pos new_pos = pos[cur_id];
								Scale new_scale = scales[cur_id];
								Rot new_rot = rot[cur_id];
								SHs<3> new_shs = shs[cur_id];
								scaleGS(scale_size, new_pos, new_scale);
								pos_world = pos_world + new_pos.z() * normal_world;
								rotateGS(matrix_rotate, new_pos, new_rot, new_shs);
								// translateGS(vector_trans, new_pos);
								new_pos = pos_world;
								pos.push_back(new_pos);
								scales.push_back(new_scale);
								rot.push_back(new_rot);
								shs.push_back(new_shs);
								opacities.push_back(opacities[cur_id]);

								Rot new_rot_toLocal;
								new_rot_toLocal.rot[0] = q_toLocal.w();
								new_rot_toLocal.rot[1] = q_toLocal.x();
								new_rot_toLocal.rot[2] = q_toLocal.y();
								new_rot_toLocal.rot[3] = q_toLocal.z();
								rot_toLocal.push_back(new_rot_toLocal);
							}
						}
					}
				}
			}
			pos.erase(pos.begin(), pos.begin() + count);
			scales.erase(scales.begin(), scales.begin() + count);
			rot.erase(rot.begin(), rot.begin() + count);
			shs.erase(shs.begin(), shs.begin() + count);
			opacities.erase(opacities.begin(), opacities.begin() + count);
			rot_toLocal.erase(rot_toLocal.begin(), rot_toLocal.begin() + count);
			count = pos.size();
		}
	}

	return count;
}

void savePly(const char* filename,
	const std::vector<Pos>& pos,
	const std::vector<SHs<3>>& shs,
	const std::vector<float>& opacities,
	const std::vector<Scale>& scales,
	const std::vector<Rot>& rot,
	const sibr::Vector3f& minn,
	const sibr::Vector3f& maxx)
{
	// Read all Gaussians at once (AoS)
	int count = 0;
	for (int i = 0; i < pos.size(); i++)
	{
		if (pos[i].x() < minn.x() || pos[i].y() < minn.y() || pos[i].z() < minn.z() ||
			pos[i].x() > maxx.x() || pos[i].y() > maxx.y() || pos[i].z() > maxx.z())
			continue;
		count++;
	}
	std::vector<RichPoint<3>> points(count);

	// Output number of Gaussians contained
	SIBR_LOG << "Saving " << count << " Gaussian splats" << std::endl;

	std::ofstream outfile(filename, std::ios_base::binary);

	outfile << "ply\nformat binary_little_endian 1.0\nelement vertex " << count << "\n";

	std::string props1[] = { "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"};
	std::string props2[] = { "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3" };

	for (auto s : props1)
		outfile << "property float " << s << std::endl;
	for (int i = 0; i < 45; i++)
		outfile << "property float f_rest_" << i << std::endl;
	for (auto s : props2)
		outfile << "property float " << s << std::endl;
	outfile << "end_header" << std::endl;

	count = 0;
	for (int i = 0; i < pos.size(); i++)
	{
		if (pos[i].x() < minn.x() || pos[i].y() < minn.y() || pos[i].z() < minn.z() ||
			pos[i].x() > maxx.x() || pos[i].y() > maxx.y() || pos[i].z() > maxx.z())
			continue;
		points[count].pos = pos[i];
		points[count].rot = rot[i];
		// Exponentiate scale
		for (int j = 0; j < 3; j++)
			points[count].scale.scale[j] = log(scales[i].scale[j]);
		// Activate alpha
		points[count].opacity = inverse_sigmoid(opacities[i]);
		points[count].shs.shs[0] = shs[i].shs[0];
		points[count].shs.shs[1] = shs[i].shs[1];
		points[count].shs.shs[2] = shs[i].shs[2];
		for (int j = 1; j < 16; j++)
		{
			points[count].shs.shs[(j - 1) + 3] = shs[i].shs[j * 3 + 0];
			points[count].shs.shs[(j - 1) + 18] = shs[i].shs[j * 3 + 1];
			points[count].shs.shs[(j - 1) + 33] = shs[i].shs[j * 3 + 2];
		}
		count++;
	}
	outfile.write((char*)points.data(), sizeof(RichPoint<3>) * points.size());
}

namespace sibr
{
	// A simple copy renderer class. Much like the original, but this one
	// reads from a buffer instead of a texture and blits the result to
	// a render target. 
	class BufferCopyRenderer
	{

	public:

		BufferCopyRenderer()
		{
			_shader.init("CopyShader",
				sibr::loadFile(sibr::getShadersDirectory("gaussian") + "/copy.vert"),
				sibr::loadFile(sibr::getShadersDirectory("gaussian") + "/copy.frag"));

			_flip.init(_shader, "flip");
			_width.init(_shader, "width");
			_height.init(_shader, "height");
		}

		void process(uint bufferID, IRenderTarget& dst, int width, int height, bool disableTest = true)
		{
			if (disableTest)
				glDisable(GL_DEPTH_TEST);
			else
				glEnable(GL_DEPTH_TEST);

			_shader.begin();
			_flip.send();
			_width.send();
			_height.send();

			dst.clear();
			dst.bind();

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferID);

			sibr::RenderUtility::renderScreenQuad();

			dst.unbind();
			_shader.end();
		}

		/** \return option to flip the texture when copying. */
		bool& flip() { return _flip.get(); }
		int& width() { return _width.get(); }
		int& height() { return _height.get(); }

	private:

		GLShader			_shader; 
		GLuniform<bool>		_flip = false; ///< Flip the texture when copying.
		GLuniform<int>		_width = 1000;
		GLuniform<int>		_height = 800;
	};
}

std::function<char* (size_t N)> resizeFunctional(void** ptr, size_t& S) {
	auto lambda = [ptr, &S](size_t N) {
		if (N > S)
		{
			if (*ptr)
				CUDA_SAFE_CALL(cudaFree(*ptr));
			CUDA_SAFE_CALL(cudaMalloc(ptr, 2 * N));
			S = 2 * N;
		}
		return reinterpret_cast<char*>(*ptr);
	};
	return lambda;
}

sibr::GaussianView::GaussianView(const sibr::BasicIBRScene::Ptr & ibrScene, uint render_w, uint render_h, const char* file, bool* messageRead, int sh_degree, bool white_bg, bool useInterop, int device) :
	_scene(ibrScene),
	_dontshow(messageRead),
	_sh_degree(sh_degree),
	sibr::ViewBase(render_w, render_h)
{
	int num_devices;
	CUDA_SAFE_CALL_ALWAYS(cudaGetDeviceCount(&num_devices));
	_device = device;
	if (device >= num_devices)
	{
		if (num_devices == 0)
			SIBR_ERR << "No CUDA devices detected!";
		else
			SIBR_ERR << "Provided device index exceeds number of available CUDA devices!";
	}
	CUDA_SAFE_CALL_ALWAYS(cudaSetDevice(device));
	cudaDeviceProp prop;
	CUDA_SAFE_CALL_ALWAYS(cudaGetDeviceProperties(&prop, device));
	if (prop.major < 7)
	{
		SIBR_ERR << "Sorry, need at least compute capability 7.0+!";
	}

	_pointbasedrenderer.reset(new PointBasedRenderer());
	_copyRenderer = new BufferCopyRenderer();
	_copyRenderer->flip() = true;
	_copyRenderer->width() = render_w;
	_copyRenderer->height() = render_h;

	std::vector<uint> imgs_ulr;
	const auto & cams = ibrScene->cameras()->inputCameras();
	for(size_t cid = 0; cid < cams.size(); ++cid) {
		if(cams[cid]->isActive()) {
			imgs_ulr.push_back(uint(cid));
		}
	}
	_scene->cameras()->debugFlagCameraAsUsed(imgs_ulr);

	so3_l1[0] << 0, 0, 0,
		0, 0, -1,
		0, 1, 0;
	so3_l1[1] << 0, 0, 1,
		0, 0, 0,
		-1, 0, 0;
	so3_l1[2] << 0, -1, 0,
		1, 0, 0,
		0, 0, 0;
	so3_l2[0] << 0, 1, 0, 0, 0,
		-1, 0, 0, 0, 0,
		0, 0, 0, -1.732051, 0,
		0, 0, 1.732051, 0, -1,
		0, 0, 0, 1, 0;
	so3_l2[1] << 0, 0, 0, 0, 2,
		0, 0, 0, 1, 0,
		0, 0, 0, 0, 0,
		0, -1, 0, 0, 0,
		-2, 0, 0, 0, 0;
	so3_l2[2] << 0, 0, 0, -1, 0,
		0, 0, -1.732051, 0, -1,
		0, 1.732051, 0, 0, 0,
		1, 0, 0, 0, 0,
		0, 1, 0, 0, 0;
	so3_l3[0] << 0, 1.224745, 0, 0, 0, 0, 0,
		-1.224745, 0, 1.581139, 0, 0, 0, 0,
		0, -1.581139, 0, 0, 0, 0, 0,
		0, 0, 0, 0, -2.449490, 0, 0,
		0, 0, 0, 2.449490, 0, -1.581139, 0,
		0, 0, 0, 0, 1.581139, 0, -1.224745,
		0, 0, 0, 0, 0, 1.224745, 0;
	so3_l3[1] << 0, 0, 0, 0, 0, 0, 3,
		0, 0, 0, 0, 0, 2, 0,
		0, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 0, 0, 0,
		0, 0, -1, 0, 0, 0, 0,
		0, -2, 0, 0, 0, 0, 0,
		-3, 0, 0, 0, 0, 0, 0;
	so3_l3[2] << 0, 0, 0, 0, 0, -1.224745, 0,
		0, 0, 0, 0, -1.581139, 0, -1.224745,
		0, 0, 0, -2.449490, 0, -1.581139, 0,
		0, 0, 2.449490, 0, 0, 0, 0,
		0, 1.581139, 0, 0, 0, 0, 0,
		1.224745, 0, 1.581139, 0, 0, 0, 0,
		0, 1.224745, 0, 0, 0, 0, 0;
	mat_P << 0, 0, 1,
		1, 0, 0,
		0, 1, 0;
	inv_P << 0, 1, 0,
		0, 0, 1,
		1, 0, 0;
	file_src = file;

	// Load the PLY data (AoS) to the GPU (SoA)
	std::vector<Pos> pos;
	std::vector<Rot> rot;
	std::vector<Scale> scale;
	std::vector<float> opacity;
	std::vector<SHs<3>> shs;
	std::vector<Rot> rot_toLocal;
	if (sh_degree == 0)
	{
		count = loadPly<0>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax, rot_toLocal);
	}
	else if (sh_degree == 1)
	{
		count = loadPly<1>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax, rot_toLocal);
	}
	else if (sh_degree == 2)
	{
		count = loadPly<2>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax, rot_toLocal);
	}
	else if (sh_degree == 3)
	{
		count = loadPly<3>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax, rot_toLocal, _synthesis, _scalingUV);
	}

	_boxmin = _scenemin;
	_boxmax = _scenemax;

	int P = count;

	// Allocate and fill the GPU data
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&pos_cuda, sizeof(Pos) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(pos_cuda, pos.data(), sizeof(Pos) * P, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rot_cuda, sizeof(Rot) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(rot_cuda, rot.data(), sizeof(Rot) * P, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&shs_cuda, sizeof(SHs<3>) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(shs_cuda, shs.data(), sizeof(SHs<3>) * P, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&opacity_cuda, sizeof(float) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(opacity_cuda, opacity.data(), sizeof(float) * P, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&scale_cuda, sizeof(Scale) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(scale_cuda, scale.data(), sizeof(Scale) * P, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rot_toLocal_cuda, sizeof(Rot) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(rot_toLocal_cuda, rot_toLocal.data(), sizeof(Rot) * P, cudaMemcpyHostToDevice));

	// Create space for view parameters
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&view_cuda, sizeof(sibr::Matrix4f)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&proj_cuda, sizeof(sibr::Matrix4f)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&cam_pos_cuda, 3 * sizeof(float)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&background_cuda, 3 * sizeof(float)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rect_cuda, 2 * P * sizeof(int)));

	float bg[3] = { white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f };
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(background_cuda, bg, 3 * sizeof(float), cudaMemcpyHostToDevice));

	gData = new GaussianData(P, 
		(float*)pos.data(),
		(float*)rot.data(),
		(float*)scale.data(),
		opacity.data(),
		(float*)shs.data());

	_gaussianRenderer = new GaussianSurfaceRenderer();

	// Create GL buffer ready for CUDA/GL interop
	glCreateBuffers(1, &imageBuffer);
	glNamedBufferStorage(imageBuffer, render_w * render_h * 3 * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT);

	if (useInterop)
	{
		if (cudaPeekAtLastError() != cudaSuccess)
		{
			SIBR_ERR << "A CUDA error occurred in setup:" << cudaGetErrorString(cudaGetLastError()) << ". Please rerun in Debug to find the exact line!";
		}
		cudaGraphicsGLRegisterBuffer(&imageBufferCuda, imageBuffer, cudaGraphicsRegisterFlagsWriteDiscard);
		useInterop &= (cudaGetLastError() == cudaSuccess);
	}
	if (!useInterop)
	{
		fallback_bytes.resize(render_w * render_h * 3 * sizeof(float));
		cudaMalloc(&fallbackBufferCuda, fallback_bytes.size());
		_interop_failed = true;
	}

	geomBufferFunc = resizeFunctional(&geomPtr, allocdGeom);
	binningBufferFunc = resizeFunctional(&binningPtr, allocdBinning);
	imgBufferFunc = resizeFunctional(&imgPtr, allocdImg);
}

void sibr::GaussianView::setScene(const sibr::BasicIBRScene::Ptr & newScene)
{
	_scene = newScene;

	// Tell the scene we are a priori using all active cameras.
	std::vector<uint> imgs_ulr;
	const auto & cams = newScene->cameras()->inputCameras();
	for (size_t cid = 0; cid < cams.size(); ++cid) {
		if (cams[cid]->isActive()) {
			imgs_ulr.push_back(uint(cid));
		}
	}
	_scene->cameras()->debugFlagCameraAsUsed(imgs_ulr);
}

void sibr::GaussianView::onRenderIBR(sibr::IRenderTarget & dst, const sibr::Camera & eye)
{
	if (currMode == "Ellipsoids")
	{
		_gaussianRenderer->process(count, *gData, eye, dst, 0.2f);
	}
	else if (currMode == "Initial Points")
	{
		_pointbasedrenderer->process(_scene->proxies()->proxy(), eye, dst);
	}
	else
	{
		// Convert view and projection to target coordinate system
		auto view_mat = eye.view();
		auto proj_mat = eye.viewproj();
		view_mat.row(1) *= -1;
		view_mat.row(2) *= -1;
		proj_mat.row(1) *= -1;

		// Compute additional view parameters
		float tan_fovy = tan(eye.fovy() * 0.5f);
		float tan_fovx = tan_fovy * eye.aspect();

		// Copy frame-dependent data to GPU
		CUDA_SAFE_CALL(cudaMemcpy(view_cuda, view_mat.data(), sizeof(sibr::Matrix4f), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(proj_cuda, proj_mat.data(), sizeof(sibr::Matrix4f), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(cam_pos_cuda, &eye.position(), sizeof(float) * 3, cudaMemcpyHostToDevice));

		float* image_cuda = nullptr;
		if (!_interop_failed)
		{
			// Map OpenGL buffer resource for use with CUDA
			size_t bytes;
			CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &imageBufferCuda));
			CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&image_cuda, &bytes, imageBufferCuda));
		}
		else
		{
			image_cuda = fallbackBufferCuda;
		}

		// Rasterize
		int* rects = _fastCulling ? rect_cuda : nullptr;
		float* boxmin = _cropping ? (float*)&_boxmin : nullptr;
		float* boxmax = _cropping ? (float*)&_boxmax : nullptr;
		CudaRasterizer::Rasterizer::forward(
			geomBufferFunc,
			binningBufferFunc,
			imgBufferFunc,
			count, _sh_degree, 16,
			background_cuda,
			_resolution.x(), _resolution.y(),
			pos_cuda,
			shs_cuda,
			nullptr,
			opacity_cuda,
			scale_cuda,
			_scalingModifier,
			rot_cuda,
			rot_toLocal_cuda,
			nullptr,
			view_cuda,
			proj_cuda,
			cam_pos_cuda,
			tan_fovx,
			tan_fovy,
			false,
			image_cuda,
			_antialiasing,
			nullptr,
			rects,
			boxmin,
			boxmax
		);

		if (!_interop_failed)
		{
			// Unmap OpenGL resource for use with OpenGL
			CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &imageBufferCuda));
		}
		else
		{
			CUDA_SAFE_CALL(cudaMemcpy(fallback_bytes.data(), fallbackBufferCuda, fallback_bytes.size(), cudaMemcpyDeviceToHost));
			glNamedBufferSubData(imageBuffer, 0, fallback_bytes.size(), fallback_bytes.data());
		}
		// Copy image contents to framebuffer
		_copyRenderer->process(imageBuffer, dst, _resolution.x(), _resolution.y());
	}

	if (cudaPeekAtLastError() != cudaSuccess)
	{
		SIBR_ERR << "A CUDA error occurred during rendering:" << cudaGetErrorString(cudaGetLastError()) << ". Please rerun in Debug to find the exact line!";
	}
}

void sibr::GaussianView::onUpdate(Input & input)
{
}

void sibr::GaussianView::onGUI()
{
	// Generate and update UI elements
	const std::string guiName = "3D Gaussians";
	if (ImGui::Begin(guiName.c_str())) 
	{
		if (ImGui::BeginCombo("Render Mode", currMode.c_str()))
		{
			if (ImGui::Selectable("Splats"))
				currMode = "Splats";
			if (ImGui::Selectable("Initial Points"))
				currMode = "Initial Points";
			if (ImGui::Selectable("Ellipsoids"))
				currMode = "Ellipsoids";
			ImGui::EndCombo();
		}
	}
	if (currMode == "Splats")
	{
		ImGui::SliderFloat("Scaling Modifier", &_scalingModifier, 0.001f, 1.0f);

		if (ImGui::Button("Change PLY")) {
			std::string selectedFile;
			if (sibr::showFilePicker(selectedFile, Default, "", "ply")) {
				if (!selectedFile.empty()) {
					file_src = selectedFile.c_str();
					SIBR_LOG << "Loaded ply (" << selectedFile << ")." << std::endl;
					flag_load = true;
				}
			}
		}
		if (ImGui::Button("Load Mesh")) {
			std::string selectedFile;
			if (sibr::showFilePicker(selectedFile, Default, "", "obj")) {
				if (!selectedFile.empty()) {
					mesh_src = selectedFile;
					SIBR_LOG << "Loaded mesh (" << selectedFile << ")." << std::endl;
					flag_load = true;
				}
			}
		}
		if (ImGui::Checkbox("Apply synthesis", &_synthesis) || ImGui::SliderFloat("Scaling UV", &_scalingUV, 0.1f, 10.f) || flag_load)
		{
			flag_load = false;

			std::vector<Pos> pos;
			std::vector<Rot> rot;
			std::vector<Scale> scale;
			std::vector<float> opacity;
			std::vector<SHs<3>> shs;
			std::vector<Rot> rot_toLocal;
			
			count = loadPly<3>(file_src, pos, shs, opacity, scale, rot, _scenemin, _scenemax, rot_toLocal, _synthesis, _scalingUV, mesh_src);

			int P = count;

			cudaFree(pos_cuda);
			cudaFree(rot_cuda);
			cudaFree(scale_cuda);
			cudaFree(opacity_cuda);
			cudaFree(shs_cuda);
			cudaFree(rot_toLocal_cuda);

			cudaFree(rect_cuda);

			// Allocate and fill the GPU data
			CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&pos_cuda, sizeof(Pos) * P));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(pos_cuda, pos.data(), sizeof(Pos) * P, cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rot_cuda, sizeof(Rot) * P));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(rot_cuda, rot.data(), sizeof(Rot) * P, cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&shs_cuda, sizeof(SHs<3>) * P));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(shs_cuda, shs.data(), sizeof(SHs<3>) * P, cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&opacity_cuda, sizeof(float) * P));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(opacity_cuda, opacity.data(), sizeof(float) * P, cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&scale_cuda, sizeof(Scale) * P));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(scale_cuda, scale.data(), sizeof(Scale) * P, cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rot_toLocal_cuda, sizeof(Rot) * P));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(rot_toLocal_cuda, rot_toLocal.data(), sizeof(Rot) * P, cudaMemcpyHostToDevice));

			// Create space for view parameters
			CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rect_cuda, 2 * P * sizeof(int)));

			// gData = new GaussianData(P,
			// 	(float*)pos.data(),
			// 	(float*)rot.data(),
			// 	(float*)scale.data(),
			// 	opacity.data(),
			// 	(float*)shs.data());
		}
	}
	ImGui::Checkbox("Fast culling", &_fastCulling);
	ImGui::Checkbox("Antialiasing", &_antialiasing);

	ImGui::Checkbox("Crop Box", &_cropping);
	if (_cropping)
	{
		ImGui::SliderFloat("Box Min X", &_boxmin.x(), _scenemin.x(), _scenemax.x());
		ImGui::SliderFloat("Box Min Y", &_boxmin.y(), _scenemin.y(), _scenemax.y());
		ImGui::SliderFloat("Box Min Z", &_boxmin.z(), _scenemin.z(), _scenemax.z());
		ImGui::SliderFloat("Box Max X", &_boxmax.x(), _scenemin.x(), _scenemax.x());
		ImGui::SliderFloat("Box Max Y", &_boxmax.y(), _scenemin.y(), _scenemax.y());
		ImGui::SliderFloat("Box Max Z", &_boxmax.z(), _scenemin.z(), _scenemax.z());
		ImGui::InputText("File", _buff, 512);
		if (ImGui::Button("Save"))
		{
			std::vector<Pos> pos(count);
			std::vector<Rot> rot(count);
			std::vector<float> opacity(count);
			std::vector<SHs<3>> shs(count);
			std::vector<Scale> scale(count);
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(pos.data(), pos_cuda, sizeof(Pos) * count, cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(rot.data(), rot_cuda, sizeof(Rot) * count, cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(opacity.data(), opacity_cuda, sizeof(float) * count, cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(shs.data(), shs_cuda, sizeof(SHs<3>) * count, cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(scale.data(), scale_cuda, sizeof(Scale) * count, cudaMemcpyDeviceToHost));
			savePly(_buff, pos, shs, opacity, scale, rot, _boxmin, _boxmax);
		}
	}

	ImGui::End();

	if(!*_dontshow && !accepted && _interop_failed)
		ImGui::OpenPopup("Error Using Interop");

	if (!*_dontshow && !accepted && _interop_failed && ImGui::BeginPopupModal("Error Using Interop", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
		ImGui::SetItemDefaultFocus();
		ImGui::SetWindowFontScale(2.0f);
		ImGui::Text("This application tries to use CUDA/OpenGL interop.\n"\
			" It did NOT work for your current configuration.\n"\
			" For highest performance, OpenGL and CUDA must run on the same\n"\
			" GPU on an OS that supports interop.You can try to pass a\n"\
			" non-zero index via --device on a multi-GPU system, and/or try\n" \
			" attaching the monitors to the main CUDA card.\n"\
			" On a laptop with one integrated and one dedicated GPU, you can try\n"\
			" to set the preferred GPU via your operating system.\n\n"\
			" FALLING BACK TO SLOWER RENDERING WITH CPU ROUNDTRIP\n");

		ImGui::Separator();

		if (ImGui::Button("  OK  ")) {
			ImGui::CloseCurrentPopup();
			accepted = true;
		}
		ImGui::SameLine();
		ImGui::Checkbox("Don't show this message again", _dontshow);
		ImGui::EndPopup();
	}
}

sibr::GaussianView::~GaussianView()
{
	// Cleanup
	cudaFree(pos_cuda);
	cudaFree(rot_cuda);
	cudaFree(scale_cuda);
	cudaFree(opacity_cuda);
	cudaFree(shs_cuda);
	cudaFree(rot_toLocal_cuda);

	cudaFree(view_cuda);
	cudaFree(proj_cuda);
	cudaFree(cam_pos_cuda);
	cudaFree(background_cuda);
	cudaFree(rect_cuda);

	if (!_interop_failed)
	{
		cudaGraphicsUnregisterResource(imageBufferCuda);
	}
	else
	{
		cudaFree(fallbackBufferCuda);
	}
	glDeleteBuffers(1, &imageBuffer);

	if (geomPtr)
		cudaFree(geomPtr);
	if (binningPtr)
		cudaFree(binningPtr);
	if (imgPtr)
		cudaFree(imgPtr);

	delete _copyRenderer;
}
