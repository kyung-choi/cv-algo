#include "pose_est_Ansar.h"
#include <pcl/common/distances.h>

namespace ky
{
	PoseEstAnsar::PoseEstAnsar(const float _fx, const float _fy, const float _ppx, const float _ppy) : 
		PoseEst(_fx, _fy, _ppx, _ppy) {}

	float PoseEstAnsar::estimatePose(const PointCloudXYZ& _model,	
		const PointCloudUV& _uv, Matrix4f& _transformation)
	{
		_computeM(_model, _uv);

		return 0;
	}

	void PoseEstAnsar::_computeM(const PointCloudXYZ& _model, const PointCloudUV& _uv)
	{
		auto n{ _model.size() };
		m_M = MatrixXf::Zero(n * (n - 1) / 2, n * (n + 1) / 2 + 1);

		for (auto k{ 0 }, i{ 0 }; i < n - 1; ++i)
		{
			Vector3f pi((_uv[i].u - m_ppx) / m_fx, (_uv[i].v - m_ppy) / m_fy, 1);
			for (auto j{ i + 1 }; j < n; ++j)
			{
				Vector3f pj((_uv[j].u - m_ppx) / m_fx, (_uv[j].v - m_ppy) / m_fy, 1);

				m_M(k, k) = -2 * pi.dot(pj);
				m_M(k, n * (n - 1) / 2 + i) = pi.dot(pi);
				m_M(k, n * (n - 1) / 2 + j) = pj.dot(pj);
				m_M(k, n * (n + 1) / 2) = -pcl::squaredEuclideanDistance(_model[i], _model[j]);
				++k;
			}
		}
	}
}
