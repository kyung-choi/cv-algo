#include "pose_est_Ansar.h"

namespace ky
{
	PoseEstAnsar::PoseEstAnsar() : m_fx(4500), m_fy(4500), m_ppx(512), m_ppy(512) {}

	float PoseEstAnsar::estimatePose(const pcl::PointCloud<pcl::PointXYZ>& _model,
		const pcl::PointCloud<pcl::PointUV>& _uv, Matrix4f& _transformation)
	{
		return 0;
	}
}
