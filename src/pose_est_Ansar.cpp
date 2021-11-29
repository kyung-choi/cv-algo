#include "pose_est_Ansar.h"

namespace ky
{
	PoseEstAnsar::PoseEstAnsar(const float _fx, const float _fy, const float _ppx, const float _ppy) : 
		PoseEst(_fx, _fy, _ppx, _ppy) {}

	float PoseEstAnsar::estimatePose(const PointCloudXYZ& _model,	
		const PointCloudUV& _uv, Matrix4f& _transformation)
	{
		return 0;
	}
}
