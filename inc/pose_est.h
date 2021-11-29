#pragma once

#include <pcl/common/eigen.h>
#include <pcl/pcl_base.h>
#include <pcl/point_types.h>

using namespace Eigen;

typedef pcl::PointUV UV;
typedef pcl::PointXYZ XYZ;
typedef pcl::PointCloud<UV> PointCloudUV;
typedef pcl::PointCloud<XYZ> PointCloudXYZ;

namespace ky
{
	class PoseEst
	{
	public:
		PoseEst(const float _fx, const float _fy, const float _ppx, const float _ppy);

		virtual ~PoseEst() {}

		/** \brief Estimate camera pose.
		 * \param[in] _model A set of 3d points of an object in model frame.
		 * \param[in] _uv Corresponding image points of model.
		 * \param[in, out] _transformation Initial guess of camera pose transformation.
		 * \return Sum of squared orthogonal projection error.
		 */
		virtual float estimatePose(const PointCloudXYZ& _model,
			const PointCloudUV& _uv, Matrix4f& _transformation) = 0;

	protected:
		float m_fx, m_fy, m_ppx, m_ppy;
	};
}
