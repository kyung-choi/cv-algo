#pragma once

#include <pcl/common/eigen.h>
#include <pcl/pcl_base.h>
#include <pcl/point_types.h>

using namespace Eigen;

namespace ky
{
	/**
  * @brief Point based camera pose estimation. Implementation is based on 
	* "Linear Pose Estimation from Points or Lines " - Ansar and Daniilidis
  */
	class PoseEstAnsar
	{
	public:
		PoseEstAnsar();

		virtual ~PoseEstAnsar() {}

		/** \brief Set camera focal length in pixel unit.
		 * \param[in] _fx Focal length in x-axis using pixel unit.
		 * \param[in] _fx Focal length in y-axis using pixel unit.
		 */
		void setFocalLength(const float _fx, const float _fy = 0)
		{
			m_fx = _fx;
			m_fy = _fy == 0 ? _fx : _fy;
		}

		/** \brief Set principal point.
		 * \param[in] _ppx Prinicipal point in x-axis using pixel unit.
		 * \param[in] _ppx Prinicipal point in y-axis using pixel unit.
		 */
		void setPP(const float _ppx, const float _ppy)
		{
			m_ppx = _ppx;
			m_ppy = _ppy;
		}

		/** \brief Estimate camera pose.
		 * \param[in] _model A set of 3d points of an object in model frame.
		 * \param[in] _uv Corresponding image points of model.
		 * \param[in, out] _transformation Initial guess of camera pose transformation.
		 * \return Sum of squared orthogonal projection error.
		 */
		float estimatePose(const pcl::PointCloud<pcl::PointXYZ>& _model, 
			const pcl::PointCloud<pcl::PointUV>& _uv,	Matrix4f& _transformation);

	private:
		float m_fx, m_fy, m_ppx, m_ppy;
	};
}
