#pragma once
#include "pose_est.h"

namespace ky
{
	/**
	* @brief Point based camera pose estimation. Implementation is based on
	* "Generalized Linear Pose Estimation " - A.Ess et al
	* @note Unlike the paper, 5+ points non-coplanar case is implemented.
	* Planar case could be handled differently. Ansar's method is advised for 4 point case . 
	*/
	class PoseEstEss : public PoseEst
	{
	public:
		explicit PoseEstEss(const float _fx, const float _fy, const float _ppx, const float _ppy);

		virtual ~PoseEstEss() {}

		/** \brief Estimate camera pose.
		 * \param[in] _model A set of 3d points of an object in model frame.
		 * \param[in] _uv Corresponding image points of model.
		 * \param[in, out] _transformation Initial guess of camera pose transformation.
		 * \return Sum of squared orthogonal projection error.
		 */
		float estimatePose(const PointCloudXYZ& _model,
			const PointCloudUV& _uv, Matrix4f& _transformation);

	private:
		MatrixXf m_A, m_B, m_X;

		void _computeABX(const PointCloudXYZ& _model, const PointCloudUV& _uv);
		void _5Points(VectorXf& _r);
		void _6plusPoints(VectorXf& _r);
		float _computeError(const PointCloudXYZ& _model, const PointCloudUV& _uv, const Matrix4f& _transformation);

	};
}
