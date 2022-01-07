#pragma once
#include "pose_est.h"

namespace ky
{
	/**
  * @brief Point based camera pose estimation. Implementation is based on 
	* "Linear Pose Estimation from Points or Lines " - Ansar and Daniilidis
  */
	class PoseEstAnsar : public PoseEst
	{
	public:
		explicit PoseEstAnsar(const float _fx, const float _fy, const float _ppx, const float _ppy);

		virtual ~PoseEstAnsar() {}

		/** \brief Estimate camera pose.
		 * \param[in] _model A set of 3d points of an object in model frame.
		 * \param[in] _uv Corresponding image points of model.
		 * \param[in, out] _transformation Initial guess of camera pose transformation.
		 * \return Sum of squared orthogonal projection error.
		 */
		float estimatePose(const PointCloudXYZ& _model, 
			const PointCloudUV& _uv,	Matrix4f& _transformation);

	private:
		MatrixXf m_M, m_K, m_V;
		VectorXf m_lambda;

		void _computeM(const PointCloudXYZ& _model,	const PointCloudUV& _uv);

		void _computeK(const size_t _npts);

		void _computeV(const size_t _npts);

		void _computeLambda();

		void _computeDepth(const PointCloudUV& _uv, PointCloudXYZ& _depth);
	};
}
