#pragma once

#include "pose_est.h"

using namespace Eigen;

namespace ky
{
	/**
  * @brief Iterative camera pose estimation. Implementation is based on "Fast and Globally Convergent Pose
	* Estimation from Video Images" - Lu, Hager, and Mjolsness
  */
	class PoseEstLu : public PoseEst
	{
	public:
		explicit PoseEstLu(const float _fx, const float _fy, const float _ppx, const float _ppy);

		virtual ~PoseEstLu() {}

		/** \brief Set the maximum iteration.
		 * \param[in] _maxIter The maximum iteration to perform before converged.
		 */
		void setMaxIter(const int _maxIter)
		{
			m_maxIter = _maxIter;
		}

		/** \brief Set convergence value.
		 * \param[in] _convergence Iteration terminates if change of error is less than convergence TH.
		 */
		void setConvergence(const float _convergence)
		{
			m_convergence = _convergence;
		}

		/** \brief Estimate camera pose.
		 * \param[in] _model A set of 3d points of an object in model frame.
		 * \param[in] _uv Corresponding image points of model.
		 * \param[in, out] _transformation Initial guess of camera pose transformation.
		 * \return Sum of squared orthogonal projection error.
		 */
		float estimatePose(const PointCloudXYZ& _model, const PointCloudUV& _uv,
			Matrix4f& _transformation);

	private:
		float m_convergence;
		int m_maxIter;
		std::vector<Matrix3f, aligned_allocator<Matrix3f>> m_V;  // Array of projection matrice.
		Matrix3f m_L;  // A pre-computed constant matrix in t(R)

		/** \brief Compute projection matrix.
		 * \param[in] _uv Detected image points of model.
		 */
		void _computeV(const PointCloudUV& _uv);

		/** \brief Compute translation using rotation matrix, t(R).
		 * \param[in] _p A set of 3d points of an object in model frame.
		 * \param[in, out] _transformation 4x4 camera pose matrix to be updated.
		 */
		void _computeT(const PointCloudXYZ& _p, Matrix4f& _transformation) const;

		void _computeVq(const PointCloudXYZ& _p, const Matrix4f& _transformation,
			PointCloudXYZ& _Vq) const;

		/** \brief Pre-compute a matrix, (I - 1/n*Sum(V))^(-1), to be used in t(R).
		*/
		void _computeL();

		/** \brief Computes the sum of squared orthogonal projection error.
		 * \param[in] _p A set of 3d points of an object in model frame.
		 * \param[in] _Vq A set of 3d point of an object, which is transformed and projected.
		 * \param[in] _transformation 4x4 camera pose matrix.
		*/
		float _computeErr(const PointCloudXYZ& _p, const PointCloudXYZ& _Vq,
			const Matrix4f& _transformation) const;
	};
}
