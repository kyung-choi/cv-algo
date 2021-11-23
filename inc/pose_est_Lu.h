#pragma once

#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/common/eigen.h>
#include <pcl/common/distances.h>

using namespace Eigen;

namespace ky
{
	class PoseEstLu
	{
	public:
		typedef pcl::PointUV UV;
		typedef pcl::PointXYZ XYZ;
		typedef pcl::PointCloud<UV> PointCloudUV;
		typedef pcl::PointCloud<XYZ> PointCloudXYZ;

		PoseEstLu();

		virtual ~PoseEstLu() {}

		void setFocalLength(const float _fx, const float _fy)
		{
			m_fx = _fx;
			m_fy = _fy == 0 ? _fx : _fy;
		}

		void setPP(const float _ppx, const float _ppy)
		{
			m_ppx = _ppx;
			m_ppy = _ppy;
		}

		void setMaxIter(const int _maxIter)
		{
			m_maxIter = _maxIter;
		}

		void setConvergence(const float _convergence)
		{
			m_convergence = _convergence;
		}

		void estimatePose(const PointCloudXYZ& _model, const PointCloudUV& _uv, 
			Matrix4f& _transformation) const;

	private:
		float m_fx, m_fy, m_ppx, m_ppy;
		float m_convergence;
		int m_maxIter;

		void _computeV(const UV& _uv, Matrix3f& _V) const;

		void _computeT(const PointCloudXYZ& _p, const Matrix3f& _L,
			const std::vector<Matrix3f, aligned_allocator<Matrix3f>> _V,
			Matrix4f& _transformation) const;

		void _computeVq(const PointCloudXYZ& _p,
			const std::vector<Matrix3f, aligned_allocator<Matrix3f>> _V,
			PointCloudXYZ& _Vq, const Matrix4f& _transformation) const;

		Matrix3f _computeL(const std::vector<Matrix3f, aligned_allocator<Matrix3f>> _V) const;

		float _computeErr(const PointCloudXYZ& _p, const PointCloudXYZ& _Vq, 
			const Matrix4f& _transformation) const;
	};
}
