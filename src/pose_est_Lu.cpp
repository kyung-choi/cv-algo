#include "pose_est_Lu.h"

namespace ky
{
	PoseEstLu::PoseEstLu() : m_fx(4500), m_fy(4500), m_ppx(512), m_ppy(512), 
		m_maxIter(50), m_convergence(0.01)	{}

	void PoseEstLu::estimatePose(const PointCloudXYZ& _model, 
		const PointCloudUV& _uv, Matrix4f& _transformation) const
	{
		auto iter{ 0 };
		auto convergence{ std::numeric_limits<float>::max() };
		auto err{ std::numeric_limits<float>::max() };

		pcl::registration::TransformationEstimationSVD<XYZ, XYZ> te;
		std::vector<Matrix3f, aligned_allocator<Matrix3f>> V(_uv.size(), Matrix3f::Identity());

		for (auto i{ 0 }; i < _uv.size(); ++i)
			_computeV(_uv[i], V[i]);

		Matrix3f L{ _computeL(V) };
		PointCloudXYZ Vq;

		while (iter++ < m_maxIter && convergence > m_convergence)
		{
			_computeT(_model, L, V, _transformation);
			_computeVq(_model, V, Vq, _transformation);
			te.estimateRigidTransformation(_model, Vq, _transformation);

			auto errNew{ _computeErr(_model, Vq, _transformation) };
			convergence = std::abs(err - errNew);
			err = errNew;
		}
	}

	float PoseEstLu::_computeErr(const PointCloudXYZ& _p, const PointCloudXYZ& _Vq, 
		const Matrix4f& _transformation) const
	{
		float err{ 0 };
		PointCloudXYZ q;
		pcl::transformPointCloud(_p, q, _transformation);

		for (auto i{ 0 }; i < _p.size(); ++i)
			err += pcl::squaredEuclideanDistance(q[i], _Vq[i]);

		return err;
	}

	void PoseEstLu::_computeV(const UV& _uv, Matrix3f& _V) const
	{
		Vector3f v((_uv.u - m_ppx) / m_fx, (_uv.v - m_ppy) / m_fy, 1);
		_V = v * v.transpose() / (v(0) * v(0) + v(1) * v(1) + 1);
	}

	void PoseEstLu::_computeT(const PointCloudXYZ& _p, const Matrix3f& _L,
		const std::vector<Matrix3f, aligned_allocator<Matrix3f>> _V,
		Matrix4f& _transformation) const
	{
		Matrix3f I{ Matrix3f::Identity() }, R{ _transformation.block<3, 3>(0, 0) };
		Vector3f rhs{ Vector3f::Zero() };

		for (size_t i{ 0 }; i < _p.size(); ++i)
			rhs += (_V[i] - I) * R * Vector3f(_p[i].x, _p[i].y, _p[i].z);

		_transformation.block<3, 1>(0, 3) = _L * rhs;
	}

	void PoseEstLu::_computeVq(const PointCloudXYZ& _p,
		const std::vector<Matrix3f, aligned_allocator<Matrix3f>> _V,
		PointCloudXYZ& _Vq, const Matrix4f& _transformation) const
	{
		PointCloudXYZ q;
		pcl::transformPointCloud(_p, q, _transformation);
		auto qIter{ q.begin() };
		auto VIter{ _V.begin() };
		_Vq.clear();
		
		for (; qIter != q.end(); ++qIter, ++VIter)
		{
			auto w{ *VIter * Vector3f(qIter->x, qIter->y, qIter->z) };
			_Vq.push_back(XYZ(w(0), w(1), w(2)));
		}
	}

	Eigen::Matrix3f PoseEstLu::_computeL(const std::vector<Matrix3f, 
		aligned_allocator<Matrix3f>> _V) const
	{
		Matrix3f S = Matrix3f::Zero();

		for (auto i{ 0 }; i < _V.size(); ++i)
			S += _V[i];

		auto n{ static_cast<float>(_V.size()) };
		return (Matrix3f::Identity() - S / n).inverse() / n;
	}
}
