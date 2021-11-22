#include "pose_est_Lu.h"

namespace ky
{
	PoseEstLu::PoseEstLu() : m_fx(4500), m_fy(4500), m_ppx(512), m_ppy(512), m_maxIter(30), m_convergence(0.001f)
	{
	}

	void PoseEstLu::setFocalLength(const float _fx, const float _fy)
	{
		m_fx = _fx;
		m_fy = _fy == 0 ? _fx : _fy;
	}

	void PoseEstLu::setPP(const float _ppx, const float _ppy)
	{
		m_ppx = _ppx;
		m_ppy = _ppy;
	}

	void PoseEstLu::setMaxIter(const int _maxIter)
	{
		m_maxIter = _maxIter;
	}

	void PoseEstLu::setConvergence(const float _convergence)
	{
		m_convergence = _convergence;
	}

	void PoseEstLu::estimatePose(const PointCloudXYZ& _model, const PointCloudXY& _uv, Matrix4f& _transformation) const
	{
		auto iter{ 0 };
		auto convergence{ std::numeric_limits<float>::max() };
		auto err{ std::numeric_limits<float>::max() }, errNew{ 0.f };

		std::vector<Matrix3f, aligned_allocator<Matrix3f>> V(_uv.size(), Matrix3f::Identity());
		for (size_t i{ 0 }; i < _uv.size(); ++i)
			_projMat(_uv[i], V[i]);

		pcl::registration::TransformationEstimationSVD<XYZ, XYZ> te;
		PointCloudXYZ test, Vq;
		while (iter++ < m_maxIter && convergence > m_convergence)
		{
			_computeT(_model, V, _transformation);
			_computeVq(_model, V, Vq, _transformation);
			te.estimateRigidTransformation(_model, Vq, _transformation);
			pcl::transformPointCloud(_model, test, _transformation);

			errNew = 0;
			for (auto i{ 0 }; i < _model.size(); ++i)
				errNew += pcl::squaredEuclideanDistance(test[i], Vq[i]);

			convergence = std::abs(err - errNew);
			err = errNew;
		}
	}

	void PoseEstLu::_projMat(const XY& _uv, Matrix3f& _V) const
	{
		Vector3f v((_uv.x - m_ppx) / m_fx, (_uv.y - m_ppy) / m_fy, 1);
		_V = v * v.transpose() / v.norm();
	}

	void PoseEstLu::_computeT(const PointCloudXYZ& _p,
		const std::vector<Matrix3f, aligned_allocator<Matrix3f>> _V,
		Matrix4f& _transformation) const
	{
		size_t n{ _p.size() };
		Matrix3f I = Matrix3f::Identity();
		Matrix3f R = _transformation.block<3, 3>(0, 0);
		Matrix3f sV = Matrix3f::Zero();
		Vector3f rhs = Vector3f::Zero();

		for (size_t i{ 0 }; i < n; ++i)
		{
			sV += _V[i];
			rhs += (_V[i] - I) * R * Vector3f(_p[i].x, _p[i].y, _p[i].z);
		}

		sV /= n;
		Matrix3f Lhs = I - sV;

		_transformation.block<3, 1>(0, 3) = Lhs.inverse() * rhs / n;
	}

	void PoseEstLu::_computeVq(const PointCloudXYZ& _p,
		const std::vector<Matrix3f, aligned_allocator<Matrix3f>> _V,
		PointCloudXYZ& _Vq, const Matrix4f& _transformation) const
	{
		PointCloudXYZ q;
		pcl::transformPointCloud(_p, q, _transformation);
		auto qIter{ q.begin() };
		auto vIter{ _V.begin() };
		_Vq.clear();
		for (; qIter != q.end(); ++qIter, ++vIter)
		{
			auto w{ *vIter * Vector3f(qIter->x, qIter->y, qIter->z) };
			_Vq.push_back(XYZ(w(0), w(1), w(2)));
		}
	}


}
