#include "pose_est_Lu.h"

namespace ky
{
	PoseEstLu::PoseEstLu() : m_fx(4500), m_fy(4500), m_ppx(512), m_ppy(512), 
		m_maxIter(50), m_convergence(0.01)	{}

	float PoseEstLu::estimatePose(const PointCloudXYZ& _model, 
		const PointCloudUV& _uv, Matrix4f& _transformation)
	{
		auto iter{ 0 };
		auto convergence{ std::numeric_limits<float>::max() }, err{ std::numeric_limits<float>::max() };
		pcl::registration::TransformationEstimationSVD<XYZ, XYZ> te;
		PointCloudXYZ Vq;

  	_computeV(_uv);
		_computeL();

		while (iter++ < m_maxIter && convergence > m_convergence)
		{
			_computeT(_model, _transformation);
			_computeVq(_model, _transformation, Vq);
			te.estimateRigidTransformation(_model, Vq, _transformation);

			auto errNew{ _computeErr(_model, Vq, _transformation) };
			convergence = std::abs(err - errNew);
			err = errNew;
		}

		return err;
	}

	float PoseEstLu::_computeErr(const PointCloudXYZ& _p, const PointCloudXYZ& _Vq, 
		const Matrix4f& _transformation) const
	{
		float err{ 0 };
		PointCloudXYZ q;
		pcl::transformPointCloud(_p, q, _transformation);
		auto qIter{ q.begin() }; 
		auto VqIter{ _Vq.begin() };

		for (; qIter!= q.end(); ++qIter, ++VqIter)
			err += pcl::squaredEuclideanDistance(*qIter, *VqIter);

		return err;
	}

	void PoseEstLu::_computeV(const PointCloudUV& _uv)
	{
		for (const auto& uv : _uv.points)
		{
			Vector3f v((uv.u - m_ppx) / m_fx, (uv.v - m_ppy) / m_fy, 1);
			m_V.push_back(v * v.transpose() / (v(0) * v(0) + v(1) * v(1) + 1));
		}
	}

	void PoseEstLu::_computeT(const PointCloudXYZ& _p, Matrix4f& _transformation) const
	{
		Matrix3f I{ Matrix3f::Identity() }, R{ _transformation.block<3, 3>(0, 0) };
		Vector3f rhs{ Vector3f::Zero() };

		for (size_t i{ 0 }; i < _p.size(); ++i)
			rhs += (m_V[i] - I) * R * Vector3f(_p[i].x, _p[i].y, _p[i].z);

		_transformation.block<3, 1>(0, 3) = m_L * rhs;
	}

	void PoseEstLu::_computeVq(const PointCloudXYZ& _p, 
		const Matrix4f& _transformation, PointCloudXYZ& _Vq) const
	{
		PointCloudXYZ q;
		pcl::transformPointCloud(_p, q, _transformation);
		auto qIter{ q.begin() };
		auto VIter{ m_V.begin() };
		_Vq.clear();
		
		for (; qIter != q.end(); ++qIter, ++VIter)
		{
			auto w{ *VIter * Vector3f(qIter->x, qIter->y, qIter->z) };
			_Vq.push_back(XYZ(w(0), w(1), w(2)));
		}
	}

	void PoseEstLu::_computeL()
	{
		Matrix3f S = Matrix3f::Zero();

		for (auto i{ 0 }; i < m_V.size(); ++i)
			S += m_V[i];

		auto n{ static_cast<float>(m_V.size()) };
		m_L = (Matrix3f::Identity() - S / n).inverse() / n;
	}
}
