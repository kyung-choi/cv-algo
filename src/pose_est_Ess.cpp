#include "pose_est_Ess.h"
#include <pcl/common/distances.h>
#include <map>

namespace ky
{
	PoseEstEss::PoseEstEss(const float _fx, const float _fy, const float _ppx, const float _ppy) :
		PoseEst(_fx, _fy, _ppx, _ppy) {}

	float PoseEstEss::estimatePose(const PointCloudXYZ& _model,
		const PointCloudUV& _uv, Matrix4f& _transformation)
	{
		_computeABX(_model, _uv);

		VectorXf r(9);
		if (_uv.size() == 5)
			_5Points(r);
		else if (_uv.size() > 5)
			_6plusPoints(r);
		else
			throw std::runtime_error("Number of input points > 4");

		Matrix3f R;
		R << r(0), r(1), r(2), r(3), r(4), r(5), r(6), r(7), r(8);
		R.transposeInPlace();
		JacobiSVD<MatrixXf> svd(R, ComputeThinU | ComputeThinV);
		R = svd.matrixU() * svd.matrixV().transpose();

		r << R(0, 0), R(1, 0), R(2, 0), R(0, 1), R(1, 1), R(2, 1), R(0, 2), R(1, 2), R(2, 2);
		MatrixXf P{ m_B.completeOrthogonalDecomposition().pseudoInverse() };
		MatrixXf t{ -P * m_A * r };

		_transformation = Matrix4f::Identity();
		_transformation.block(0, 0, 3, 3) = R;
		_transformation.block(0, 3, 3, 1) = t;

		std::cout << _transformation << std::endl;

		return 0;
	}

	void PoseEstEss::_computeABX(const PointCloudXYZ& _model, const PointCloudUV& _uv)
	{
		auto n{ static_cast<int>(_model.size()) };
		m_A = MatrixXf::Zero(n * 3, 9);
		m_B = MatrixXf::Zero(n * 3, 3);
		MatrixXf q(3, 1);
		for (auto i{ 0 }; i < n; ++i)
		{
			q << (_uv[i].u - m_ppx) / m_fx, (_uv[i].v - m_ppy) / m_fy, 1.;
			MatrixXf Q = MatrixXf::Identity(3, 3) - (q * q.transpose() / std::pow(q.norm(), 2));

			m_B.block(3 * i, 0, 3, 3) = Q;
			auto x{ _model[i].x }, y{ _model[i].y }, z{ _model[i].z };
			for (auto j{ 0 }; j < 3; ++j)
			{
				m_A(3 * i + j, 0) = x * Q(0, j);
				m_A(3 * i + j, 1) = x * Q(1, j);
				m_A(3 * i + j, 2) = x * Q(2, j);
				m_A(3 * i + j, 3) = y * Q(0, j);
				m_A(3 * i + j, 4) = y * Q(1, j);
				m_A(3 * i + j, 5) = y * Q(2, j);
				m_A(3 * i + j, 6) = z * Q(0, j);
				m_A(3 * i + j, 7) = z * Q(1, j);
				m_A(3 * i + j, 8) = z * Q(2, j);
			}
		}

		JacobiSVD<MatrixXf> svd(m_B, ComputeThinU | ComputeThinV);
		MatrixXf U{ svd.matrixU() };
		m_X = m_A - U * U.transpose() * m_A;
	}

	void PoseEstEss::_6plusPoints(VectorXf& _r)
	{
		JacobiSVD<MatrixXf> svd(m_X, ComputeThinU | ComputeThinV);
		MatrixXf V{ svd.matrixV() };
		_r = V.col(V.cols() - 1);

		Vector3f r1(3), r2(3), r3(3);
		r1 << _r(0), _r(1), _r(2); r1 *= std::sqrt(3);
		r2 << _r(3), _r(4), _r(5); r2 *= std::sqrt(3);
		r3 = r1.cross(r2);
		r3 /= r3.norm();

		_r << r1(0), r1(1), r1(2), r2(0), r2(1), r2(2), r3(0), r3(1), r3(2);
	}

	void PoseEstEss::_5Points(VectorXf& _r)
	{
		Eigen::FullPivHouseholderQR<MatrixXf> qr;
		qr.compute(m_X.transpose());
		MatrixXf Q{ qr.matrixQ() };
		VectorXf r1{ Q.col(Q.cols() - 2) };  // Dimension of null space is 2 (= 12 - n*2), n = 5.
		VectorXf r2{ Q.col(Q.cols() - 1) };  //
		MatrixXf L(Q.rows(), 3);  // A quadratic constraint yields 3 coefficient multipliers to estimate.

		for (auto i{ 0 }; i < L.rows(); ++i)
		{
			L(i, 0) = r1(i) * r1(i);
			L(i, 1) = r1(i) * r2(i) * 2;
			L(i, 2) = r2(i) * r2(i);
		}

		VectorXf b(9);  // Vectorized identity matrix.
		b << 1, 0, 0, 0, 1, 0, 0, 0, 1;

		// finding coefficients of null space basis using LS.
		VectorXf a{ L.bdcSvd(ComputeThinU | ComputeThinV).solve(b) };

		// resolving sign.
		VectorXf r(r1.size());
		if (a(1) > 0)
			r = r1 * std::sqrt(a(0)) + r2 * std::sqrt(a(2));
		else
			r = r1 * std::sqrt(a(0)) - r2 * std::sqrt(a(2));

		MatrixXf R(3, 3);
		R << r(0), r(1), r(2), r(3), r(4), r(5), r(6), r(7), r(8);

		_r = (R.determinant() < 0) ? -r : r;
	}
}
