#include "pose_est_Ansar.h"
#include <pcl/common/distances.h>
#include <map>

namespace ky
{
	PoseEstAnsar::PoseEstAnsar(const float _fx, const float _fy, const float _ppx, const float _ppy) : 
		PoseEst(_fx, _fy, _ppx, _ppy) {}

	float PoseEstAnsar::estimatePose(const PointCloudXYZ& _model,	
		const PointCloudUV& _uv, Matrix4f& _transformation)
	{
		_computeM(_model, _uv);

		_computeK(_uv.size());


		return 0;
	}

	void PoseEstAnsar::_computeM(const PointCloudXYZ& _model, const PointCloudUV& _uv)
	{
		auto n{ _model.size() };  // n >= 4
		m_M = MatrixXf::Zero(n * (n - 1) / 2, n * (n + 1) / 2 + 1);

		for (auto k{ 0 }, i{ 0 }; i < n - 1; ++i)
		{
			Vector3f pi((_uv[i].u - m_ppx) / m_fx, (_uv[i].v - m_ppy) / m_fy, 1);
			for (auto j{ i + 1 }; j < n; ++j)
			{
				Vector3f pj((_uv[j].u - m_ppx) / m_fx, (_uv[j].v - m_ppy) / m_fy, 1);

				m_M(k, k) = -2 * pi.dot(pj);
				m_M(k, n * (n - 1) / 2 + i) = pi.dot(pi);
				m_M(k, n * (n - 1) / 2 + j) = pj.dot(pj);
				auto dx{ _model[i].x - _model[j].x };
				auto dy{ _model[i].y - _model[j].y };
				auto dz{ _model[i].z - _model[j].z };
				dx /= 100.f;
				dy /= 100.f;
				dz /= 100.f;
				m_M(k, n * (n + 1) / 2) = -(dx*dx + dy*dy + dz*dz);
				++k;
			}
		}
	}

	void PoseEstAnsar::_computeK(const size_t n)
	{
		Eigen::FullPivHouseholderQR<MatrixXf> qr;
		qr.compute(m_M.transpose());
		MatrixXf Q{ qr.matrixQ() };
		size_t N{ n + 1 };
		MatrixXf V{ Q.block(0, Q.cols() - N, Q.rows(), N) };  // V: Ker(M)
		m_K = MatrixXf(N * (N + 1) / 2, N);  

		std::vector<std::tuple<size_t, size_t, size_t> > ijk;
		for (size_t i{ 0 }; i < n; ++i)
			for (size_t j{ 0 }; j < n - 1; ++j)
				for (size_t k{ j + 1 }; k < n; ++k)
					if (i != j && i != k)
						ijk.emplace_back(i, j, k);

		for (size_t i{ 0 }; i < n; ++i)
			for (size_t j{ 0 }; j < n; ++j)
				if (i != j)
					ijk.emplace_back(i, j, j);

		size_t row{ 0 }, col{ 0 };
		std::map<std::tuple<size_t, size_t>, size_t> x;
		for (size_t i{ 0 }; i < n - 1; ++i)
			for (size_t j{ i + 1 }; j < n; ++j)
				x.insert(std::pair<std::tuple<size_t, size_t>, size_t>(std::make_tuple(i, j), row++));

		for (size_t i{ 0 }; i < n; ++i)
			x.insert(std::pair<std::tuple<size_t, size_t>, size_t>(std::make_tuple(i, i), row++));

		for (const auto& idx : ijk)
		{
			size_t i{ std::get<0>(idx) }, j{ std::get<1>(idx) }, k{ std::get<2>(idx) };
			std::tuple<size_t, size_t> ij{ std::make_tuple(i, i) };
			std::tuple<size_t, size_t> kl{ std::make_tuple(j, k) };
			std::tuple<size_t, size_t> _ij{ i > j ? std::make_tuple(j, i) : std::make_tuple(i, j) };
			std::tuple<size_t, size_t> _kl{ i > k ? std::make_tuple(k, i) : std::make_tuple(i, k) };

			row = 0;
			for (size_t a{ 0 }; a < N - 1; ++a)
				for (size_t b{ a + 1 }; b < N; ++b)
					m_K(row++, col) = 2 * (V(x.at(ij), a) * V(x.at(kl), b) - V(x.at(_ij), a) * V(x.at(_kl), b));

			for (size_t a{ 0 }; a < N; ++a)
				m_K(row++, col) = V(x.at(ij), a) * V(x.at(kl), a) - V(x.at(_ij), a) * V(x.at(_kl), a);

			if (++col == N)
				break;
		}

		// TO-DO: revise contructing K.
		JacobiSVD<MatrixXf> svd(m_K, ComputeThinU | ComputeThinV);
		VectorXf lambda = svd.matrixV().col(N - 1);
	}
}
