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
		m_K = MatrixXf(n * n * (n - 1) / 2, (n + 1) * (n + 2) / 2);

		std::vector<std::vector<size_t> > iijk, ijik;
		for (size_t i{ 0 }; i < n; ++i)
			for (size_t j{ 0 }; j < n; ++j)
				if (i != j)
					for (size_t k{ j }; k < n; ++k)
						if (i != k)
						{
							iijk.emplace_back(std::vector<size_t>{i, i, j, k});
							std::vector<size_t> idx{ i, j, i, k };
							if (i > j) { std::swap(idx[0], idx[1]); }
							if (i > k) { std::swap(idx[2], idx[3]); }
							ijik.push_back(idx);
						}

		size_t row{ 0 };
		std::map<std::tuple<size_t, size_t>, size_t> x;
		for (size_t i{ 0 }; i < n - 1; ++i)
			for (size_t j{ i + 1 }; j < n; ++j)
				x.insert(std::pair<std::tuple<size_t, size_t>, size_t>(std::make_tuple(i, j), row++));

		for (size_t i{ 0 }; i < n; ++i)
			x.insert(std::pair<std::tuple<size_t, size_t>, size_t>(std::make_tuple(i, i), row++));

		row = 0;
		for (; row < m_K.rows(); ++row)
		{
			std::tuple<size_t, size_t> ij{ std::make_tuple(iijk[row][0], iijk[row][1]) };
			std::tuple<size_t, size_t> kl{ std::make_tuple(iijk[row][2], iijk[row][3]) };
			std::tuple<size_t, size_t> pq{ std::make_tuple(ijik[row][0], ijik[row][1]) };
			std::tuple<size_t, size_t> rs{ std::make_tuple(ijik[row][2], ijik[row][3]) };

			for (size_t c{ 0 }, a{ 0 }; a < N; ++a)
				for (size_t b{ a }; b < N; ++b)
					if (a == b)
						m_K(row, c++) = (V(x.at(ij), a) * V(x.at(kl), b) - V(x.at(pq), a) * V(x.at(rs), b));
					else
						m_K(row, c++) = (V(x.at(ij), a) * V(x.at(kl), b) - V(x.at(pq), a) * V(x.at(rs), b)) +	
						(V(x.at(kl), a) * V(x.at(ij), b) - V(x.at(rs), a) * V(x.at(pq), b));
		}
	}
}
