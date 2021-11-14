#include "tsai.h"

namespace ky
{
	Tsai::Tsai(const std::vector<cv::Point2f>& _uv, const std::vector<cv::Point3f>& _xyz, const cv::Size& _image_size) :
		CamCalib(_uv, _xyz, _image_size) {}

	bool Tsai::Calibrate()
	{
		assert(m_uv.cols >= static_cast<int>(minPts));

		_RTxTy();
		_FTz();

		return true;
	}

	void Tsai::_FTz()
	{
		const double x0{ m_imageSize.width * 0.5 }, y0{ m_imageSize.height * 0.5 };
		cv::Mat_<double> A(2 * m_uv.cols, 2), b(2 * m_uv.cols, 1), x(2, 1);

		for (int32_t i = 0; i < m_uv.cols; ++i)
		{
			auto u{ m_uv(0, i) - x0 };
			auto v{ m_uv(1, i) - y0 };
			cv::Mat w{ (cv::Mat_<double>(1, 3) << m_xyz(0, i), m_xyz(1, i), m_xyz(2, i)) };
			A(2 * i, 0) = m_hScale * (m_R.row(0).dot(w) + m_t(0));
			A(2 * i, 1) = -u;
			A(2 * i + 1, 0) = m_R.row(1).dot(w) + m_t(1);
			A(2 * i + 1, 1) = -v;
			b(2 * i, 0) = m_R.row(2).dot(w) * u; 
			b(2 * i + 1, 0) = m_R.row(2).dot(w) * v;
		}

		cv::solve(A, b, x, cv::DECOMP_SVD);

		if (x(0) < 0) // Resolve sign of RTxy.
		{
			m_R = -m_R;
			m_R.row(2) = -m_R.row(2);
			m_t = -m_t;
			m_t(2) = -m_t(2);
			x(0) = -x(0);
		}

		m_K(0, 0) = m_K(1, 1) = x(0);
		m_K(0, 2) = x0;
		m_K(1, 2) = y0;
		m_t(2) = x(1);
	}

	void Tsai::_RTxTy()
	{
		const int32_t N{ 8 };
		const double x0{ m_imageSize.width * 0.5 }, y0{ m_imageSize.height * 0.5 };
		cv::Mat_<double> L{ cv::Mat::zeros(m_uv.cols, N, CV_64F) };

		for (auto i{0}; i < m_uv.cols; ++i)
		{
			auto x = m_xyz(0, i);
			auto y = m_xyz(1, i);
			auto z = m_xyz(2, i);
			auto u = m_uv(0, i) - x0;
			auto v = m_uv(1, i) - y0;
			L(i, 0) = x * v;
			L(i, 1) = y * v;
			L(i, 2) = z * v;
			L(i, 3) = v;
			L(i, 4) = -x * u;
			L(i, 5) = -y * u;
			L(i, 6) = -z * u;
			L(i, 7) = -u;
		}

		cv::Mat S, U, Vt;
		cv::SVD::compute(L, S, U, Vt);

		cv::Mat_<double> x{ Vt.row(N - 1) };

		if (x(4) * x(4) + x(5) * x(5) + x(6) * x(6) < std::numeric_limits<double>::epsilon())
			throw std::runtime_error("[Tsai] Invalid extrinsic property.");

		auto c{ 1 / std::sqrt(x(4) * x(4) + x(5) * x(5) + x(6) * x(6)) };
		m_hScale = c * std::sqrt(x(0) * x(0) + x(1) * x(1) + x(2) * x(2));

		if (m_hScale < std::numeric_limits<double>::epsilon())
			throw std::runtime_error("[Tsai] Invalid unknown horizontal scale factor.");

		x *= c;
		m_t(0) = x(3) / m_hScale;
		m_t(1) = x(7);
		cv::Mat r1{ (cv::Mat_<double>(1, 3) << x(0), x(1), x(2)) };
		cv::Mat r2{ (cv::Mat_<double>(1, 3) << x(4), x(5), x(6)) };
		auto k{ -0.5 * r1.dot(r2) };
		r1 /= m_hScale;
		cv::Mat r1n {r1 + k * r2};
		cv::Mat r2n {r2 + k * r1};
		r1n /= cv::norm(r1n);
		r2n /= cv::norm(r2n);
		cv::Mat r3n{ r1.cross(r2) };
		r1n.copyTo(m_R.row(0));
		r2n.copyTo(m_R.row(1));
		r3n.copyTo(m_R.row(2));

		cv::SVD::compute(m_R, S, U, Vt);
		m_R = U * Vt;
	}
}