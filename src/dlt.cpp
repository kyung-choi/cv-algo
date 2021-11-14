#include "dlt.h"

namespace ky
{
	DLT::DLT(const std::vector<cv::Point2f>& _uv, const std::vector<cv::Point3f>& _xyz) :
		CamCalib(_uv, _xyz) {}

	bool DLT::Calibrate()
	{
		assert(m_uv.cols >= static_cast<int>(DLT::minPts));
		
		_Projection();
		_Decompose();

		return true;
	}

	void DLT::_Decompose()
	{
		m_P /= cv::norm(m_P(cv::Rect(0, 2, 3, 1)));

		if (m_P(2, 3) < 0)
			m_P = -m_P;

		m_t(2) = m_P(2, 3);
		cv::Mat_<double> q1{m_P(cv::Rect(0, 0, 3, 1))};
		cv::Mat_<double> q2{m_P(cv::Rect(0, 1, 3, 1))};
		cv::Mat_<double> q3{m_P(cv::Rect(0, 2, 3, 1))};

		m_K(0, 2) = q1.dot(q3);
		m_K(1, 2) = q2.dot(q3);
		m_K(0, 0) = std::sqrt(q1.dot(q1) - std::pow(m_K(0, 2), 2));
		m_K(1, 1) = std::sqrt(q2.dot(q2) - std::pow(m_K(1, 2), 2));
		m_t(0) = (m_K(0, 2) * m_t(2) - m_P(0, 3)) / m_K(0, 0);
		m_t(1) = (m_K(1, 2) * m_t(2) - m_P(1, 3)) / m_K(1, 1);

		for (auto i{ 0 }; i < 3; ++i)
		{
			m_R(0, i) = (m_K(0, 2) * q3(i) - q1(i)) / m_K(0, 0);
			m_R(1, i) = (m_K(1, 2) * q3(i) - q2(i)) / m_K(1, 1);
			m_R(2, i) = q3(i);
		}

		cv::Mat S, U, Vt;
		cv::SVD::compute(m_R, S, U, Vt);

		m_R = U * Vt;
		double rms{ RMS() };

		_SwitchSign();
		double rmsSignSwitch{ RMS() };

		if (rms < rmsSignSwitch)
			_SwitchSign();
	}

	DLT::DataNormalization DLT::_Normalize() const
	{
		double uvLen{ 0 }, xyzLen{ 0 };
		cv::Mat_<double> uvMean, xyzMean;
		cv::reduce(m_uv, uvMean, 1, cv::REDUCE_AVG);
		cv::reduce(m_xyz, xyzMean, 1, cv::REDUCE_AVG);

		for (auto i{ 0 }; i < m_uv.cols; ++i)
		{
			uvLen += cv::norm(m_uv.col(i) - uvMean);
			xyzLen += cv::norm(m_xyz.col(i) - xyzMean);
		}

		uvLen /= m_uv.cols;
		xyzLen /= m_uv.cols;
		
		cv::Mat_<double> T{ cv::Mat_<double>::eye(3, 3) };  // T: for image points.
		cv::Mat_<double> W{ cv::Mat_<double>::eye(4, 4) };  // W: for object points.
		
		T(0, 0) = T(1, 1) = std::sqrt(2) / uvLen;
		T(0, 2) = -std::sqrt(2) * uvMean(0) / uvLen;
		T(1, 2) = -std::sqrt(2) * uvMean(1) / uvLen;

		W(0, 0) = W(1, 1) = W(2, 2) = std::sqrt(3) / xyzLen;
		W(0, 3) = -std::sqrt(3) * xyzMean(0) / xyzLen;
		W(1, 3) = -std::sqrt(3) * xyzMean(1) / xyzLen;
		W(2, 3) = -std::sqrt(3) * xyzMean(2) / xyzLen;

		return DLT::DataNormalization(T, W);
	}

	void DLT::_Projection()
	{
		DLT::DataNormalization hartley{ _Normalize() };

		cv::Mat_<double> uvn{ hartley.T * m_uv };
		cv::Mat_<double> xyzn{ hartley.W * m_xyz };
		cv::Mat_<double> L{ cv::Mat::zeros(2 * m_uv.cols, 12, CV_64F) };

		for (auto i{ 0 }; i < m_uv.cols; ++i)
		{
			auto x{ xyzn(0, i) };
			auto y{ xyzn(1, i) };
			auto z{ xyzn(2, i) };
			auto u{ uvn(0, i) };
			auto v{ uvn(1, i) };
			L(2 * i, 0) = L(2 * i + 1, 4) = x;
			L(2 * i, 1) = L(2 * i + 1, 5) = y;
			L(2 * i, 2) = L(2 * i + 1, 6) = z;
			L(2 * i, 3) = L(2 * i + 1, 7) = 1;
			L(2 * i, 8) = -x * u;
			L(2 * i, 9) = -y * u;
			L(2 * i, 10) = -z * u;
			L(2 * i, 11) = -u;
			L(2 * i + 1, 8) = -x * v;
			L(2 * i + 1, 9) = -y * v;
			L(2 * i + 1, 10) = -z * v;
			L(2 * i + 1, 11) = -v;
		}

		cv::Mat S, U, Vt;
		cv::SVD::compute(L, S, U, Vt);

		m_P = hartley.T.inv() * Vt.row(Vt.rows - 1).reshape(0, 3) * hartley.W;
	}

	void DLT::_SwitchSign()
	{
		m_R = -m_R;
		m_t = -m_t;
		m_P(cv::Rect(0, 2, 3, 1)).copyTo(m_R.row(2));
		m_t(2) = m_P(2, 3);
	}
}