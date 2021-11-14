#include "camcalib.h"

namespace ky
{
	CamCalib::CamCalib(const std::vector<cv::Point2f>& _uv, const std::vector<cv::Point3f>& _xyz, const cv::Size& _imageSize) : 
		m_imageSize(_imageSize)
	{
		m_uv = cv::Mat_<double>(3, static_cast<int32_t>(_uv.size()), 1.0);
		m_xyz = cv::Mat_<double>(4, static_cast<int32_t>(_uv.size()), 1.0);

		auto uvIter{ _uv.begin() };
		auto xyzIter{ _xyz.begin() };
		for (auto i{ 0 }; uvIter != _uv.end(); ++uvIter, ++xyzIter, ++i)
		{
			m_uv(0, i) = uvIter->x;
			m_uv(1, i) = uvIter->y;
			m_xyz(0, i) = xyzIter->x;
			m_xyz(1, i) = xyzIter->y;
			m_xyz(2, i) = xyzIter->z;
		}

		m_K = cv::Mat::eye(3, 3, CV_64F);
		m_R = cv::Mat::eye(3, 3, CV_64F);
		m_t = cv::Mat_<double>(3, 1, 0.0);
		m_distortCoeffs = cv::Mat_<double>(1, 5, 0.0);
	}

	cv::Mat_<double> CamCalib::Rt() const
	{
		assert(m_R.rows == 3 && m_R.cols == 3 && m_t.rows == 3 && m_t.cols == 1);

		cv::Mat_<double> Rt(3, 4);
		cv::hconcat(m_R, m_t, Rt);

		return Rt;
	}

	double CamCalib::RMS() const
	{
		assert(m_distortCoeffs.total() > 2 && m_distortCoeffs.total() < 6);
		assert(m_R.rows == 3 && m_R.cols == 3 && m_t.rows == 3 && m_t.cols == 1);

		const int32_t n{ m_uv.cols };
		cv::Mat_<double> rvec;
		cv::Rodrigues(m_R, rvec);
		std::vector<cv::Point3d> xyz;
		for (auto i{ 0 }; i < n; ++i)
			xyz.emplace_back(m_xyz(0, i), m_xyz(1, i), m_xyz(2, i));

		std::vector<cv::Point2d> uv;
		cv::projectPoints(xyz, rvec, m_t, m_K, m_distortCoeffs, uv);

		double rms{ 0 };
		for (auto i{ 0 }; i < n; ++i)
			rms += std::pow(m_uv(0, i) - uv[i].x, 2) + std::pow(m_uv(1, i) - uv[i].y, 2);

		return std::sqrt(rms / n);
	}

	double CamCalib::RMS3D() const
	{
		assert(m_distortCoeffs.total() > 2 && m_distortCoeffs.total() < 6);
		assert(m_R.rows == 3 && m_R.cols == 3 && m_t.rows == 3 && m_t.cols == 1);
		assert(!m_uv.empty() && !m_xyz.empty());

		const int32_t n{ m_uv.cols };
		cv::Mat_<double> rvec;
		cv::Rodrigues(m_R, rvec);

		std::vector<cv::Point2d> uv, uvn;
		for (auto i{ 0 }; i < n; ++i)
			uv.emplace_back(m_uv(0, i), m_uv(1, i));

		cv::undistortPoints(uv, uvn, m_K, m_distortCoeffs);   // uvn: uv is expressed in normalized image frame.
		cv::Mat_<double> xyzc{ Rt() * m_xyz };    // xyzc: object points in camera frame.

		double rms{ 0 };
		for (auto i{ 0 }; i < n; ++i)
		{
			cv::Mat_<double> objPt{ (cv::Mat_<double>(3, 1) << xyzc(0, i), xyzc(1, i), xyzc(2, i)) };
			cv::Mat_<double> lineDir{ (cv::Mat_<double>(3, 1) << uvn[i].x, uvn[i].y, 1.0) };
			cv::Mat_<double> linePt{ lineDir * (objPt.dot(lineDir) / lineDir.dot(lineDir)) };
			rms += std::pow(objPt(0) - linePt(0), 2) + std::pow(objPt(1) - linePt(1), 2) + std::pow(objPt(2) - linePt(2), 2);
		}

		return std::sqrt(rms / n);
	}
}