#include "quad_fit.h"
#include <opencv2/imgproc.hpp>
#include <execution>

namespace ky
{
	QuadFit::QuadFit(const int _maxIter, const double _converge, const double _maxCorr) :
		m_maxIter(_maxIter), m_converge(_converge), m_maxCorr(_maxCorr) {}

	void QuadFit::Detect(const cv::Mat& _img, std::vector<cv::KeyPoint>& _keyPts) 
	{
		// Solving pseudo inverse once instead of least square every iteration
		// by keeping the coordinate of ROI centered to nominal window
		cv::Mat_<double> A{ _NominalWindow() }, P;  // P = pinv(A)
		cv::invert(A, P, cv::DecompTypes::DECOMP_SVD);
		cv::Mat imgBlur{ _Blur(_img, _keyPts) };

		auto op = [&imgBlur, &P, this](cv::KeyPoint keyPt)
		{
			int hw{ interpRate * basin / 2 }, iter{ 0 };
			double change{ m_converge + 1.0 }, q[4], shift{ 0.5 }, step{ 1.0 / interpRate };
			double x{ keyPt.pt.x }, y{ keyPt.pt.y };

			while (iter++ < m_maxIter && change > m_converge)
			{
				if (y < basin || x < basin || y + basin > imgBlur.rows || x + basin > imgBlur.cols)  // a border condition.
					break;

				auto alpha{ 0.0 }, beta{ 0.0 }, gamma{ 0.0 }, delta{ 0.0 }, eta{ 0.0 };
				for (auto k{ 0 }, i{ -hw }; i <= hw; ++i)
				{
					auto v{ y + step * i }, vf{ std::remainder(v, 1.0) };
					auto top{ static_cast<int>(v + shift) - 1 };

					for (auto j{ -hw }; j <= hw; ++j)
					{
						auto u{ x + step * j }, uf{ std::remainder(u, 1.0) };
						auto left{ static_cast<int>(u + shift) - 1 };

						for (auto m{ 0 }; m < 4; ++m)  // Inline bi-cubic interpolation.
						{
							double* p = imgBlur.ptr<double>(top + m); p += left;
							q[m] = p[1] + 0.5 * uf * (p[2] - p[0] + uf * (2 * p[0] - 5 * p[1] + 4 * p[2] - p[3] + uf * (3 * (p[1] - p[2]) + p[3] - p[0])));
						}

						auto b{ q[1] + 0.5 * vf * (q[2] - q[0] + vf * (2 * q[0] - 5 * q[1] + 4 * q[2] - q[3] + vf * (3 * (q[1] - q[2]) + q[3] - q[0]))) };
						// surface parameters are being solved... 
						alpha += P(0, k) * b;
						beta += P(1, k) * b;
						gamma += P(2, k) * b;
						delta += P(3, k) * b;
						eta += P(4, k++) * b;
					}
				}
				
				auto det{ 4 * alpha * gamma - beta * beta };
				if (std::abs(det) < std::numeric_limits<double>::epsilon())
					break;

  			auto xNew{ x - (2 * gamma * delta - beta * eta) / det };  // applying difference from nominal 
  			auto yNew{ y - (2 * alpha * eta - beta * delta) / det };  // instead of setting absolute coordinate.
  			change = std::sqrt((x - xNew) * (x - xNew) + (y - yNew) * (y - yNew));
  			x = xNew;
  			y = yNew;
			}

			return (std::pow((keyPt.pt.x - x), 2) + std::pow((keyPt.pt.y - y), 2) < m_maxCorr * m_maxCorr) ?
				cv::KeyPoint(static_cast<float>(x), static_cast<float>(y), keyPt.size, keyPt.angle, keyPt.response,
					keyPt.octave, keyPt.class_id) : keyPt;
		};

		std::transform(std::execution::par, _keyPts.begin(), _keyPts.end(), _keyPts.begin(), op);
	}

	cv::Mat QuadFit::_Blur(const cv::Mat& _src, const std::vector<cv::KeyPoint>& _keyPts) const
	{
		cv::Mat dst(_src.size(), CV_64FC1, cv::Scalar(0));
		for (const auto& keyPt : _keyPts)
		{
			int N{ static_cast<int>(keyPt.size) < basin ? basin : static_cast<int>(keyPt.size) };
			cv::Rect roi(static_cast<int>(keyPt.pt.x) - N, static_cast<int>(keyPt.pt.y) - N, N * 2, N * 2);
			_src(roi).convertTo(dst(roi), CV_64FC1);
			cv::GaussianBlur(dst(roi), dst(roi), cv::Size(), N * 0.25);  // sigma: half the radius
		}

		return dst;
	}

	cv::Mat QuadFit::_NominalWindow() const
	{
		int hw{ interpRate * basin / 2 }, fw{ 2 * hw + 1 };
		double step{ 1.0 / interpRate };
		cv::Mat_<double> A(fw * fw, 6);
		for (auto k{ 0 }, i{ -hw }; i <= hw; ++i)
			for (auto j{ -hw }; j <= hw; ++j)
			{
				A(k, 0) = step * step * j * j;
				A(k, 1) = step * step * i * j;
				A(k, 2) = step * step * i * i;
				A(k, 3) = step * j;
				A(k, 4) = step * i;
				A(k++, 5) = 1.0;
			}

		return A;
	}
}
