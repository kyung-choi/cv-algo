#pragma once

#include <opencv2/core.hpp>

namespace ky
{
	class QuadFit
	{
	public:
		/**
		* @brief Constructor
		* @param[in] _maxIter - Maximum search iteration before termination.
		* @param[in] _converge - Position change in pixel before considered converged.
		* @param[in] _maxCorr - Max position correction in pixel for a valid result.
		*/
		explicit QuadFit(const int _maxIter = 10, const double _converge = 1E-2, const double _maxCorr = 2.0);

		virtual ~QuadFit() {}

		/**
		* @brief Detect the center of circular features.
		* @param[in] _img - 8-bit single channel input image.
		* @param[in, out] _keyPts - A key point vector containing location and size of feature points.
		*/
		void Detect(const cv::Mat& _img, std::vector<cv::KeyPoint>& _keyPts);

	private:

		int m_maxIter;
		double m_converge;
		double m_maxCorr;

		inline static const int interpRate{ 4 };          // Surface interpolation rate.
		inline static const int basin{ 5 };               // The size of convergence basin in pixel.

		/**
		* @brief Blur the region of each feature according to its size.
		* @return Locally blurred image.
		*/
		cv::Mat _Blur(const cv::Mat& _img, const std::vector<cv::KeyPoint>& _keyPts) const;

		cv::Mat _NominalWindow() const;
	};
}
