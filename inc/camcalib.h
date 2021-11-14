#pragma once

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

namespace ky
{
	/**
	* @brief Base class of camera calibration. Derived class must implement Calibrate().
	*/
	class CamCalib
	{
	public:
		/**
		* @brief Constructor
		* @param[in] _uv - Feature points in image space, cv::Point2f format.
		* @param[in] _xyz - Feature points in object space, cv::Point3f format.
		* @param[in] _imageSize - Image size in cv::Size format.
		*/
		CamCalib(const std::vector<cv::Point2f >& _uv, const std::vector<cv::Point3f >& _xyz, const cv::Size& _imageSize = cv::Size(1024, 1024));

		/**
		* @brief Destructor
		*/
		virtual ~CamCalib() {}

		/**
		* @brief Get 3x3 camera matrix.
		* @return - Camera matrix in cv::Mat format.
		*/
		virtual inline cv::Mat_<double> K() const { return m_K; }

		/**
		* @brief Get distortion coefficients.
		* @return - Distortion coefficient in cv::Mat format.
		*/
		virtual inline cv::Mat_<double> DistortCoeffs() const { return m_distortCoeffs; }

		/**
		* @brief Get camera pose matrix in cv::Mat format.
		* @return - 3x4 camera pose matrix consisting of 3x3 orthonormal rotation matrix and 3x1 translation vector, [R t].
		*/
		virtual inline cv::Mat_<double> Rt() const;

		/**
		* @brief Perform calibration.
		*/
		virtual bool Calibrate() = 0;

		/**
		* @brief Compute root mean square reprojection error. This function makes use of cv::projectPoints().
		* @return RMS
		*/
		virtual double RMS() const;

		/**
		* @brief Compute root mean square orthogonal distance between calibrated rays and associated object points. 
		* @return RMS
		*/
		virtual double RMS3D() const;

	protected:
		cv::Mat_<double> m_uv, m_xyz;         /*!< Input image and object points in cv::Mat format. */
		cv::Mat_<double> m_K, m_R, m_t, m_distortCoeffs;  /*!< Intrinsic(m_K)/Extrisic(m_R, m_t)/Distortion(m_distortCoeffs) in cv::Mat format. */
		cv::Size m_imageSize;                 /*!< Input image size. */
	};
}

