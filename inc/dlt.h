#pragma once

#include "CamCalib.h"

namespace ky
{
	/**
	* @brief Direct Linear Transformation. Implementation is based on "Introductory Techniques for 3-D Computer Vision - by E. Trucco and A. Verri"
	* @note: Data normalization is implemented for a stable projection matrix.
	* @note: Usage of OpenCV QR/RQ factorization is not advised.
	*/
	class DLT : public CamCalib
	{
	public:
		inline static const size_t minPts{ 6 };  /*!< The minimum number of points to calibrate using DLT. */

		/**
		* @brief Constructor
		* @param[in] _uv - 6 or more detected feature points in image, cv::Point2f format.
		* @param[in] _xyz - Control points on the object associated to _uv, cv::Point3f format.
		*/
		DLT(const std::vector<cv::Point2f>& _uv, const std::vector<cv::Point3f>& _xyz);

		virtual ~DLT() {}

		/**
		* @brief Implementation of calibration function.
		*/
		bool Calibrate();

	private:
		struct DataNormalization
		{
			cv::Mat_<double> T, W;

			DataNormalization(const cv::Mat_<double> _T = cv::Mat_<double>(), const cv::Mat_<double> _W = cv::Mat_<double>()) : T(_T), W(_W) {}
		};

		cv::Mat_<double> m_P;  /*!< Projection matrix */   

		void _Decompose();     /*!< Decompose projection matrx to calibration parameters. */		
		DLT::DataNormalization _Normalize() const;  /*!< Hartley's data normalization. https://cs.adelaide.edu.au/~wojtek/papers/pami-nals2.pdf */
		void _Projection();    /*!< Find projection matrx. */
		void _SwitchSign();    /*!< Switch sign of first two rows of m_R and m_t. */
	};
}
