#pragma once

#include "camcalib.h"

namespace ky
{
	/**
	* @brief Tsai's camera calibration (linear solution only). Implementation is based on http://people.csail.mit.edu/bkph/articles/Tsai_Revisited.pdf
	*/
	class Tsai : public CamCalib
	{
	public:
		inline static const size_t minPts{ 8 };       /*!< The minimum number of points to calibrate using Tsai. */
		inline static const int32_t numParams{ 15 };  /*!< The number of parameters to calibrate using Tsai. */
		
	  /**
		* @brief Constructor
		* @param[in] _uv - 8 or more detected feature points in image, cv::Point2f format.
		* @param[in] _xyz - Control points on the object associated to _uv, cv::Point3f format.
		* @param[in] _image_size - Calibration image size in cv::Size format.
		*/
		explicit Tsai(const std::vector<cv::Point2f>& _uv, const std::vector<cv::Point3f>& _xyz, const cv::Size& _image_size);

		virtual ~Tsai() {}

		/**
		* @brief Implementation of calibration function.
		*/
		bool Calibrate();

		/**
		* @brief Returns unknown horizontal scale factor.
		*/
		inline double HorizontalScaleFactor() const { return m_hScale; }

	private:
		double m_hScale{ 1.0 };    /*!< Unknown horizontal scale factor. */	

		void _RTxTy();      /*!< Compute rotation matrix and two components in translation. */
		void _FTz();        /*!< Compute focal length and one component in translation. */
	};
}
