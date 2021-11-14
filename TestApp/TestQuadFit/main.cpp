#include "quad_fit.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <filesystem>
#include <random>

static std::random_device rd;

void addGaussianNoise(std::vector<cv::KeyPoint>& _keyPts, const float _stdev);

cv::Mat_<double> computeError(const std::vector<cv::KeyPoint>& _gtKeyPts,
	const std::vector<cv::KeyPoint>& _testKeyPts, double* _mean, double* _stdev);

int main()
{
	// Load ground truth feature point coordinates.
	std::filesystem::path gt_path{ std::filesystem::current_path() };
	gt_path += std::filesystem::path("/../../data/QuadFit/gt.json");

	if (!std::filesystem::exists(gt_path))
		return 1;

	cv::FileStorage fs(gt_path.string(), cv::FileStorage::READ);
	cv::Mat_<float> gt{ fs["UV"].mat() };

	fs.release();

	// Load ground truth test image with additive noise.
	std::filesystem::path img_path{ std::filesystem::current_path() };
	img_path += std::filesystem::path("/../../data/QuadFit/test.png");

	cv::Mat img{ cv::imread(img_path.string(), cv::IMREAD_GRAYSCALE) };

	// Store ground truth coordinate and size info to a KeyPoint object.
	const float feature_size{ 11.f };  // should be given by other method such as CCL.
	std::vector<cv::KeyPoint> gtKeyPts;
	for (auto i{ 0 }; i < gt.cols; ++i)
		gtKeyPts.push_back(cv::KeyPoint(gt(0, i), gt(1, i), feature_size));

	// Test the method with initial points away from ground truth.
	std::vector<cv::KeyPoint> testKeyPts(gtKeyPts);
	addGaussianNoise(testKeyPts, 0.5f);

	// Feature detector finding the peak of quadratic surface of a circular feature.
	ky::QuadFit peakDetector;
	peakDetector.Detect(img, testKeyPts);

	// Compute detection error.
	double mean, stdev;
	cv::Mat_<double> error{ computeError(gtKeyPts, testKeyPts, &mean, &stdev) };

	// Draw results on the input image.
	cv::Mat imgKeyPts;
	cv::drawKeypoints(img, testKeyPts, imgKeyPts);

	std::ostringstream sstream;
	sstream << std::fixed << std::setprecision(2);
	
	for (auto i{ 0 }; i < testKeyPts.size(); ++i)
	{
  	sstream.str("");
  	sstream << error(i);
  	cv::putText(imgKeyPts, sstream.str(), testKeyPts[i].pt + cv::Point2f(5, -5),
  		cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar{ 255, 127, 63 }, 1);
  }

	std::string title = "error: " + std::to_string(mean) + ", " + std::to_string(stdev);
	cv::imshow(title, imgKeyPts);
	cv::waitKey(0);
	cv::destroyWindow(title);

	return 0;
}

void addGaussianNoise(std::vector<cv::KeyPoint>& _keyPts, const float _stdev)
{
	std::mt19937 gen(rd());
	std::normal_distribution<float> distrib(0, _stdev);
	for (auto iter{ _keyPts.begin() }; iter != _keyPts.end(); ++iter)
	{
		iter->pt.x += distrib(gen);
		iter->pt.y += distrib(gen);
	}
}

cv::Mat_<double> computeError(const std::vector<cv::KeyPoint>& _gtKeyPts,
	const std::vector<cv::KeyPoint>& _testKeyPts, double* _mean, double* _stdev)
{
	cv::Mat_<double> error(0, 1);
	for (auto gtIter{ _gtKeyPts.begin() }, testIter{ _testKeyPts.begin() }; 
		gtIter != _gtKeyPts.end(); gtIter++, testIter++)
		error.push_back(cv::norm(gtIter->pt - testIter->pt));

	cv::Scalar mean, stdev;
	cv::meanStdDev(error, mean, stdev);

	*_mean = mean(0);
	*_stdev = stdev(0);

	return error;
}
