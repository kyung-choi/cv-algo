#include <random>
#include <pcl/common/distances.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <boost/filesystem.hpp>
#include "transformation_estimation_uq.h"

using namespace pcl::registration;

typedef pcl::PointXYZ XYZ;
typedef pcl::PointCloud<XYZ> PointCloudXYZ;

void addNoise(PointCloudXYZ::Ptr _pts, const float _level);

Eigen::Matrix4f getTransformation(const float _w, const float _x, const float _y, const float _z,
	const float _tx, const float _ty, const float _tz, const float _scale = 1.f);

int main ()
{
	try
	{
		// Setup a ground truth transformation
		float qw{ 0.9988145f }, qx{ 0.0256408f }, qy{ -0.0395854f }, qz{ 0.0120474f };  // keep transformation envelop small 
		float tx{ -4.9f }, ty{ 3.1f }, tz{ 2.4f }, scale{ 0.99f };                      // for ICP test.
		Eigen::Matrix4f transform = getTransformation(qw, qx, qy, qz, tx, ty, tz, scale);

		// Load source point cloud
		boost::filesystem::path path = boost::filesystem::current_path();
		path += boost::filesystem::path("/../../../data/UQ/xyz.pcd");
		if (!boost::filesystem::exists(path))
			throw std::runtime_error("File not found, " + path.string() + ", in testAbsOrientUQ().");

		PointCloudXYZ::Ptr cloud_src(new PointCloudXYZ);
		pcl::io::loadPCDFile(path.string(), *cloud_src);

		// Set target point cloud by transforming the source using the ground truth transformation.
		PointCloudXYZ::Ptr cloud_tgt(new PointCloudXYZ);
		pcl::transformPointCloud(*cloud_src, *cloud_tgt, transform);

		// Apply additive noise to the target point cloud.
		addNoise(cloud_tgt, 0.3f);

		Eigen::Matrix4f transform_uq;

		// Solve absolute orientation using unit quaternion, Horn's method (simplified version).
		TransformationEstimation<XYZ, XYZ, float>::Ptr teuq(new TransformationEstimationUQ<XYZ, XYZ, float>);
		teuq->estimateRigidTransformation(*cloud_src, *cloud_tgt, transform_uq);

		std::cout << "g.t. transformation: \n" << transform << std::endl;
		std::cout << "estimated transformation: \n" << transform_uq << std::endl;

		// Evaluation.
		float distance{ 0 };
		PointCloudXYZ::Ptr cloud_transformed(new PointCloudXYZ), cloud_aligned(new PointCloudXYZ);
		pcl::transformPointCloud(*cloud_src, *cloud_transformed, transform_uq);

		for (auto i{ 0 }; i < cloud_tgt->size(); ++i)
			distance += pcl::squaredEuclideanDistance(cloud_transformed->points[i], cloud_tgt->points[i]);

		distance /= cloud_src->size();

		// TransformationEstimationUQ being part of ICP pipeline.
		pcl::IterativeClosestPoint<XYZ, XYZ, float> icp;
		icp.setTransformationEstimation(teuq);
		icp.setInputSource(cloud_src);
		icp.setInputTarget(cloud_tgt);
		icp.align(*cloud_aligned);
		Eigen::Matrix4f transform_icp = icp.getFinalTransformation();
		auto fitness = icp.getFitnessScore();

		std::cout << "estimated transformation in icp pipeline: \n" << transform_icp << std::endl;
	}
	catch (const std::runtime_error& rt_err)
	{
		std::cout << rt_err.what() << std::endl;
	}
	catch (const std::exception& ex)
	{
		std::cout << ex.what() << std::endl;
	}

	return 0;
}

void addNoise(PointCloudXYZ::Ptr _pts, const float _level)
{
	Eigen::MatrixXf random = Eigen::MatrixXf::Random(3, _pts->size());
	random *= _level;
	int i{ 0 };
	for (auto iter{ _pts->points.begin() }; iter != _pts->points.end(); ++iter)
	{
		iter->x += random(0, i);
		iter->y += random(1, i);
		iter->z += random(2, i++);
	}
}

Eigen::Matrix4f getTransformation(const float _w, const float _x, const float _y, const float _z,
	const float _tx, const float _ty, const float _tz, const float _scale)
{
	Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
	transform.block<3, 3>(0, 0) = _scale * Eigen::Quaternionf(_w, _x, _y, _z).toRotationMatrix();
	transform.block<3, 1>(0, 3) = Eigen::Vector3f(_tx, _ty, _tz);

	return transform;
}