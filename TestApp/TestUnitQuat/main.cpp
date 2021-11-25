#include <random>
#include <pcl/common/distances.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <boost/filesystem.hpp>
#include "transformation_estimation_uq.h"
#include "pose_est_Lu.h"

using namespace pcl::registration;

typedef pcl::PointXYZ XYZ;
typedef pcl::PointCloud<XYZ> PointCloudXYZ;

static std::random_device rd;

void addGaussianNoise(PointCloudXYZ::Ptr _pts, const double _stdev);

Eigen::Matrix4d getTransformation(const double _w, const double _x, const double _y, const double _z,
	const double _tx, const double _ty, const double _tz, const double _scale);

int main ()
{
	ky::PoseEstLu poseEst;

	// Setup a ground truth transformation
	double qw{ 0.9988145 }, qx{ 0.0256408 }, qy{ -0.0395854 }, qz{ 0.0120474 };  // keep transformation envelop small 
	double tx{ -4.9 }, ty{ 3.1 }, tz{ 2.4 }, scale{ 0.99 };                      // for ICP test.
	Eigen::Matrix4d transform = getTransformation(qw, qx, qy, qz, tx, ty, tz, scale);

	// Load source point cloud
	boost::filesystem::path path = boost::filesystem::current_path();
	path += boost::filesystem::path("/../../../data/UQ/xyz.pcd");
	if (!boost::filesystem::exists(path))
		return 1;

	PointCloudXYZ::Ptr cloud_src(new PointCloudXYZ);
	pcl::io::loadPCDFile(path.string(), *cloud_src);

	// Set target point cloud by transforming the source using the ground truth transformation.
	PointCloudXYZ::Ptr cloud_tgt(new PointCloudXYZ);
	pcl::transformPointCloud(*cloud_src, *cloud_tgt, transform);

	// Apply additive noise to the target point cloud.
	double noise_stdev{ 1E-1 };
	addGaussianNoise(cloud_tgt, noise_stdev);

	Eigen::Matrix4d transform_uq = Eigen::Matrix4d::Identity();

	// Solve absolute orientation using unit quaternion, Horn's method (simplified version).
	TransformationEstimation<XYZ, XYZ, double>::Ptr teuq(new TransformationEstimationUQ<XYZ, XYZ, double>);
	teuq->estimateRigidTransformation(*cloud_src, *cloud_tgt, transform_uq);

	// Evaluation.
	float distance{ 0 };
	PointCloudXYZ::Ptr cloud_transformed(new PointCloudXYZ), cloud_aligned(new PointCloudXYZ);
	pcl::transformPointCloud(*cloud_src, *cloud_transformed, transform_uq);

	for (auto i{ 0 }; i < cloud_tgt->size(); ++i)
		distance += pcl::squaredEuclideanDistance(cloud_transformed->points[i], cloud_tgt->points[i]);

	distance /= cloud_src->size();

	// TransformationEstimationUQ being part of ICP pipeline.
	pcl::IterativeClosestPoint<XYZ, XYZ, double> icp;
	icp.setTransformationEstimation(teuq);
	icp.setInputSource(cloud_src);
	icp.setInputTarget(cloud_tgt);
	icp.align(*cloud_aligned);
	Eigen::Matrix4d transform_icp = icp.getFinalTransformation();
	auto fitness = icp.getFitnessScore();

	return 0;
}

void addGaussianNoise(PointCloudXYZ::Ptr _pts, const double _stdev)
{
	std::mt19937 gen(rd());
	std::normal_distribution<> distrib(0, _stdev);
	for (auto iter{ _pts->points.begin() }; iter != _pts->points.end(); ++iter)
	{
		iter->x += distrib(gen);
		iter->y += distrib(gen);
		iter->z += distrib(gen);
	}
}

Eigen::Matrix4d getTransformation(const double _w, const double _x, const double _y, const double _z,
	const double _tx, const double _ty, const double _tz, const double _scale)
{
	Eigen::Quaternion<double> q(_w, _x, _y, _z);
	Eigen::Matrix3d R = q.toRotationMatrix();
	Eigen::Vector3d t;
	t << _tx, _ty, _tz;
	Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
	transform.block<3, 3>(0, 0) = _scale * R;
	transform.block<3, 1>(0, 3) = t;

	return transform;
}
