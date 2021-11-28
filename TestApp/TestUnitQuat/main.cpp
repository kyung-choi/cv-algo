#include <random>
#include <pcl/common/distances.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <boost/filesystem.hpp>
#include "transformation_estimation_uq.h"
#include "pose_est_Lu.h"

using namespace pcl::registration;

typedef pcl::PointUV UV;
typedef pcl::PointCloud<UV> PointCloudUV;
typedef pcl::PointXYZ XYZ;
typedef pcl::PointCloud<XYZ> PointCloudXYZ;

void addNoise(PointCloudUV::Ptr _pts, const float _level);
void addNoise(PointCloudXYZ::Ptr _pts, const float _level);

void generatePoseEstDataSet(const float _fx, const float _fy, const float _ppx, const float _ppy,
	const PointCloudXYZ& _xyz, PointCloudUV& _uv, Eigen::Matrix4f& _Rt);

Eigen::Matrix4f getTransformation(const float _w, const float _x, const float _y, const float _z,
	const float _tx, const float _ty, const float _tz, const float _scale = 1.f);

void testAbsOrientUQ();
void testPoseEstLu();

int main ()
{
	try
	{
		testAbsOrientUQ();
		testPoseEstLu();
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

void addNoise(PointCloudUV::Ptr _pts, const float _level)
{
	Eigen::MatrixXf random = Eigen::MatrixXf::Random(2, _pts->size());
	random *= _level;
	int i{ 0 };
	for (auto iter{ _pts->points.begin() }; iter != _pts->points.end(); ++iter)
	{
		iter->u += random(0, i);
		iter->v += random(1, i++);
	}
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

void generatePoseEstDataSet(const float _fx, const float _fy, const float _ppx, const float _ppy,
	const PointCloudXYZ& _xyz, PointCloudUV& _uv, Eigen::Matrix4f& _Rt)
{
	// nominal camera pose transformation.
	_Rt = Eigen::Matrix4f::Identity();
	_Rt(1, 1) = _Rt(2, 2) = -1;
	_Rt.block<3, 1>(0, 3) = Eigen::Vector3f(1, -10, 1050);

	PointCloudXYZ::Ptr xyzc(new PointCloudXYZ);
	pcl::transformPointCloud(_xyz, *xyzc, _Rt);

	_uv.clear();
	UV pt;
	for (const auto& wc : xyzc->points)
	{
		pt.u = (_fx * wc.x + _ppx * wc.z) / wc.z;
		pt.v = (_fy * wc.y + _ppy * wc.z) / wc.z;
		_uv.push_back(pt);
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

void testAbsOrientUQ()
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
	float noise_level{ 0.3f };
	addNoise(cloud_tgt, noise_level);

	Eigen::Matrix4f transform_uq = Eigen::Matrix4f::Identity();

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

void testPoseEstLu()
{
	// Pose estimation test
	boost::filesystem::path path = boost::filesystem::current_path();
	path += boost::filesystem::path("/../../../data/UQ/xyz.pcd");
	if (!boost::filesystem::exists(path))
		throw std::runtime_error("File not found, " + path.string() + ", in testPoseEstLu().");

	PointCloudXYZ::Ptr cloud_xyz(new PointCloudXYZ);
	pcl::io::loadPCDFile(path.string(), *cloud_xyz);

	// Intrinsic camera parameters
	const float fx{ 4500 }, fy{ 4500 }, ppx{ 512 }, ppy{ 512 };

	PointCloudUV::Ptr cloud_uv(new PointCloudUV);  // ground-truth image coordinates.
	Eigen::Matrix4f Rt;  // ground-truth camera pose.
	generatePoseEstDataSet(fx, fy, ppx, ppy, *cloud_xyz, *cloud_uv, Rt);

	// Initial guess of camera pose is off from ground-truth; by applying an error transformation below.
	float qw{ 0.9933643f }, qx{ 0.0376655f }, qy{ -0.0688858f }, qz{ 0.0840444f };
	float tx{ 25.f }, ty{ -35.f }, tz{ 30.f };

	// Initial geuss of camera pose is, in general, from a linear method and inaccurate.
	Eigen::Matrix4f Rt_est = getTransformation(qw, qx, qy, qz, tx, ty, tz) * Rt;

	// Adding noise to image coordinate.
	addNoise(cloud_uv, 0.3f);

	std::cout << "ground-truth:\n" << Rt << std::endl;
	std::cout << "initial guess:\n" << Rt_est << std::endl;

	// Estimate camera pose with noisy measurements and an initial guess.
	ky::PoseEstLu poseEst;
	poseEst.setFocalLength(fx, fy);
	poseEst.setPP(ppx, ppy);
	poseEst.estimatePose(*cloud_xyz, *cloud_uv, Rt_est);

	std::cout << "estimated:\n" << Rt_est << std::endl;
}
