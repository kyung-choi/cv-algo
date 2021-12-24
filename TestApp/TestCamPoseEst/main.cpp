#include "pose_est_Lu.h"
#include "pose_est_Ansar.h"
#include <pcl/registration/transforms.h>

void addNoise(PointCloudUV::Ptr _pts, const float _level);

void generatePoseEstDataSet(const float _fx, const float _fy, const float _ppx, const float _ppy,
	PointCloudXYZ& _xyz, PointCloudUV& _uv, Eigen::Matrix4f& _Rt);

Eigen::Matrix4f getTransformation(const Eigen::Vector3f& _rpy ,	const Eigen::Vector3f& _t);

int main ()
{
	try
	{
		// Intrinsic camera parameters
		float fx{ 4500 }, fy{ 4500 }, ppx{ 512 }, ppy{ 512 };

		PointCloudUV::Ptr cloud_uv(new PointCloudUV);  // ground-truth image coordinates.
		PointCloudXYZ::Ptr cloud_xyz(new PointCloudXYZ);  // object points.
		Eigen::Matrix4f Rt;  // ground-truth camera pose.
		generatePoseEstDataSet(fx, fy, ppx, ppy, *cloud_xyz, *cloud_uv, Rt);

		ky::PoseEstAnsar ansar(fx, fy, ppx, ppy);
		Eigen::Matrix4f estRt;  // ground-truth camera pose.
		ansar.estimatePose(*cloud_xyz, *cloud_uv, estRt);

		std::cout << "ground-truth:\n" << Rt << std::endl;

		// Initial guess transformation and feature detection error range.
		float rotErr{ 10.f }, transErr{ 30.f }, detectErr{ 0.3f };

		// Initial guess of camera pose is off from ground-truth; by applying an error transformation below.
		Eigen::Vector3f rpy = Eigen::Vector3f::Random() * rotErr * EIGEN_PI / 180;  // rot error in initial transformation.
		Eigen::Vector3f t = Eigen::Vector3f::Random() * transErr;  // translation error in initial transformation.
		// Initial geuss of camera pose is, in general, from a linear method and inaccurate.
		Eigen::Matrix4f Rt_est = getTransformation(rpy, t) * Rt;

		std::cout << "initial guess:\n" << Rt_est << std::endl;

		// Adding noise to image coordinates.
		addNoise(cloud_uv, detectErr);

		// Estimate camera pose with noisy measurements and an initial guess.
		ky::PoseEstLu poseEstLu(fx, fy, ppx, ppy);
		poseEstLu.estimatePose(*cloud_xyz, *cloud_uv, Rt_est);

		std::cout << "estimated:\n" << Rt_est << std::endl;
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

void generatePoseEstDataSet(const float _fx, const float _fy, const float _ppx, const float _ppy,
	PointCloudXYZ& _xyz, PointCloudUV& _uv, Eigen::Matrix4f& _Rt)
{
	// a nominal camera pose transformation.
	_Rt = getTransformation(Eigen::Vector3f(EIGEN_PI, 0, 0), Eigen::Vector3f(1, -10, 1050));

	// some non-collinear model point.
	_xyz.clear();
	_xyz.points.emplace_back(-50,   0,  50);
	_xyz.points.emplace_back(-70,  10,   0);
	_xyz.points.emplace_back( -5, -50, -10);
	_xyz.points.emplace_back( 60,  20,  20);

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

Eigen::Matrix4f getTransformation(const Eigen::Vector3f& _rpy, const Eigen::Vector3f& _t)
{
	Eigen::AngleAxisf roll(_rpy(0), Eigen::Vector3f::UnitX());
	Eigen::AngleAxisf pitch(_rpy(1), Eigen::Vector3f::UnitY());
	Eigen::AngleAxisf yaw(_rpy(2), Eigen::Vector3f::UnitZ());

	Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
	transform.block<3, 3>(0, 0) = Eigen::Quaternionf(yaw * pitch * roll).toRotationMatrix();
	transform.block<3, 1>(0, 3) = _t;

	return transform;
}