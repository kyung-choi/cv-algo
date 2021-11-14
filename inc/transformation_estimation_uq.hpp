#ifndef TRANSFORMATION_ESTIMATION_UQ_HPP_
#define TRANSFORMATION_ESTIMATION_UQ_HPP_

#include <pcl/common/eigen.h>
#include <Eigen/Eigenvalues>

namespace pcl {
  namespace registration {
    template <typename PointSource, typename PointTarget, typename Scalar>
    inline void TransformationEstimationUQ<PointSource, PointTarget, Scalar>::estimateRigidTransformation(
      const pcl::PointCloud<PointSource>& cloud_src,
      const pcl::PointCloud<PointTarget>& cloud_tgt,
      Matrix4& transformation_matrix) const
    {
      const auto nr_points = cloud_src.size();
      if (cloud_tgt.size() != nr_points) {
        PCL_ERROR("[pcl::TransformationEstimationUQ::estimateRigidTransformation] Number "
          "or points in source (%zu) differs than target (%zu)!\n",
          static_cast<std::size_t>(nr_points),
          static_cast<std::size_t>(cloud_tgt.size()));
        return;
      }

      ConstCloudIterator<PointSource> source_it(cloud_src);
      ConstCloudIterator<PointTarget> target_it(cloud_tgt);
      estimateRigidTransformation(source_it, target_it, transformation_matrix);
    }

    template <typename PointSource, typename PointTarget, typename Scalar>
    void TransformationEstimationUQ<PointSource, PointTarget, Scalar>::
      estimateRigidTransformation(const pcl::PointCloud<PointSource>& cloud_src,
        const std::vector<int>& indices_src,
        const pcl::PointCloud<PointTarget>& cloud_tgt,
        Matrix4& transformation_matrix) const
    {
      if (indices_src.size() != cloud_tgt.size()) {
        PCL_ERROR("[pcl::TransformationUQ::estimateRigidTransformation] Number or points "
          "in source (%zu) differs than target (%zu)!\n",
          indices_src.size(),
          static_cast<std::size_t>(cloud_tgt.size()));
        return;
      }

      ConstCloudIterator<PointSource> source_it(cloud_src, indices_src);
      ConstCloudIterator<PointTarget> target_it(cloud_tgt);
      estimateRigidTransformation(source_it, target_it, transformation_matrix);
    }

    template <typename PointSource, typename PointTarget, typename Scalar>
    inline void
      TransformationEstimationUQ<PointSource, PointTarget, Scalar>::
      estimateRigidTransformation(const pcl::PointCloud<PointSource>& cloud_src,
        const std::vector<int>& indices_src,
        const pcl::PointCloud<PointTarget>& cloud_tgt,
        const std::vector<int>& indices_tgt,
        Matrix4& transformation_matrix) const
    {
      if (indices_src.size() != indices_tgt.size()) {
        PCL_ERROR("[pcl::TransformationEstimationUQ::estimateRigidTransformation] Number "
          "or points in source (%zu) differs than target (%zu)!\n",
          indices_src.size(),
          indices_tgt.size());
        return;
      }

      ConstCloudIterator<PointSource> source_it(cloud_src, indices_src);
      ConstCloudIterator<PointTarget> target_it(cloud_tgt, indices_tgt);
      estimateRigidTransformation(source_it, target_it, transformation_matrix);
    }

    template <typename PointSource, typename PointTarget, typename Scalar>
    void TransformationEstimationUQ<PointSource, PointTarget, Scalar>::
      estimateRigidTransformation(const pcl::PointCloud<PointSource>& cloud_src,
        const pcl::PointCloud<PointTarget>& cloud_tgt,
        const pcl::Correspondences& correspondences,
        Matrix4& transformation_matrix) const
    {
      ConstCloudIterator<PointSource> source_it(cloud_src, correspondences, true);
      ConstCloudIterator<PointTarget> target_it(cloud_tgt, correspondences, false);
      estimateRigidTransformation(source_it, target_it, transformation_matrix);
    }

    template <typename PointSource, typename PointTarget, typename Scalar>
    inline void TransformationEstimationUQ<PointSource, PointTarget, Scalar>::
      estimateRigidTransformation(ConstCloudIterator<PointSource>& source_it,
        ConstCloudIterator<PointTarget>& target_it,
        Matrix4& transformation_matrix) const
    {
      source_it.reset();
      target_it.reset();

      transformation_matrix.setIdentity();

      Eigen::Matrix<Scalar, 4, 1> centroid_src, centroid_tgt;
      // Estimate the centroids of source, target
      compute3DCentroid(source_it, centroid_src);
      compute3DCentroid(target_it, centroid_tgt);
      source_it.reset();
      target_it.reset();

      // Subtract the centroids from source, target
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> cloud_src_demean, cloud_tgt_demean;
      demeanPointCloud(source_it, centroid_src, cloud_src_demean);
      demeanPointCloud(target_it, centroid_tgt, cloud_tgt_demean);

      getTransformationFromUQ(cloud_src_demean, centroid_src, cloud_tgt_demean, centroid_tgt, transformation_matrix);
    }

    template <typename PointSource, typename PointTarget, typename Scalar>
    inline void TransformationEstimationUQ<PointSource, PointTarget, Scalar>::
      getTransformationFromUQ(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& cloud_src_demean,
        const Eigen::Matrix<Scalar, 4, 1>& centroid_src,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& cloud_tgt_demean,
        const Eigen::Matrix<Scalar, 4, 1>& centroid_tgt,
        Matrix4& transformation_matrix) const
    {
      Scalar scale = getScale(cloud_src_demean, cloud_tgt_demean);

      Eigen::Matrix<Scalar, 3, 3> M = Eigen::Matrix<Scalar, 3, 3>::Zero();
      for (int i = 0; i < cloud_src_demean.cols(); ++i)
        M += cloud_src_demean.block<3, 1>(0, i) * cloud_tgt_demean.block<3, 1>(0, i).transpose();

      Eigen::Matrix<Scalar, 4, 4> N;
      N(0, 0) = M(0, 0) + M(1, 1) + M(2, 2);
      N(1, 1) = M(0, 0) - M(1, 1) - M(2, 2);
      N(2, 2) = M(1, 1) - M(0, 0) - M(2, 2);
      N(3, 3) = M(2, 2) - M(0, 0) - M(1, 1);
      N(0, 1) = N(1, 0) = M(1, 2) - M(2, 1);
      N(0, 2) = N(2, 0) = M(2, 0) - M(0, 2);
      N(0, 3) = N(3, 0) = M(0, 1) - M(1, 0);
      N(1, 2) = N(2, 1) = M(0, 1) + M(1, 0);
      N(1, 3) = N(3, 1) = M(2, 0) + M(0, 2);
      N(2, 3) = N(3, 2) = M(1, 2) + M(2, 1);

      Eigen::EigenSolver<Eigen::Matrix<Scalar, 4, 4> > eig(N);
      Eigen::Matrix<Scalar, 4, 1> eigVals = eig.eigenvalues().real();

      int maxIdx(0);
      for (int i = 1; i < 4; ++i)
        if (eigVals(maxIdx) < eigVals(i))
          maxIdx = i;

      Eigen::Matrix<Scalar, 4, 1> q = eig.eigenvectors().real().block<4, 1>(0, maxIdx);
      Eigen::Quaternion<Scalar> quat(q(0), q(1), q(2), q(3));
      Eigen::Matrix<Scalar, 3, 3> R = quat.toRotationMatrix();
      Eigen::Matrix<Scalar, 3, 1> t = centroid_tgt.block<3, 1>(0, 0) - scale * R * centroid_src.block<3, 1>(0, 0);

      transformation_matrix = Eigen::Matrix<Scalar, 4, 4>::Identity();
      transformation_matrix.block<3, 3>(0, 0) = scale * R;
      transformation_matrix.block<3, 1>(0, 3) = t;
    }

    template <typename PointSource, typename PointTarget, typename Scalar> 
    inline Scalar TransformationEstimationUQ<PointSource, PointTarget, Scalar>::
      getScale(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& cloud_src_demean,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& cloud_tgt_demean) const
    {
      Scalar num(0), den(0);
      for (int i = 0; i < cloud_src_demean.cols(); ++i)
      {
        num += std::pow(cloud_tgt_demean(0, i), 2) + std::pow(cloud_tgt_demean(1, i), 2) + std::pow(cloud_tgt_demean(2, i), 2);
        den += std::pow(cloud_src_demean(0, i), 2) + std::pow(cloud_src_demean(1, i), 2) + std::pow(cloud_src_demean(2, i), 2);
      }

      return std::sqrt(num / den);
    }
  } // namespace registration
} // namespace pcl

#endif