#pragma once

#include <pcl/registration/transformation_estimation.h>
#include <pcl/cloud_iterator.h>

namespace pcl {
  namespace registration {
    template <typename PointSource, typename PointTarget, typename Scalar = float>
    class TransformationEstimationUQ : public TransformationEstimation<PointSource, PointTarget, Scalar> {
    public:
      using Ptr = std::shared_ptr<TransformationEstimationUQ<PointSource, PointTarget, Scalar> >;
      using ConstPtr = std::shared_ptr<const TransformationEstimationUQ<PointSource, PointTarget, Scalar> >;
      using Matrix4 = typename TransformationEstimation<PointSource, PointTarget, Scalar>::Matrix4;

      TransformationEstimationUQ() {};

      virtual ~TransformationEstimationUQ() {};

      /** \brief Estimate a rigid rotation transformation between a source and a target
       * point cloud using dual quaternion optimization \param[in] cloud_src the source
       * point cloud dataset \param[in] cloud_tgt the target point cloud dataset \param[out]
       * transformation_matrix the resultant transformation matrix
       */
      inline void estimateRigidTransformation(const pcl::PointCloud<PointSource>& cloud_src,
          const pcl::PointCloud<PointTarget>& cloud_tgt,
          Matrix4& transformation_matrix) const;

      /** \brief Estimate a rigid rotation transformation between a source and a target
       * point cloud using dual quaternion optimization \param[in] cloud_src the source
       * point cloud dataset \param[in] indices_src the vector of indices describing the
       * points of interest in \a cloud_src
       * \param[in] cloud_tgt the target point cloud dataset
       * \param[out] transformation_matrix the resultant transformation matrix
       */
      inline void estimateRigidTransformation(const pcl::PointCloud<PointSource>& cloud_src,
          const std::vector<int>& indices_src,
          const pcl::PointCloud<PointTarget>& cloud_tgt,
          Matrix4& transformation_matrix) const;

      /** \brief Estimate a rigid rotation transformation between a source and a target
       * point cloud using dual quaternion optimization \param[in] cloud_src the source
       * point cloud dataset \param[in] indices_src the vector of indices describing the
       * points of interest in \a cloud_src
       * \param[in] cloud_tgt the target point cloud dataset
       * \param[in] indices_tgt the vector of indices describing the correspondences of the
       * interest points from \a indices_src
       * \param[out] transformation_matrix the resultant transformation matrix
       */
      inline void estimateRigidTransformation(const pcl::PointCloud<PointSource>& cloud_src,
          const std::vector<int>& indices_src,
          const pcl::PointCloud<PointTarget>& cloud_tgt,
          const std::vector<int>& indices_tgt,
          Matrix4& transformation_matrix) const;

      /** \brief Estimate a rigid rotation transformation between a source and a target
       * point cloud using dual quaternion optimization \param[in] cloud_src the source
       * point cloud dataset \param[in] cloud_tgt the target point cloud dataset \param[in]
       * correspondences the vector of correspondences between source and target point cloud
       * \param[out] transformation_matrix the resultant transformation matrix
       */
      void estimateRigidTransformation(const pcl::PointCloud<PointSource>& cloud_src,
          const pcl::PointCloud<PointTarget>& cloud_tgt,
          const pcl::Correspondences& correspondences,
          Matrix4& transformation_matrix) const;

    protected:
      /** \brief Estimate a rigid rotation transformation between a source and a target
       * \param[in] source_it an iterator over the source point cloud dataset
       * \param[in] target_it an iterator over the target point cloud dataset
       * \param[out] transformation_matrix the resultant transformation matrix
       */
      void estimateRigidTransformation(ConstCloudIterator<PointSource>& source_it,
          ConstCloudIterator<PointTarget>& target_it,
          Matrix4& transformation_matrix) const;

      void getTransformationFromUQ(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& cloud_src_demean,
        const Eigen::Matrix<Scalar, 4, 1>& centroid_src,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& cloud_tgt_demean,
        const Eigen::Matrix<Scalar, 4, 1>& centroid_tgt,
        Matrix4& transformation_matrix) const;

      Scalar getScale(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& cloud_src_demean,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& cloud_tgt_demean) const;
    };

  } // namespace registration
} // namespace pcl

#include "transformation_estimation_uq.hpp"