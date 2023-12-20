#ifndef DETECTED_CLUSTER_FILTER__CLUSTER_LANELET_FILTER_HPP_
#define DETECTED_CLUSTER_FILTER__CLUSTER_LANELET_FILTER_HPP_

#include "utils/utils.hpp"

#include <lanelet2_extension/utility/message_conversion.hpp>
#include <lanelet2_extension/utility/query.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tier4_autoware_utils/geometry/geometry.hpp>

#include <autoware_auto_mapping_msgs/msg/had_map_bin.hpp>
#include <tier4_perception_msgs/msg/detected_object_with_feature.hpp>
#include <tier4_perception_msgs/msg/detected_objects_with_feature.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <string>

namespace cluster_lanelet_filter
{
using tier4_autoware_utils::LinearRing2d;
using tier4_autoware_utils::MultiPoint2d;
using tier4_autoware_utils::Point2d;
using tier4_autoware_utils::Polygon2d;

class ClusterLaneletFilterNode : public rclcpp::Node
{
public:
  explicit ClusterLaneletFilterNode(const rclcpp::NodeOptions & node_options);

private:
  void clusterCallback(const tier4_perception_msgs::msg::DetectedObjectsWithFeature::ConstSharedPtr);
  void mapCallback(const autoware_auto_mapping_msgs::msg::HADMapBin::ConstSharedPtr);

  rclcpp::Publisher<tier4_perception_msgs::msg::DetectedObjectsWithFeature>::SharedPtr cluster_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_pub_;
  rclcpp::Subscription<autoware_auto_mapping_msgs::msg::HADMapBin>::SharedPtr map_sub_;
  rclcpp::Subscription<tier4_perception_msgs::msg::DetectedObjectsWithFeature>::SharedPtr cluster_sub_;

  lanelet::LaneletMapPtr lanelet_map_ptr_;
  lanelet::ConstLanelets road_lanelets_;
  std::string lanelet_frame_id_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  LinearRing2d getConvexHull(const tier4_perception_msgs::msg::DetectedObjectsWithFeature &);
  lanelet::ConstLanelets getIntersectedLanelets(
    const LinearRing2d &, const lanelet::ConstLanelets &);
  bool isPolygonOverlapLanelets(const Polygon2d &, const lanelet::ConstLanelets &);
  geometry_msgs::msg::Polygon setFootprint(
    const tier4_perception_msgs::msg::DetectedObjectWithFeature &);
};

}  // namespace cluster_lanelet_filter

#endif  // DETECTED_CLUSTER_FILTER__CLUSTER_LANELET_FILTER_HPP_
