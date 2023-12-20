// Copyright 2022 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "detected_cluster_filter/cluster_lanelet_filter.hpp"

#include <object_recognition_utils/object_recognition_utils.hpp>
#include <tier4_autoware_utils/tier4_autoware_utils.hpp>

#include <boost/geometry/algorithms/convex_hull.hpp>
#include <boost/geometry/algorithms/disjoint.hpp>
#include <boost/geometry/algorithms/intersects.hpp>

#include <lanelet2_core/geometry/Polygon.h>

#include "utils/utils.hpp"

#include <iostream>

namespace cluster_lanelet_filter
{
ClusterLaneletFilterNode::ClusterLaneletFilterNode(const rclcpp::NodeOptions & node_options)
: Node("cluster_lanelet_filter_node", node_options),
  tf_buffer_(this->get_clock()),
  tf_listener_(tf_buffer_)
{
  using std::placeholders::_1;

  // Set publisher/subscriber
  map_sub_ = this->create_subscription<autoware_auto_mapping_msgs::msg::HADMapBin>(
    "input/vector_map", rclcpp::QoS{1}.transient_local(),
    std::bind(&ClusterLaneletFilterNode::mapCallback, this, _1));
    cluster_sub_ = this->create_subscription<tier4_perception_msgs::msg::DetectedObjectsWithFeature>(
      "input/cluster", rclcpp::QoS{1}, std::bind(&ClusterLaneletFilterNode::clusterCallback, this, _1));
    cluster_pub_ = this->create_publisher<tier4_perception_msgs::msg::DetectedObjectsWithFeature>(
      "output/cluster", rclcpp::QoS{1});
    debug_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("debug/output_clusters", 1);
}

void ClusterLaneletFilterNode::mapCallback(
  const autoware_auto_mapping_msgs::msg::HADMapBin::ConstSharedPtr map_msg)
{
  lanelet_frame_id_ = map_msg->header.frame_id;
  lanelet_map_ptr_ = std::make_shared<lanelet::LaneletMap>();
  lanelet::utils::conversion::fromBinMsg(*map_msg, lanelet_map_ptr_);
  const lanelet::ConstLanelets all_lanelets = lanelet::utils::query::laneletLayer(lanelet_map_ptr_);
  road_lanelets_ = lanelet::utils::query::roadLanelets(all_lanelets);
}

void ClusterLaneletFilterNode::clusterCallback(
  const tier4_perception_msgs::msg::DetectedObjectsWithFeature::ConstSharedPtr input_msg)
{

  // Guard
  if (cluster_pub_->get_subscription_count() < 1) return;

  tier4_perception_msgs::msg::DetectedObjectsWithFeature output_cluster_msg;
  output_cluster_msg.header = input_msg->header;

  if (!lanelet_map_ptr_) {
    RCLCPP_ERROR(get_logger(), "No vector map received.");
    return;
  }
  tier4_perception_msgs::msg::DetectedObjectsWithFeature transformed_clusters;
  if (!object_recognition_utils::transformClusters(
        *input_msg, lanelet_frame_id_, tf_buffer_, transformed_clusters)) {
    RCLCPP_ERROR(get_logger(), "Failed transform to %s.", lanelet_frame_id_.c_str());
    return;
  }

  // calculate convex hull
  const auto convex_hull = getConvexHull(transformed_clusters);
  // get intersected lanelets
  lanelet::ConstLanelets intersected_lanelets = getIntersectedLanelets(convex_hull, road_lanelets_);

  int index = 0;
  for (const auto & cluster : transformed_clusters.feature_objects) {
    // Assuming 'setFootprint' is adapted to work with the new message type
    const auto footprint = setFootprint(cluster);
    Polygon2d polygon;

    for (const auto & point : footprint.points) {
      // Transform the point based on the cluster's pose
      // Note: You will need to adapt this line to the new message structure
      const geometry_msgs::msg::Point32 point_transformed =
        tier4_autoware_utils::transformPoint(point, cluster.object.kinematics.pose_with_covariance.pose);
      polygon.outer().emplace_back(point_transformed.x, point_transformed.y);
    }
    polygon.outer().push_back(polygon.outer().front());

    std::cout << "If in the lanelets: " << isPolygonOverlapLanelets(polygon, intersected_lanelets) << std::endl;
  
    // Check if the polygon overlaps with any lanelets
    if (isPolygonOverlapLanelets(polygon, intersected_lanelets)) {
      // Add the cluster to the output message
      // Note: You will need to adapt this line to the new message structure
      output_cluster_msg.feature_objects.emplace_back(input_msg->feature_objects.at(index));
      
    }
    ++index;
  }
  // Publish the filtered clusters
  cluster_pub_->publish(output_cluster_msg);
  // build debug msg
  if (debug_pub_->get_subscription_count() < 1) {
    return;
  }
  {
    sensor_msgs::msg::PointCloud2 debug;
    utils::convertObjectMsg2SensorMsg(output_cluster_msg, debug);
    debug_pub_->publish(debug);
  }
}

geometry_msgs::msg::Polygon ClusterLaneletFilterNode::setFootprint(
  const tier4_perception_msgs::msg::DetectedObjectWithFeature & detected_cluster)
{
  geometry_msgs::msg::Polygon footprint;

  const auto object_size = detected_cluster.feature.cluster;
  const double x_front = 0.1 / 2.0;
  const double x_rear = -0.1 / 2.0;
  const double y_left = 0.1 / 2.0;
  const double y_right = -0.1 / 2.0;

  footprint.points.push_back(
    geometry_msgs::build<geometry_msgs::msg::Point32>().x(x_front).y(y_left).z(0.0));
  footprint.points.push_back(
      geometry_msgs::build<geometry_msgs::msg::Point32>().x(x_front).y(y_right).z(0.0));
  footprint.points.push_back(
    geometry_msgs::build<geometry_msgs::msg::Point32>().x(x_rear).y(y_right).z(0.0));
  footprint.points.push_back(
    geometry_msgs::build<geometry_msgs::msg::Point32>().x(x_rear).y(y_left).z(0.0));

  return footprint;
}

LinearRing2d ClusterLaneletFilterNode::getConvexHull(
  const tier4_perception_msgs::msg::DetectedObjectsWithFeature & detected_clusters)
{
  MultiPoint2d candidate_points;
  for (const auto & object : detected_clusters.feature_objects) {
    const auto & pos = object.object.kinematics.pose_with_covariance.pose.position;
    const auto footprint = setFootprint(object);
    for (const auto & p : footprint.points) {
      candidate_points.emplace_back(p.x + pos.x, p.y + pos.y);
    }
  }

  LinearRing2d convex_hull;
  boost::geometry::convex_hull(candidate_points, convex_hull);

  return convex_hull;
}

lanelet::ConstLanelets ClusterLaneletFilterNode::getIntersectedLanelets(
  const LinearRing2d & convex_hull, const lanelet::ConstLanelets & road_lanelets)
{
  lanelet::ConstLanelets intersected_lanelets;
  for (const auto & road_lanelet : road_lanelets) {
    if (boost::geometry::intersects(convex_hull, road_lanelet.polygon2d().basicPolygon())) {
      intersected_lanelets.emplace_back(road_lanelet);
    }
  }
  return intersected_lanelets;
}

bool ClusterLaneletFilterNode::isPolygonOverlapLanelets(
  const Polygon2d & polygon, const lanelet::ConstLanelets & intersected_lanelets)
{
  for (const auto & lanelet : intersected_lanelets) {
    if (!boost::geometry::disjoint(polygon, lanelet.polygon2d().basicPolygon())) {
      return true;
    }
  }
  return false;
}

}  // namespace cluster_lanelet_filter

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(cluster_lanelet_filter::ClusterLaneletFilterNode)

