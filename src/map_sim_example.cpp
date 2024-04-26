/**************************************************************************

Copyright <2022> <Gang Chen>

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


Author: Gang Chen

Date: 2021/8/19

Description: This is a ROS example to use the DSP map. The map object is my_map
and updated in Function cloudCallback. We also add some visualization functions
in this file. The visualization results can be viewed with RVIZ.

**************************************************************************/

#include "dsp_dynamic.h" // You can change the head file to "dsp_dynamic_multiple_neighbors.h" or "dsp_static.h" to use different map types. For more information, please refer to the readme file.
#include "gazebo_msgs/ModelStates.h"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/PoseStamped.h"
#include "nav_msgs/Odometry.h"
#include "ros/ros.h"
#include "std_msgs/Float64.h"
#include <geometry_msgs/TwistStamped.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <queue>
#include <sensor_msgs/PointCloud2.h>
#include <tuple>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <boost/make_shared.hpp>
#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/radius_outlier_removal.h> //滤波相关
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>

/// Define a map object
DSPMap my_map;
const float res =
    0.1; // Smaller res will get better tracking result but is slow.
/// Set global variables
queue<double> pose_att_time_queue;
queue<Eigen::Vector3d> uav_position_global_queue;
queue<Eigen::Quaternionf> uav_att_global_queue;
Eigen::Vector3d uav_position_global;
Eigen::Quaternionf uav_att_global;

const unsigned int MAX_POINT_NUM =
    5000; // Estimated max point cloud number after down sample. To define the
          // vector below.
float point_clouds[MAX_POINT_NUM * 3]; // Container for point cloud. We use
                                       // naive vector for efficiency purpose.

// The following range parameters are calculated with map parameters to remove
// the point cloud outside of the map range.
float x_min = -MAP_LENGTH_VOXEL_NUM * VOXEL_RESOLUTION / 2;
float x_max = MAP_LENGTH_VOXEL_NUM * VOXEL_RESOLUTION / 2;
float y_min = -MAP_WIDTH_VOXEL_NUM * VOXEL_RESOLUTION / 2;
float y_max = MAP_WIDTH_VOXEL_NUM * VOXEL_RESOLUTION / 2;
float z_min = -MAP_HEIGHT_VOXEL_NUM * VOXEL_RESOLUTION / 2;
float z_max = MAP_HEIGHT_VOXEL_NUM * VOXEL_RESOLUTION / 2;

ros::Publisher cloud_pub, map_center_pub, gazebo_model_states_pub,
    current_velocity_pub, single_object_velocity_pub,
    single_object_velocity_truth_pub;
ros::Publisher future_status_pub, current_marker_pub, fov_pub, update_time_pub;
gazebo_msgs::ModelStates ground_truth_model_states;

int ground_truth_updated = 0;
bool state_locked = false;

/***
 * Summary: This function is for actor true position visualization
 */
void actor_publish(const vector<Eigen::Vector3d> &actors, int id, float r,
                   float g, float b, float width, int publish_num) {
  if (actors.empty())
    return;

  visualization_msgs::MarkerArray marker_array;

  visualization_msgs::Marker marker;

  marker.header.frame_id = "world";
  marker.header.stamp = ros::Time::now();
  marker.type = visualization_msgs::Marker::CYLINDER;
  marker.action = visualization_msgs::Marker::ADD;
  marker.ns = "actors";

  marker.scale.x = 0.4;
  marker.scale.y = 0.4;
  marker.scale.z = 1.7;
  marker.color.a = 0.6;
  marker.color.r = 0.3;
  marker.color.g = 0.3;
  marker.color.b = 0.9;

  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;

  for (int i = 0; i < actors.size(); ++i) {
    marker.id = i;
    marker.pose.position.x = actors[i].x();
    marker.pose.position.y = actors[i].y();
    marker.pose.position.z = actors[i].z();
    marker_array.markers.push_back(marker);
  }

  current_marker_pub.publish(marker_array);
}

/***
 * Summary: This is function used in showFOV. For visualization.
 */
static void rotateVectorByQuaternion(geometry_msgs::Point &vector,
                                     Eigen::Quaternionf att) {
  // Lazy. Use Eigen directly
  Eigen::Quaternionf ori_vector_quaternion, vector_quaternion;
  ori_vector_quaternion.w() = 0;
  ori_vector_quaternion.x() = vector.x;
  ori_vector_quaternion.y() = vector.y;
  ori_vector_quaternion.z() = vector.z;

  vector_quaternion = att * ori_vector_quaternion * att.inverse();
  vector.x = vector_quaternion.x();
  vector.y = vector_quaternion.y();
  vector.z = vector_quaternion.z();
}

/***
 * Summary: This function is for FOV visualization
 */
void showFOV(Eigen::Vector3d &position, Eigen::Quaternionf &att, double angle_h,
             double angle_v, double length) {
  geometry_msgs::Point p_cam;
  p_cam.x = position(0);
  p_cam.y = position(1);
  p_cam.z = position(2);

  geometry_msgs::Point p1, p2, p3, p4;
  p1.x = length;
  p1.y = length * tan(angle_h / 2);
  p1.z = length * tan(angle_v / 2);
  rotateVectorByQuaternion(p1, att);

  p2.x = -length;
  p2.y = length * tan(angle_h / 2);
  p2.z = length * tan(angle_v / 2);
  rotateVectorByQuaternion(p2, att);

  p3.x = length;
  p3.y = length * tan(angle_h / 2);
  p3.z = -length * tan(angle_v / 2);
  rotateVectorByQuaternion(p3, att);

  p4.x = -length;
  p4.y = length * tan(angle_h / 2);
  p4.z = -length * tan(angle_v / 2);
  rotateVectorByQuaternion(p4, att);

  visualization_msgs::Marker fov;
  fov.header.frame_id = "world";
  fov.header.stamp = ros::Time::now();
  fov.action = visualization_msgs::Marker::ADD;
  fov.ns = "lines_and_points";
  fov.id = 999;
  fov.type = 4;

  fov.scale.x = 0.1;
  fov.scale.y = 0.1;
  fov.scale.z = 0.1;

  fov.color.r = 0.8;
  fov.color.g = 0.5;
  fov.color.b = 0.5;
  fov.color.a = 0.8;
  fov.lifetime = ros::Duration(0);

  fov.points.push_back(p1);
  fov.points.push_back(p2);
  fov.points.push_back(p_cam);
  fov.points.push_back(p4);
  fov.points.push_back(p3);
  fov.points.push_back(p_cam);
  fov.points.push_back(p1);
  fov.points.push_back(p3);
  fov.points.push_back(p4);
  fov.points.push_back(p2);
  fov_pub.publish(fov);
}

/***
 * Summary: This is function used in colorAssign. For visualization.
 */
int inRange(float &low, float &high, float &x) {
  if (x > low && x < high) {
    return 1;
  } else {
    return 0;
  }
}

/***
 * Summary: This function is for future status visualization
 */
void colorAssign(int &r, int &g, int &b, int &a, float v, float value_min = 0.f,
                 float value_max = 1.f, int reverse_color = 0) {
  v = std::max(v, value_min);
  v = std::min(v, value_max);

  float v_range = value_max - value_min;
  int value = floor((v - value_min) / v_range * 240); // Mapping 0~1.0 to 0~240
  value = std::min(value, 240);

  if (reverse_color) {
    value = 240 - value;
  }

  int section = value / 2; // 把0-240等分
  // float float_key = (value % 60) / (float)60 * 255;
  // int key = floor(float_key);
  // int nkey = 255 - key;

  switch (section) {
  case 0: // G increase
    r = 255;
    g = 255;
    b = 0;
    a = 255;
    break;
  default: // White
    r = 0;
    g = 0;
    b = 0;
    a = 0;
  }
}

// void removeEmptySurroundings(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud) {

//   if (cloud == nullptr) {
//     std::cerr << "Error: Input point cloud pointer is null!" << std::endl;
//   }
//   // 检查点云是否为空
//   if (cloud->empty()) {
//     std::cerr << "Error: Input point cloud is empty!" << std::endl;
//     return;
//   }
//   // 创建 KD 树对象
//   pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
//   kdtree.setInputCloud(cloud);

//   // 存储索引的容器
//   std::vector<int> pointIdxRadiusSearch;
//   std::vector<float> pointRadiusSquaredDistance;

//   // 定义半径搜索的半径大小
//   float radius = 0.05; // 调整这个值以适应你的需求

//   // 存储非空点的索引
//   std::vector<int> validIndices;

//   // 遍历点云中的每个点
//   for (size_t i = 0; i < cloud->size(); ++i) {
//     std::cout << "cloud size is:" << cloud->size() << std::endl;
//     std::cout << "Now i is:" << i << std::endl;

//     if (i >= cloud->size()) {
//       std::cerr << "Error: Index out of range!" << std::endl;
//     }

//     // 在半径范围内搜索点
//     if (kdtree.radiusSearch(cloud->points[i], radius, pointIdxRadiusSearch,
//                             pointRadiusSquaredDistance) > 0) {
//       // 如果周围有其他点，则将当前点的索引存储起来
//       if (i < cloud->size() && i > 0) {
//         validIndices.push_back(i);
//         std::cout << "Index is:" << i << std::endl;
//       }
//     }
//   }

//   // 检查是否没有找到有效点
//   if (validIndices.empty()) {
//     std::cerr << "Error: No valid points found in the vicinity!" <<
//     std::endl; return;
//   }

//   // 从点云中提取有效点
//   pcl::ExtractIndices<pcl::PointXYZRGBA> extract;
//   pcl::PointIndices::Ptr indices(new pcl::PointIndices());
//   indices->indices = validIndices;
//   extract.setInputCloud(cloud);
//   extract.setIndices(indices);
//   extract.setNegative(false);
//   std::cout << "5" << std::endl;

//   // if (!cloud) {
//   //   std::cerr << "Error: Input point cloud pointer is null!" << std::endl;
//   //   return;
//   // }
//   // for (size_t i = 0; i < validIndices.size(); ++i) {
//   //   if (validIndices[i] <= 0) {
//   //     std::cerr << "validIndices小于索引值" << std::endl;
//   //   } else if (validIndices[i] > cloud->size()) {
//   //     std::cerr << "validIndices大于索引值" << std::endl;
//   //   }
//   // }

//   extract.filter(*cloud);
//   std::cout << "6" << std::endl;
// }

/***
 * Summary: This is the main callback to update map.
 */
pcl::PointCloud<pcl::PointXYZ>::Ptr
    cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr
    cloud_filtered_ob(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr
    cloud_filtered_future(new pcl::PointCloud<pcl::PointXYZRGBA>());
void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &cloud) {
  /// Simple synchronizer for point cloud data and pose
  Eigen::Vector3d uav_position = uav_position_global;
  Eigen::Quaternionf uav_att = uav_att_global;

  static Eigen::Quaternionf quad_last_popped(-10.f, -10.f, -10.f, -10.f);
  static Eigen::Vector3d position_last_popped(-10000.f, -10000.f, -10000.f);
  static double last_popped_time = 0.0;

  ros::Rate loop_rate(500);
  while (state_locked) {
    loop_rate.sleep();
    ros::spinOnce();
  }
  state_locked = true;

  while (!pose_att_time_queue.empty()) { // Synchronize pose by queue
    double time_stamp_pose = pose_att_time_queue.front();
    if (time_stamp_pose >= cloud->header.stamp.toSec()) {
      uav_att = uav_att_global_queue.front();
      uav_position = uav_position_global_queue.front();

      // linear interpolation
      if (quad_last_popped.x() >= -1.f) {
        double time_interval_from_last_time =
            time_stamp_pose - last_popped_time;
        double time_interval_cloud =
            cloud->header.stamp.toSec() - last_popped_time;
        double factor = time_interval_cloud / time_interval_from_last_time;
        uav_att = quad_last_popped.slerp(factor, uav_att);
        uav_position =
            position_last_popped * (1.0 - factor) + uav_position * factor;
      }

      // ROS_INFO_THROTTLE(3.0, "cloud mismatch time = %lf",
      // cloud->header.stamp.toSec() - time_stamp_pose);

      break;
    }

    quad_last_popped = uav_att_global_queue.front();
    position_last_popped = uav_position_global_queue.front();
    last_popped_time = time_stamp_pose;

    pose_att_time_queue.pop();
    uav_att_global_queue.pop();
    uav_position_global_queue.pop();
  }
  state_locked = false;

  /// Point cloud preprocess
  double data_time_stamp = cloud->header.stamp.toSec();

  // convert cloud to pcl form
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(
      new pcl::PointCloud<pcl::PointXYZ>());
  pcl::fromROSMsg(*cloud, *cloud_in);

  // down-sample for all
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  sor.setInputCloud(cloud_in);
  sor.setLeafSize(res, res, res);
  sor.filter(*cloud_filtered);

  int useful_point_num = 0;
  for (int i = 0; i < cloud_filtered->width; i++) {
    float x = cloud_filtered->points.at(i).z;
    float y = -cloud_filtered->points.at(i).x;
    float z = -cloud_filtered->points.at(i).y;

    if (inRange(x_min, x_max, x) && inRange(y_min, y_max, y) &&
        inRange(z_min, z_max, z)) {
      point_clouds[useful_point_num * 3] = x;
      point_clouds[useful_point_num * 3 + 1] = y;
      point_clouds[useful_point_num * 3 + 2] = z;
      ++useful_point_num;

      if (useful_point_num >= MAX_POINT_NUM) { // In case the buffer overflows
        break;
      }
    }
  }

  /// Update map
  clock_t start1, finish1;
  start1 = clock();

  // std::cout << "uav_position="<<uav_position.x() <<",
  // "<<uav_position.y()<<",
  // "<<uav_position.z()<<endl;

  // This is the core function we use
  if (!my_map.update(useful_point_num, 3, point_clouds, uav_position.x(),
                     uav_position.y(), uav_position.z(), data_time_stamp,
                     uav_att.w(), uav_att.x(), uav_att.y(), uav_att.z())) {
    return;
  }

  /// Display update time
  finish1 = clock();
  double duration1 = (double)(finish1 - start1) / CLOCKS_PER_SEC;
  // printf( "****** Map update time %f seconds\n", duration1);

  static double total_time = 0.0;
  static int update_times = 0;

  total_time += duration1;
  update_times++;
  // printf( "****** Map avg time %f seconds\n \n", total_time /
  // update_times);

  /// Get occupancy status, including future status.
  clock_t start2, finish2;
  start2 = clock();

  int occupied_num = 0;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_to_publish(
      (new pcl::PointCloud<pcl::PointXYZ>));
  sensor_msgs::PointCloud2 cloud_to_pub_transformed;
  static float
      future_status[VOXEL_NUM]
                   [PREDICTION_TIMES]; // future_status[体素数][预测时间]
  /** Note: The future status is stored with voxel structure.
   * The voxels are indexed with one dimension.
   * You can use Function getVoxelPositionFromIndexPublic() to convert index
   *to real position. future_status[*][0] is current status considering delay
   *compensation.
   **/

  my_map.getOccupancyMapWithFutureStatus(occupied_num, *cloud_to_publish,
                                         &future_status[0][0], 0.2);

  /// Publish Point cloud and center position
  // 发布最初状态点云

  pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem_ob; // 创建滤波器
  if (!cloud_to_publish->empty()) {
    outrem_ob.setInputCloud(cloud_to_publish); // 设置输入点云
    outrem_ob.setRadiusSearch(0.3); // 设置半径为0.15的范围内找临近点
    outrem_ob.setMinNeighborsInRadius(2); // 设置查询点的邻域点集数小于2的删除
    // outrem_ob.setNegative(false);

    outrem_ob.filter(
        *cloud_filtered_ob); // 执行条件滤波  在半径为0.15
                             // 在此半径内必须要有两个邻居点，此点才会保存
  }

  pcl::toROSMsg(*cloud_filtered_ob, cloud_to_pub_transformed);
  cloud_to_pub_transformed.header.frame_id = "world";
  cloud_to_pub_transformed.header.stamp = cloud->header.stamp;
  cloud_pub.publish(cloud_to_pub_transformed);

  geometry_msgs::PoseStamped map_pose;
  map_pose.header.stamp = cloud_to_pub_transformed.header.stamp;
  map_pose.pose.position.x = uav_position.x();
  map_pose.pose.position.y = uav_position.y();
  map_pose.pose.position.z = uav_position.z();
  map_pose.pose.orientation.x = uav_att.x();
  map_pose.pose.orientation.y = uav_att.y();
  map_pose.pose.orientation.z = uav_att.z();
  map_pose.pose.orientation.w = uav_att.w();
  map_center_pub.publish(map_pose);

  /// Publish future status of all layers 叠加立体版
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr future_status_cloud(
      new pcl::PointCloud<pcl::PointXYZRGBA>);
  for (int i = 0;
       i < MAP_HEIGHT_VOXEL_NUM * MAP_WIDTH_VOXEL_NUM * MAP_LENGTH_VOXEL_NUM;
       i++) {
    for (int n = 0; n < PREDICTION_TIMES; ++n) {
      pcl::PointXYZRGBA p_this;
      std::vector<std::tuple<float, float, float>> added_positions;
      my_map.getVoxelPositionFromIndexPublic(
          i, p_this.x, p_this.y, p_this.z); // 通过索引获得体素的位置

      float weight_this = future_status[i][n]; // 权重
      int r, g, b, a;
      colorAssign(r, g, b, a, weight_this, 0.f, 0.1f, 1);
      p_this.r = r;
      p_this.g = g;
      p_this.b = b;
      p_this.a = a;
      // 检查是否已经添加过该位置
      bool already_added = false;
      for (const auto &position : added_positions) {
        if (std::get<0>(position) == p_this.x &&
            std::get<1>(position) == p_this.y &&
            std::get<2>(position) == p_this.z) {
          already_added = true;
          break;
        }
      }
      // 选择colorAssign的某种或某几种case显示
      if (!already_added && p_this.r == 255 && p_this.g == 255 &&
          p_this.b == 0 && p_this.a == 255) {
        future_status_cloud->push_back(p_this);
        added_positions.emplace_back(p_this.x, p_this.y,
                                     p_this.z); // 将位置添加到向量中
      }
    }
  }

  pcl::RadiusOutlierRemoval<pcl::PointXYZRGBA> outrem_future; // 创建滤波器
  if (!future_status_cloud->empty()) {
    outrem_future.setInputCloud(future_status_cloud); // 设置输入点云
    outrem_future.setRadiusSearch(0.5); // 设置半径为0.15的范围内找临近点
    outrem_future.setMinNeighborsInRadius(
        2); // 设置查询点的邻域点集数小于2的删除
    // outrem_future.setNegative(false);
    outrem_future.filter(
        *cloud_filtered_future); // 执行条件滤波  在半径为0.15
                                 // 在此半径内必须要有两个邻居点，此点才会保存
  }

  sensor_msgs::PointCloud2 cloud_future_transformed;
  pcl::toROSMsg(*cloud_filtered_future, cloud_future_transformed);
  cloud_future_transformed.header.frame_id = "world";
  cloud_future_transformed.header.stamp = cloud->header.stamp;
  future_status_pub.publish(cloud_future_transformed);

  finish2 = clock();

  double duration2 = (double)(finish2 - start2) / CLOCKS_PER_SEC;
  // printf( "****** Map publish time %f seconds\n \n", duration2);

  /// Publish update time for evaluation tools
  std_msgs::Float64 update_time;
  update_time.data = duration1 + duration2;
  update_time_pub.publish(update_time);
}

/***
 * Summary: This function is used to tell pedestrians' names in
 * simObjectStateCallback
 */
static void split(const string &s, vector<string> &tokens,
                  const string &delimiters = " ") {
  string::size_type lastPos = s.find_first_not_of(delimiters, 0);
  string::size_type pos = s.find_first_of(delimiters, lastPos);
  while (string::npos != pos || string::npos != lastPos) {
    tokens.push_back(s.substr(lastPos, pos - lastPos));
    lastPos = s.find_first_not_of(delimiters, pos);
    pos = s.find_first_of(delimiters, lastPos);
  }
}

/***
 * Summary: Ros callback function to get true position of pedestrians in
 * Gazebo. Just for visualization
 */
void simObjectStateCallback(const gazebo_msgs::ModelStates &msg) {
  ground_truth_model_states = msg;
  ground_truth_updated = 1;

  vector<Eigen::Vector3d> actor_visualization_points;

  for (int i = 0; i < msg.name.size(); ++i) {
    vector<string> name_splited;
    split(msg.name[i], name_splited, "_");
    if (name_splited[0] == "actor") {
      Eigen::Vector3d p;
      p.x() = msg.pose[i].position.x - uav_position_global.x();
      p.y() = msg.pose[i].position.y - uav_position_global.y();
      p.z() = msg.pose[i].position.z - uav_position_global.z();
      actor_visualization_points.push_back(p);
    }
  }
  actor_publish(actor_visualization_points, 6, 1.f, 0.f, 0.f, 0.8, -1);
  gazebo_model_states_pub.publish(ground_truth_model_states);
}

/***
 * Summary: Ros callback function to get pose of the drone (camera) to update
 * map.
 */
void simPoseCallback(const geometry_msgs::PoseStamped &msg) {
  if (!state_locked) {
    state_locked = true;
    uav_position_global.x() = msg.pose.position.x;
    uav_position_global.y() = msg.pose.position.y;
    uav_position_global.z() = msg.pose.position.z;

    uav_att_global.x() = msg.pose.orientation.x;
    uav_att_global.y() = msg.pose.orientation.y;
    uav_att_global.z() = msg.pose.orientation.z;
    uav_att_global.w() = msg.pose.orientation.w;

    uav_position_global_queue.push(uav_position_global);
    uav_att_global_queue.push(uav_att_global);
    pose_att_time_queue.push(msg.header.stamp.toSec());
    // ROS_INFO("Pose updated");
  }

  state_locked = false;

  Eigen::Quaternionf axis; //= quad * q1 * quad.inverse();
  axis.w() = cos(-M_PI / 4.0);
  axis.x() = 0.0;
  axis.y() = 0.0;
  axis.z() = sin(-M_PI / 4.0);
  Eigen::Quaternionf rotated_att = uav_att_global * axis;

  showFOV(uav_position_global, rotated_att, 90.0 / 180.0 * M_PI,
          54.0 / 180.0 * M_PI, 5);

  // Eigen::Quaterniond body_q = Eigen::Quaterniond(msg.pose.orientation.w,
  //                                              msg.pose.orientation.x,
  //                                              msg.pose.orientation.y,
  //                                              msg.pose.orientation.z);
  // my_map.body_r_m = body_q.toRotationMatrix();
  my_map.body_t_m(0) = msg.pose.position.x;
  my_map.body_t_m(1) = msg.pose.position.y;
  my_map.body_t_m(2) = msg.pose.position.z;

  // static tf2_ros::TransformBroadcaster br;
  // geometry_msgs::TransformStamped transformStamped;
  // transformStamped.header.stamp = ros::Time::now();
  // transformStamped.header.frame_id = "world";
  // transformStamped.child_frame_id = "body";
  // transformStamped.transform.translation.x = uav_position_global.x();
  // transformStamped.transform.translation.y = uav_position_global.y();
  // transformStamped.transform.translation.z = uav_position_global.z();

  // transformStamped.transform.rotation.x = uav_att_global.x();
  // transformStamped.transform.rotation.y = uav_att_global.y();
  // transformStamped.transform.rotation.z = uav_att_global.z();
  // transformStamped.transform.rotation.w = uav_att_global.w();

  // br.sendTransform(transformStamped);
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "map_sim_example_with_cluster");
  ros::NodeHandle n;

  /// Map parameters that can be changed dynamically. But usually we still use
  /// them as static parameters.
  my_map.setPredictionVariance(0.05,
                               0.05); // StdDev for prediction. velocity StdDev,
                                      // position StdDev, respectively.
  my_map.setObservationStdDev(0.1);   // StdDev for update. position StdDev.
  // 更改新生成粒子数：20->10
  my_map.setNewBornParticleNumberofEachPoint(
      10); // Number of new particles generated from one measurement point.
  my_map.setNewBornParticleWeight(0.0001); // Initial weight of particles.
  DSPMap::setOriginalVoxelFilterResolution(
      res); // Resolution of the voxel filter used for point cloud
            // pre-process.

  my_map.setParticleRecordFlag(
      1, 15.0); // Set the first parameter to 1 to save particles at a time:
                // e.g. 19.0s. Saving will take a long time. Don't use it in
                // realtime applications.

  /// Gazebo pedestrain's pose. Just for visualization
  ros::Subscriber object_states_sub =
      n.subscribe("/gazebo/model_states", 1, simObjectStateCallback);

  /// Input data for the map
  ros::Subscriber point_cloud_sub = n.subscribe(
      "/iris_D435i/realsense/depth_camera/depth/points", 1, cloudCallback);
  ros::Subscriber pose_sub =
      n.subscribe("/mavros/local_position/pose", 1, simPoseCallback);

  /// Visualization topics
  cloud_pub =
      n.advertise<sensor_msgs::PointCloud2>("/my_map/cloud_ob", 1, true);
  map_center_pub =
      n.advertise<geometry_msgs::PoseStamped>("/my_map/map_center", 1, true);
  gazebo_model_states_pub =
      n.advertise<gazebo_msgs::ModelStates>("/my_map/model_states", 1, true);

  future_status_pub = n.advertise<sensor_msgs::PointCloud2>(
      "/my_map/future_status", 1,
      true); // 发布sensor_msgs::PointCloud2形式的点云地图消息

  current_velocity_pub = n.advertise<visualization_msgs::MarkerArray>(
      "/my_map/velocity_marker", 1);
  single_object_velocity_pub = n.advertise<geometry_msgs::TwistStamped>(
      "/my_map/single_object_velocity", 1);
  single_object_velocity_truth_pub = n.advertise<geometry_msgs::TwistStamped>(
      "/my_map/single_object_velocity_ground_truth", 1);
  current_marker_pub =
      n.advertise<visualization_msgs::MarkerArray>("/visualization_marker", 1);
  fov_pub = n.advertise<visualization_msgs::Marker>("/visualization_fov", 1);

  update_time_pub = n.advertise<std_msgs::Float64>("/map_update_time", 1);

  // while (ros::ok())
  // {
  //     tf_body2world(br);
  // }

  /// Ros spin
  ros::AsyncSpinner spinner(3); // Use 3 threads
  spinner.start();
  ros::waitForShutdown();

  return 0;
}
