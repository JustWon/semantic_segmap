# from IMU, for later use
time time_stamp

# lidar point cloud
sensor_msgs/PointCloud2 point_cloud

# from IMU, for later use
sensor_msgs/Imu odometry

# per-point object id
# -1 if the point belongs to no object
int32[] object_id

# per-point semantic id
#-1: Misc
# 0: Car
# 1: Pedestrian
# 2: Cyclist
# 3: Ground
int32[] semantic_id