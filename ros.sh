#!/bin/bash

source /opt/ros/humble/setup.bash

# ROS2 + FastDDS Discovery Server client settings
# REQUIRES BOTH: XML config file AND environment variable
export FASTRTPS_DEFAULT_PROFILES_FILE=$PWD/scripts/real_robot/super_client_configuration_file.xml
export ROS_DISCOVERY_SERVER="${ROS_DISCOVERY_SERVER:-10.0.0.30:11811}"  # R1 Pro onboard discovery server
export ROS_DOMAIN_ID=0
unset ROS_LOCALHOST_ONLY LOCAL_HOST_ONLY 2>/dev/null
