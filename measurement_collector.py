#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
import time
from datetime import datetime

class SimpleMeasurementCollector(Node):
    def __init__(self):
        super().__init__('measurement_collector')
        self.bridge = CvBridge()
        self.measurements = []
        
        self.color_info = None
        self.last_depth_image = None
        self.last_color_image = None
        
        self.create_subscription(Image, "/color/image_raw", self.color_callback, 10)
        self.create_subscription(CameraInfo, "/color/camera_info", self.color_info_callback, 10)
        self.create_subscription(Image, "/depth/image_raw", self.depth_callback, 10)
        
        self.create_timer(3.0, self.collect_data)
        
        print("🚁 Measurement collector started!")
        print("Looking for palm trees...")

    def color_info_callback(self, msg):
        self.color_info = msg

    def color_callback(self, msg):
        self.last_color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def depth_callback(self, msg):
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if self.color_info:
            self.last_depth_image = cv2.resize(depth_image, (self.color_info.width, self.color_info.height))

    def find_palm_trees(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([25, 25, 25])
        upper_green = np.array([95, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        palm_centers = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    palm_centers.append((cx, cy))
        
        return palm_centers

    def simulate_stereo_depth(self, real_depth):
        baseline = 0.3
        focal_px = 678
        
        noise = np.random.normal(0, 1.0)
        disparity = (baseline * focal_px) / real_depth
        noisy_disparity = disparity + noise
        
        if noisy_disparity > 0:
            return (baseline * focal_px) / noisy_disparity
        return None

    def collect_data(self):
        if self.last_color_image is None or self.last_depth_image is None:
            print("⏳ Waiting for camera data...")
            return
        
        palm_centers = self.find_palm_trees(self.last_color_image)
        
        if not palm_centers:
            print("🔍 No palm trees detected")
            return
        
        print(f"🌴 Found {len(palm_centers)} palm trees!")
        
        for i, (cx, cy) in enumerate(palm_centers):
            real_depth = self.last_depth_image[cy, cx] / 1000.0
            
            if real_depth > 0:
                stereo_depth = self.simulate_stereo_depth(real_depth)
                
                if stereo_depth:
                    error = abs(real_depth - stereo_depth)
                    rel_error = (error / real_depth) * 100
                    
                    measurement = {
                        'timestamp': datetime.now().isoformat(),
                        'tree_id': i + 1,
                        'pixel_x': cx,
                        'pixel_y': cy,
                        'd435_depth': real_depth,
                        'stereo_depth': stereo_depth,
                        'absolute_error': error,
                        'relative_error': rel_error
                    }
                    
                    self.measurements.append(measurement)
                    
       

def main():
    rclpy.init()
    collector = SimpleMeasurementCollector()
    
    try:
      
        rclpy.spin(collector)
    except KeyboardInterrupt:
        print("\n Stopping...")
        filename = collector.save_data()
        if filename:
            print(f"  Run: python3 analyze.py {filename}")
    finally:
        collector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

