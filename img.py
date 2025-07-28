import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import os

class RGBDProcessor(Node):
    def __init__(self):
        super().__init__('rgbd_processor')
        self.bridge = CvBridge()
        self.color_info = None
        self.depth_info = None
        self.last_depth_image = None
        self.palm_lower_hsv = np.array([25, 25, 25])
        self.palm_upper_hsv = np.array([95, 255, 255])
        self.palm_detection_threshold = 45.0
        self.palm_color = (255, 140, 0)
        self.non_palm_color = (0, 0, 255)
        self.text_bg_alpha = 0.7
        self.create_subscription(Image, "/color/image_raw", self.color_callback, 10)
        self.create_subscription(CameraInfo, "/color/camera_info", self.color_info_callback, 10)
        self.create_subscription(Image, "/depth/image_raw", self.depth_callback, 10)
        self.create_subscription(CameraInfo, "/depth/camera_info", self.depth_info_callback, 10)
        os.makedirs("results", exist_ok=True)
        self.get_logger().info("RGB-D processor node started with enhanced palm tree detection.")

    def color_info_callback(self, msg):
        self.color_info = msg

    def depth_info_callback(self, msg):
        self.depth_info = msg

    def color_callback(self, msg):
        if self.color_info is None:
            self.get_logger().warn("No color camera info received yet.")
            return
        color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        os.makedirs("report_results", exist_ok=True)
        cv2.imwrite(os.path.join("report_results", "a_original_rgb_image.png"), color_image)
        centroids, regions, debug_image = self.process_image_array(color_image, min_area=100, output_dir="report_results")
        if not centroids:
            self.get_logger().info("No objects detected.")
            return
        h, w = color_image.shape[:2]
        num_detections = len(centroids)
        summary_height = max(200, 50 + (num_detections * 40))
        result_image = np.zeros((h + summary_height, w, 3), dtype=np.uint8)
        result_image[:h, :w] = debug_image.copy()
        detection_info = []
        palm_tree_counter = 1
        non_palm_tree_counter = 1
        for i, ((u, v), region) in enumerate(zip(centroids, regions)):
            is_palm, confidence = self.is_palm_tree(color_image, region)
            if is_palm:
                label = f"Palm Tree {palm_tree_counter}"
                tree_id = palm_tree_counter
                color = self.palm_color
                status = "DETECTED"
                palm_tree_counter += 1
            else:
                label = f"Tree {non_palm_tree_counter}"
                tree_id = non_palm_tree_counter
                color = self.non_palm_color
                status = "DETECTED"
                non_palm_tree_counter += 1
            x, y, w_box, h_box = region
            cv2.rectangle(result_image, (x, y), (x+w_box, y+h_box), color, 3)
            depth_text = "No Depth"
            point_3d = (0, 0, 0)
            if (self.last_depth_image is not None and 
                0 <= v < self.last_depth_image.shape[0] and 
                0 <= u < self.last_depth_image.shape[1]):
                depth = self.last_depth_image[v, u] / 1000.0
                point_3d = self.deproject_pixel(u, v, depth)
                depth_text = f"({point_3d[0]:.2f}, {point_3d[1]:.2f}, {point_3d[2]:.2f}m)"
            detection_info.append({
                'id': tree_id,
                'label': label,
                'confidence': confidence,
                'position': depth_text,
                'coordinates_3d': point_3d,
                'status': status,
                'color': color,
                'is_palm': is_palm
            })
            label_text = label
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            cv2.rectangle(result_image, 
                         (x, y - text_height - 10), 
                         (x + text_width + 10, y), 
                         color, -1)
            cv2.putText(result_image, label_text, (x + 5, y - 5), 
                       font, font_scale, (255, 255, 255), thickness)
        panel_start_y = h + 20
        cv2.putText(result_image, "DETECTION SUMMARY:", (10, panel_start_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        for i, info in enumerate(detection_info):
            y_pos = panel_start_y + 40 + (i * 45)
            status_text = f"{info['label']}: {info['status']}"
            cv2.putText(result_image, status_text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, info['color'], 2)
            conf_text = f"Confidence: {info['confidence']:.1f}%"
            cv2.putText(result_image, conf_text, (250, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(result_image, info['position'], (450, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            self.get_logger().info(f"{info['label']}: {info['status']} - {info['confidence']:.1f}% at {info['position']}")
        cv2.imwrite(os.path.join("report_results", "c_final_detection_with_results.png"), result_image)
        self.save_detection_summary(detection_info)
        cv2.imshow("Palm Tree Detection", result_image)
        cv2.waitKey(1)
        
    def depth_callback(self, msg):
        if self.color_info is None:
            return
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.last_depth_image = cv2.resize(depth_image, 
                                          (self.color_info.width, self.color_info.height), 
                                          interpolation=cv2.INTER_LINEAR)

    def deproject_pixel(self, u, v, depth):
        if self.depth_info is None or self.color_info is None:
            return (0, 0, depth)
        fx = self.color_info.k[0]
        fy = self.color_info.k[4]
        cx = self.color_info.k[2]
        cy = self.color_info.k[5]
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        return (x, y, z)
    
    def is_palm_tree(self, img, region):
        x, y, w, h = region
        y = max(0, y)
        x = max(0, x)
        h = min(h, img.shape[0] - y)
        w = min(w, img.shape[1] - x)
        if w <= 0 or h <= 0:
            return False, 0
        object_img = img[y:y+h, x:x+w]
        score = 0
        hsv_img = cv2.cvtColor(object_img, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv_img, self.palm_lower_hsv, self.palm_upper_hsv)
        green_percentage = (np.sum(green_mask > 0) / (w * h)) * 100
        color_score = min(100, green_percentage * 1.8)
        gray = cv2.cvtColor(object_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shape_score = 0
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            circularity_score = 100 * (1 - abs(0.5 - circularity) * 2)
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            solidity_score = 100 * (1 - abs(0.65 - solidity) * 2)
            shape_score = (circularity_score + solidity_score) / 2
        M = cv2.moments(thresh)
        radial_score = 0
        if M["m00"] > 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            mask = np.zeros_like(thresh)
            for angle in range(0, 360, 5):
                radian = np.deg2rad(angle)
                end_x = center_x + int(np.cos(radian) * max(w, h))
                end_y = center_y + int(np.sin(radian) * max(w, h))
                cv2.line(mask, (center_x, center_y), (end_x, end_y), 255, 1)
            intersection = cv2.bitwise_and(thresh, mask)
            intersection_count = np.sum(intersection > 0)
            mask_count = np.sum(mask > 0)
            if mask_count > 0:
                intersection_ratio = intersection_count / mask_count
                radial_score = min(100, intersection_ratio * 600)
        gabor_responses = []
        for theta in np.arange(0, np.pi, np.pi/8):
            kernel = cv2.getGaborKernel((21, 21), 5.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            gabor_img = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            gabor_responses.append(np.std(gabor_img))
        texture_complexity = np.std(gabor_responses)
        texture_score = min(100, max(0, 100 - abs(texture_complexity - 20) * 10))
        frond_score = 0
        if contours and len(contours) > 0:
            frond_candidates = 0
            for contour in contours:
                if cv2.contourArea(contour) < 50:
                    continue
                x, y, w_box, h_box = cv2.boundingRect(contour)
                aspect_ratio = float(w_box) / h_box if h_box > 0 else 0
                if 0.1 < aspect_ratio < 10:
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    contour_area = cv2.contourArea(contour)
                    if hull_area > 0:
                        concavity = contour_area / hull_area
                        if concavity < 0.85:
                            frond_candidates += 1
            frond_score = min(100, frond_candidates * 15)
        pine_score = self.detect_pine_characteristics(object_img)
        palm_specific_score = self.detect_palm_specific_features(object_img)
        final_score = (0.20 * color_score + 0.15 * shape_score +
                      0.30 * radial_score + 0.15 * texture_score +
                      0.10 * frond_score + 0.15 * palm_specific_score
                      - 0.05 * pine_score)
        is_palm = final_score > self.palm_detection_threshold
        debug_img = object_img.copy()
        cv2.putText(debug_img, f"Green: {green_percentage:.1f}%", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(debug_img, f"Shape: {shape_score:.1f}", (5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(debug_img, f"Radial: {radial_score:.1f}", (5, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        cv2.putText(debug_img, f"Texture: {texture_score:.1f}", (5, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(debug_img, f"Frond: {frond_score:.1f}", (5, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        cv2.putText(debug_img, f"Pine: {pine_score:.1f}", (5, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(debug_img, f"Palm: {palm_specific_score:.1f}", (5, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(debug_img, f"Score: {final_score:.1f}", (5, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        os.makedirs("results", exist_ok=True)
        cv2.imwrite(os.path.join("results", f"object_analysis_{x}_{y}.png"), debug_img)
        return is_palm, final_score
    
    def detect_pine_characteristics(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pine_indicators = 0
        texture_variance = np.var(gray)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.8:
                    pine_indicators += 40
                if texture_variance < 100:
                    pine_indicators += 30
        return min(100, pine_indicators)
    
    def detect_palm_specific_features(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        palm_indicators = 0
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant_contours = [c for c in contours if cv2.contourArea(c) > 50]
        if len(significant_contours) >= 4:
            palm_indicators += 30
        elongated_shapes = 0
        for contour in significant_contours:
            x, y, w_box, h_box = cv2.boundingRect(contour)
            aspect_ratio = max(w_box, h_box) / min(w_box, h_box) if min(w_box, h_box) > 0 else 0
            if aspect_ratio > 1.5:
                elongated_shapes += 1
        if elongated_shapes >= 3:
            palm_indicators += 30
        h, w = gray.shape
        center_x, center_y = w // 2, h // 2
        radial_consistency = 0
        for angle in range(0, 360, 45):
            radian = np.deg2rad(angle)
            end_x = int(center_x + np.cos(radian) * min(w, h) // 4)
            end_y = int(center_y + np.sin(radian) * min(w, h) // 4)
            if 0 <= end_x < w and 0 <= end_y < h:
                if thresh[end_y, end_x] > 0:
                    radial_consistency += 1
        if radial_consistency >= 4:
            palm_indicators += 40
        return min(100, palm_indicators)
    
    def save_detection_summary(self, detection_info):
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary_content = f"""
PALM TREE DETECTION RESULTS SUMMARY
=====================================
Generated: {timestamp}

DETECTION OVERVIEW:
Total Objects Detected: {len(detection_info)}
Palm Trees: {sum(1 for info in detection_info if info['is_palm'])}
Other Trees: {sum(1 for info in detection_info if not info['is_palm'])}

DETAILED RESULTS:
"""
        for info in detection_info:
            tree_type = "PALM TREE" if info['is_palm'] else "TREE"
            summary_content += f"""
{info['label']}:
  - Type: {tree_type}
  - Detection Status: {info['status']}
  - Confidence Score: {info['confidence']:.2f}%
  - 3D Coordinates: {info['coordinates_3d']}
  - Position String: {info['position']}
"""
        summary_content += f"""
DETECTION PARAMETERS:
- Palm Detection Threshold: {self.palm_detection_threshold}
- HSV Color Range: {self.palm_lower_hsv} to {self.palm_upper_hsv}
- Minimum Object Area: 100 pixels

FILES GENERATED FOR REPORT:
(a) a_original_rgb_image.png - Original RGB input image
(b) Edge Detection Steps:
    - b1_grayscale.png - Grayscale conversion
    - b2_gaussian_blur.png - Gaussian blur preprocessing
    - b3_canny_edges.png - Canny edge detection
    - b4_morphology_close.png - Morphological closing
    - b5_final_edges.png - Final edge image after dilation
    - b6_detected_contours.png - Detected contours with bounding boxes
(c) c_final_detection_with_results.png - Final detection with confidence scores and 3D coordinates

This summary file: detection_summary.txt
"""
        with open(os.path.join("report_results", "detection_summary.txt"), 'w') as f:
            f.write(summary_content)
        self.get_logger().info(f"Detection summary saved to report_results/detection_summary.txt")
    
    def process_image_array(self, img, min_area=100, output_dir="report_results"):
        os.makedirs(output_dir, exist_ok=True)
        img = cv2.resize(img, (640, 480))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(output_dir, "b1_grayscale.png"), gray)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        cv2.imwrite(os.path.join(output_dir, "b2_gaussian_blur.png"), blurred)
        edges = cv2.Canny(blurred, 15, 75)
        cv2.imwrite(os.path.join(output_dir, "b3_canny_edges.png"), edges)
        kernel = np.ones((13, 13), np.uint8)
        cleaned_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(os.path.join(output_dir, "b4_morphology_close.png"), cleaned_edges)
        cleaned_edges = cv2.dilate(cleaned_edges, np.ones((3, 3), np.uint8), iterations=1)
        cv2.imwrite(os.path.join(output_dir, "b5_final_edges.png"), cleaned_edges)
        contours, _ = cv2.findContours(cleaned_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centroids = []
        regions = []
        output = img.copy()
        if len(contours) > 0:
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                regions.append((x, y, w_box, h_box))
                cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
                cv2.rectangle(output, (x, y), (x+w_box, y+h_box), (255, 0, 0), 2)
                cv2.circle(output, (cx, cy), 4, (0, 0, 255), -1)
            cv2.imwrite(os.path.join(output_dir, "b6_detected_contours.png"), output)
            return centroids, regions, output    
        return [], [], img

def main(args=None):
    rclpy.init(args=args)
    node = RGBDProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

