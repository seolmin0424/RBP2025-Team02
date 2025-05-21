import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header

def detect_monitor(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    kernel = np.ones((5, 5), np.uint8)
    mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel)
    mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        eps = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) == 4 and cv2.contourArea(cnt) >= 2000:
            x, y, w, h = cv2.boundingRect(cnt)
            return image[y:y+h, x:x+w]
    return None

def determine_color(image):
    h, w = image.shape[:2]
    roi = image[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    def mask_area(mask):
        kernel = np.ones((5,5), np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.countNonZero(cleaned)

    r1 = cv2.inRange(hsv, np.array([0,50,50]), np.array([20,255,255]))
    r2 = cv2.inRange(hsv, np.array([160,50,50]), np.array([180,255,255]))
    mask_r = cv2.bitwise_or(r1, r2)
    mask_g = cv2.inRange(hsv, np.array([40,50,50]), np.array([80,255,255]))
    mask_b = cv2.inRange(hsv, np.array([100,50,50]), np.array([140,255,255]))

    red_area = mask_area(mask_r)
    green_area = mask_area(mask_g)
    blue_area = mask_area(mask_b)
    total_area = roi.shape[0] * roi.shape[1]

    ratios = {
        "R": red_area / total_area,
        "G": green_area / total_area,
        "B": blue_area / total_area
    }

    dominant_color = max(ratios, key=ratios.get)
    if ratios[dominant_color] < 0.05:
        return "G"
    return dominant_color


class DetermineColor(Node):
    def __init__(self):
        super().__init__('color_detector')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.callback, 10)
        self.color_pub = self.create_publisher(Header, '/rotate_cmd', 10)

    def callback(self, data):
        try:
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            msg = Header()
            msg.stamp = data.header.stamp
            msg.frame_id = '0'
            roi = detect_monitor(image)
            if roi is None:
                roi = image
            color = determine_color(roi)
            if color == "R":
                msg.frame_id = '-1'
            elif color == "B":
                msg.frame_id = '+1'
            else:
                msg.frame_id = '0'
            self.color_pub.publish(msg)
        except CvBridgeError:
            self.get_logger().error('Failed to convert image')

if __name__ == '__main__':
    rclpy.init()
    node = DetermineColor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

