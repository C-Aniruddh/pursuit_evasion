#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
import message_filters
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge, CvBridgeError

import os
import sys
import numpy as np

from detector import ObjectDetector


class person_follower:

	def __init__(self):
		print('ROS_OpenCV_bridge initialized')
		self.image_pub = rospy.Publisher("/CV_image", Image, queue_size=5)
		self.cmd_vel_pub = rospy.Publisher("/tb3_0/cmd_vel", Twist, queue_size=5)
		self.bridge = CvBridge()
        self.object_detector = ObjectDetector()

		image_sub = rospy.Subscriber("/tb3_0/camera/rgb/image_raw", Image, self.callback)

	def callback(self, ros_image):
		print("[INFO] Received new image from ROS")
		try:
			cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
		except CvBridgeError as e:
			print(e)
		# print(type(cv_image))     # <type 'numpy.ndarray'>

		output_dict = self.run_inference_for_single_image(cv_image)
		self.draw_output(cv_image, output_dict)

		try:
			self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
		except CvBridgeError as e:
			print(e)

		# >>> Follow the person >>>
		# self.follow_person(output_dict)
	
	def in_pursuit_evador(self, output_dict):
		move_cmd = Twist()		# all values are 0.0 by default

		index = -1
		for i in range(len(output_dict['detection_classes'])):
			if (output_dict['detection_classes'][i] == 1 and output_dict['detection_scores'][i] > 0.4):		# if we detected a person
				index = i	# keep the index
				break		# break the for loop

		if (index > -1):   # if we found a person
			box = output_dict['detection_boxes'][index]
			box_center = (box[1] + box[3])/2.0

			# 0.872 radians is 50 degrees
			move_cmd.angular.z = -1.1 * (box_center - 0.5)	# subtract 0.5 so that it is a value between -0.5 and 0.5
			move_cmd.linear.x = 0.21

			debug = False
			debug = True
			if(debug):
				print('box_center: ', box_center)
				print('angular.z: {:.2f}'.format(move_cmd.angular.z))
					
			# >>> publish the movement speeds >>>
			self.cmd_vel_pub.publish(move_cmd)

				

def main(args):
	rospy.init_node('person_follower', anonymous=False)  # we only need one of these nodes so make anonymous=False
	pf = person_follower()
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)
