import cv2 as cv
import numpy as np
from collections import deque, Counter


class VideoProcessor:
	"""Reads frames from a video source, detects face/eyes and computes tilt majority.

	Methods
	-------
	read_action()
		Reads one frame, updates sliding window and returns (action, quit_flag).
		action: -1 (left), 0 (straight/uncertain), 1 (right), None (no frame/read error)
		quit_flag: True if user requested exit (ESC) or capture failed.
	release()
		Releases resources.
	"""

	def __init__(self, source=0, window_size=10, show=True):
		self.capture = cv.VideoCapture(source)
		if not self.capture or not self.capture.isOpened():
			raise RuntimeError(f"Could not open video source: {source}")

		# load cascades
		self.face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
		self.eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
		if self.face_cascade.empty() or self.eye_cascade.empty():
			raise RuntimeError("Haar cascade xml files not found. Ensure opencv-python is installed.")

		self.tilt_history = deque(maxlen=window_size)
		self.last_majority = None
		self.show = show

	def _label_from_angle(self, angle_deg):
		if angle_deg > 10:
			return "right"
		elif angle_deg < -10:
			return "left"
		else:
			return "straight"

	def _majority_label(self):
		if len(self.tilt_history) < self.tilt_history.maxlen:
			return None, None
		counts = Counter(self.tilt_history)
		most_common_label, most_common_count = counts.most_common(1)[0]
		# require strict majority and prefer non-'none'
		if most_common_label != "none" and most_common_count > (self.tilt_history.maxlen // 2):
			return most_common_label, most_common_count
		return "uncertain", most_common_count

	def _map_label_to_action(self, label):
		if label == "left":
			return -1
		if label == "right":
			return 1
		# uncertain / straight / none -> 0
		return 0

	def read_action(self):
		"""Read one frame, update sliding window, return (action, quit_flag).

		- action: -1 (left), 0 (straight/uncertain), 1 (right), None when frame read failed
		- quit_flag: True when ESC pressed or capture failed
		"""
		ret, frame = self.capture.read()
		if not ret or frame is None:
			return None, True

		gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)

		if len(faces) == 0:
			tilt_label = "none"
		else:
			x, y, w, h = faces[0]
			roi_gray = gray[y:y + h, x:x + w]
			eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 4)

			eye_boxes = []
			for (ex, ey, ew, eh) in eyes:
				ex_abs, ey_abs = x + ex, y + ey
				eye_boxes.append((ex_abs, ey_abs, ew, eh))

			if len(eye_boxes) >= 2:
				eye_boxes.sort(key=lambda b: b[0])
				left_eye = eye_boxes[0]
				right_eye = eye_boxes[1]
				left_eye_center = (int(left_eye[0] + left_eye[2] / 2), int(left_eye[1] + left_eye[3] / 2))
				right_eye_center = (int(right_eye[0] + right_eye[2] / 2), int(right_eye[1] + right_eye[3] / 2))
				lx, ly = left_eye_center
				rx, ry = right_eye_center
				delta_x = rx - lx
				delta_y = ry - ly
				angle_rad = np.arctan2(delta_y, delta_x)
				angle_deg = np.degrees(angle_rad)
				tilt_label = self._label_from_angle(angle_deg)
				# draw annotation
				if self.show:
					cv.putText(frame, f'{tilt_label.upper()} : {int(angle_deg)} deg', (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_4)
			else:
				tilt_label = "none"

		self.tilt_history.append(tilt_label)

		majority_label, majority_count = self._majority_label()
		action = None
		if majority_label is None:
			if self.show:
				cv.putText(frame, f'Collecting: {len(self.tilt_history)}/{self.tilt_history.maxlen}', (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2, cv.LINE_AA)
			action = 0
		else:
			if majority_label == "uncertain":
				action = 0
				if self.show:
					cv.putText(frame, f'MAJORITY: UNCERTAIN ({majority_count}/{self.tilt_history.maxlen})', (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2, cv.LINE_AA)
			else:
				action = self._map_label_to_action(majority_label)
				if self.show:
					cv.putText(frame, f'MAJORITY: {majority_label.upper()} ({majority_count}/{self.tilt_history.maxlen})', (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 0), 2, cv.LINE_AA)

			# print only when majority changes
			if majority_label != self.last_majority:
				print("Majority tilt ->", majority_label)
				self.last_majority = majority_label

		if self.show:
			cv.imshow('Frame', frame)
			key = cv.waitKey(1) & 0xFF
			if key == 27:
				return action, True

		return action, False

	def release(self):
		try:
			if self.capture and self.capture.isOpened():
				self.capture.release()
		finally:
			cv.destroyAllWindows()

