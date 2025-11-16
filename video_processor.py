import cv2 as cv
import numpy as np
from collections import deque, Counter


class VideoProcessor:
	"""Reads frames from a video source, detects face/eyes and computes tilt majority.

	Methods
	-------
	read_action()
		Reads one frame, updates sliding window and returns (action, quit_flag).
		action: -1 (right), 0 (straight/uncertain), 1 (left), None (no frame/read error)
		quit_flag: True if user requested exit (ESC) or capture failed.
	release()
		Releases resources.
	"""

	def __init__(self, source=0, window_size=10, show=True):
		self.capture = cv.VideoCapture(source)
		if not self.capture or not self.capture.isOpened():
			raise RuntimeError(f"Could not open video source: {source}")

		# load cascades
		self.face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
		self.eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
		self.nose_cascade = cv.CascadeClassifier('haarcascade_mcs_nose.xml')
		if self.face_cascade.empty() or self.eye_cascade.empty():
			raise RuntimeError("Haar cascade xml files not found. Ensure opencv-python is installed.")
		if self.nose_cascade.empty():
			print("Warning: nose cascade 'haarcascade_mcs_nose.xml' not found, nose detection disabled.")
			self.nose_cascade = None

		self.tilt_history = deque(maxlen=window_size)
		self.last_majority = None
		self.show = show

	def _label_from_angle(self, angle_deg):
		if angle_deg > 10:
			return "left"
		elif angle_deg < -10:
			return "right"
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
			return 1
		if label == "right":
			return -1
		# uncertain / straight / none -> 0
		return 0

	# small helper to draw a simple tilt meter inset
	def _draw_tilt_meter(self, frame, angle_deg, topleft=(20, 90), size=(140, 40)):
		x, y = topleft
		w, h = size
		# background
		cv.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), -1)
		# three zones: left, center, right
		cv.rectangle(frame, (x + 2, y + 2), (x + w//3 - 2, y + h - 2), (180, 50, 50), -1)    # left zone (red-ish)
		cv.rectangle(frame, (x + w//3 + 1, y + 2), (x + 2*w//3 - 1, y + h - 2), (80, 180, 80), -1)  # center (green)
		cv.rectangle(frame, (x + 2*w//3 + 2, y + 2), (x + w - 2, y + h - 2), (50, 50, 180), -1)  # right (blue-ish)

		# clamp angle to reasonable range for meter mapping
		clamped = max(-45.0, min(45.0, angle_deg))
		# map -45..45 to meter inner width
		inner_x0 = x + 4
		inner_x1 = x + w - 4
		pos = int(np.interp(clamped, [-45.0, 45.0], [inner_x0, inner_x1]))
		# marker
		cv.line(frame, (pos, y + 4), (pos, y + h - 4), (255, 255, 255), 2)
		cv.putText(frame, f'{int(angle_deg)} deg', (x + w + 8, y + h - 6), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv.LINE_AA)

	def read_action(self):
		"""Read one frame, update sliding window, return (action, quit_flag).

		- action: 1 (left), 0 (straight/uncertain), -1 (right), None when frame read failed
		- quit_flag: True when ESC pressed or capture failed
		"""
		ret, frame = self.capture.read()
		if not ret or frame is None:
			return None, True

		gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)

		angle_deg = 0.0  # default angle for meter when not computed
		if len(faces) == 0:
			tilt_label = "none"
		else:
			x, y, w, h = faces[0]
			# draw face rectangle for visualization
			if self.show:
				cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
				cv.circle(frame, (x + w // 2, y + h // 2), 3, (0, 255, 0), -1)
				
			roi_gray = gray[y:y + h, x:x + w]
			
			# nose detection (greatly reduces likelihood of classifying nostrils as eyes)
			nose_rect = None
			if self.nose_cascade is not None:
				noses = self.nose_cascade.detectMultiScale(roi_gray, 1.1, 5)
				# draw first detected nose (convert to full-frame coords)
				if len(noses) > 0 and self.show:
					nx, ny, nw, nh = noses[0]
					nx_abs, ny_abs = x + nx, y + ny
					nose_rect = (nx_abs, ny_abs, nw, nh)
					cv.rectangle(frame, (nx_abs, ny_abs), (nx_abs + nw, ny_abs + nh), (255, 255, 0), 2)
					cv.circle(frame, (nx_abs + nw // 2, ny_abs + nh // 2), 3, (255, 255, 0), -1)
				elif len(noses) > 0:
					# still record nose rect even if not drawing
					nx, ny, nw, nh = noses[0]
					nx_abs, ny_abs = x + nx, y + ny
					nose_rect = (nx_abs, ny_abs, nw, nh)

			eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 4)

			eye_boxes = []
			# helper to test whether a point is inside a rect
			def _point_in_rect(px, py, rect):
				rx, ry, rw, rh = rect
				return (px >= rx) and (px <= rx + rw) and (py >= ry) and (py <= ry + rh)

			for (ex, ey, ew, eh) in eyes:
				ex_abs, ey_abs = x + ex, y + ey
				cx, cy = ex_abs + ew // 2, ey_abs + eh // 2
				# skip eyes whose center lies inside the detected nose bounding box
				if nose_rect is not None and _point_in_rect(cx, cy, nose_rect):
					continue
				eye_boxes.append((ex_abs, ey_abs, ew, eh))
				# draw eye rectangles
				if self.show:
					cv.rectangle(frame, (ex_abs, ey_abs), (ex_abs + ew, ey_abs + eh), (0, 0, 255), 2)

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

				# draw visualization elements
				if self.show:
					# centers
					cv.circle(frame, left_eye_center, 4, (0, 255, 0), -1)
					cv.circle(frame, right_eye_center, 4, (0, 255, 0), -1)

					# line between eyes (primary measurement)
					cv.line(frame, left_eye_center, right_eye_center, (255, 0, 0), 2)

					# midpoint between eyes
					midx, midy = (lx + rx) // 2, (ly + ry) // 2
					cv.circle(frame, (midx, midy), 3, (0, 255, 255), -1)

					# vertical reference line from midpoint (shows tilt relative to vertical)
					ref_len = max(30, int(0.3 * h))
					cv.line(frame, (midx, midy - ref_len), (midx, midy + ref_len), (200, 200, 200), 1, cv.LINE_AA)

					# arrow showing measured tilt direction (from midpoint in the direction of the eyes line)
					arrow_len = max(40, int(0.4 * w))
					end_x = int(midx + arrow_len * np.cos(angle_rad))
					end_y = int(midy + arrow_len * np.sin(angle_rad))
					cv.arrowedLine(frame, (midx, midy), (end_x, end_y), (0, 200, 255), 2, tipLength=0.2)

					# small angle arc near midpoint (visual cue)
					arc_radius = 30
					# compute start and end angles for cv.ellipse (degrees)
					# angle of the line in degrees (cv ellipse uses degrees with 0 at x-axis)
					ellipse_angle = -angle_deg
					cv.ellipse(frame, (midx, midy), (arc_radius, arc_radius), 0, 0, -int(angle_deg), (180, 180, 0), 2)

					# text label
					cv.putText(frame, f'{tilt_label.upper()} : {int(angle_deg)} deg', (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_4)

			else:
				tilt_label = "none"

		# append label (use "none" when no reliable eyes detected)
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

		# draw tilt meter inset (visualization of current measured angle)
		if self.show:
			self._draw_tilt_meter(frame, angle_deg, topleft=(20, 90), size=(140, 40))

		if self.show:
			cv.imshow('Frame', frame)
			key = cv.waitKey(1) & 0xFF
			if key == 27:
				return majority_label, True

		return majority_label, False

	def release(self):
		try:
			if self.capture and self.capture.isOpened():
				self.capture.release()
		finally:
			cv.destroyAllWindows()
