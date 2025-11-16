import cv2 as cv
import numpy as np
import sys
from collections import deque, Counter

# 0 for webcam feed ; add "path to file" for video file
capture = cv.VideoCapture(0)

# load cascades from opencv package data directory
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

if face_cascade.empty() or eye_cascade.empty():
    print("Error: Haar cascade xml files not found. Ensure opencv-python is installed.")
    sys.exit(1)


tilt_history = deque(maxlen=10)
last_majority = None

while True:
    ret, frame = capture.read()
    if not ret or frame is None:
        print("Warning: failed to read frame from capture")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        cv.imshow('Frame', frame)
        if cv.waitKey(1) & 0xFF == 27:
            break
        continue

    # process first detected face (change to loop if you want all)
    x, y, w, h = faces[0]
    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.circle(frame, (x + w // 2, y + h // 2), 4, (0, 255, 0), -1)

    roi_gray = gray[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)

    eye_boxes = []
    for (ex, ey, ew, eh) in eyes:
        # convert eye coords to full-frame coordinates
        ex_abs, ey_abs = x + ex, y + ey
        eye_boxes.append((ex_abs, ey_abs, ew, eh))
        cv.rectangle(frame, (ex_abs, ey_abs), (ex_abs + ew, ey_abs + eh), (0, 0, 255), 2)

    if len(eye_boxes) >= 2:
        # pick two eyes by x coordinate (left, right)
        eye_boxes.sort(key=lambda b: b[0])
        left_eye = eye_boxes[0]
        right_eye = eye_boxes[1]

        left_eye_center = (int(left_eye[0] + left_eye[2] / 2), int(left_eye[1] + left_eye[3] / 2))
        right_eye_center = (int(right_eye[0] + right_eye[2] / 2), int(right_eye[1] + right_eye[3] / 2))

        lx, ly = left_eye_center
        rx, ry = right_eye_center

        delta_x = rx - lx
        delta_y = ry - ly

        # use arctan2 to handle delta_x == 0
        angle_rad = np.arctan2(delta_y, delta_x)
        angle_deg = np.degrees(angle_rad)

        # per-frame label
        if angle_deg > 10:
            tilt_label = "right"
            cv.putText(frame, f'RIGHT TILT : {int(angle_deg)} deg', (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_4)
        elif angle_deg < -10:
            tilt_label = "left"
            cv.putText(frame, f'LEFT TILT : {int(angle_deg)} deg', (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_4)
        else:
            tilt_label = "straight"
            cv.putText(frame, f'STRAIGHT : {int(angle_deg)} deg', (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_4)
    else:
        tilt_label = "none"

    # update sliding window and compute majority once we have samples
    tilt_history.append(tilt_label)

    if len(tilt_history) < tilt_history.maxlen:
        cv.putText(frame, f'Collecting: {len(tilt_history)}/{tilt_history.maxlen}', (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2, cv.LINE_AA)
    else:
        counts = Counter(tilt_history)
        # prefer left/right/straight over 'none' by using the most common non-'none' if tied
        most_common_label, most_common_count = counts.most_common(1)[0]
        # require a strict majority to declare a tilt
        if most_common_label != "none" and most_common_count > (tilt_history.maxlen // 2):
            majority = most_common_label
            cv.putText(frame, f'MAJORITY: {majority.upper()} ({most_common_count}/{tilt_history.maxlen})', (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 0), 2, cv.LINE_AA)
        else:
            majority = "uncertain"
            cv.putText(frame, f'MAJORITY: UNCERTAIN ({most_common_count}/{tilt_history.maxlen})', (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2, cv.LINE_AA)

        # print only when majority changes (useful later for triggering keyboard)
        if majority != last_majority:
            print("Majority tilt ->", majority)
            last_majority = majority

    cv.imshow('Frame', frame)

    if cv.waitKey(1) & 0xFF == 27:
        break

capture.release()
cv.destroyAllWindows()