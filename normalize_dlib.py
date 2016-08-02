import numpy as np
import cv2

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

class Detect:
	NORMAL, HIGH_SENS, SINGLE_EYE, NO_FACE_FAIL, NO_EYES_FAIL = range(5)

FINAL_WIDTH = 90
FINAL_HEIGHT = 120
LR_OFFSET = 0.32
U_OFFSET = 0.4


def get_dlib_points(img):
	faces = detector(img)

	max_area = 0
	for (x, y, w, h) in faces:
		if (w * h) > max_area:
			max_area = w * h
			primary_coords = (x, y, w, h)
	if (not primary_coords):
		raise RuntimeError(Detect.NO_FACE_FAIL)

	return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def procrustes(points1, points2):
	points1 = points1.astype(np.float64)
	points2 = points2.astype(np.float64)

	mu1 = np.mean(points1, axis=0)
	mu2 = np.mean(points2, axis=0)
	points1 -= mu1
	points2 -= mu2

	s1 = np.std(
