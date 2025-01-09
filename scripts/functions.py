import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from rtree import index
from shapely.geometry import Polygon, LineString
from scipy.interpolate import make_interp_spline, splprep, splev
import pickle
from typing import Tuple, List, Literal, Callable
import json
from ultralytics import YOLO

MatLike = np.ndarray


def gaussian_blur(kernel, sigma) -> Callable[[MatLike], MatLike]:
    def func(image):
        return cv2.GaussianBlur(image, kernel, sigma)
    return func

def morph_dilate(kernel, iterations = 1) -> Callable[[MatLike], MatLike]:
    def func(image):
        return cv2.dilate(image, kernel, iterations)
    return func

def morph_erode(kernel, iterations = 1) -> Callable[[MatLike], MatLike]:
    def func(image):
        return cv2.erode(image, kernel, iterations)
    return func

# def threshold(thresh, type) -> Callable[[MatLike], MatLike]:
#     def func(image):
#         _, threshed = cv2.threshold(image, thresh, 255, type)
#         return threshed
#     return func

def adaptive_threshold(block_size, C) -> Callable[[MatLike], MatLike]:
    def func(image):
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, C)
    return func

def threshold(min, max, type='binary') -> Callable[[MatLike], MatLike]:
    if type == 'binary':
        cv_type = cv2.THRESH_BINARY
    elif type == 'binary_inv':
        cv_type = cv2.THRESH_BINARY_INV
    elif type == 'trunc':
        cv_type = cv2.THRESH_TRUNC
    elif type == 'tozero':
        cv_type = cv2.THRESH_TOZERO
    elif type == 'tozero_inv':
        cv_type = cv2.THRESH_TOZERO_INV
    else:
        raise ValueError('Invalid threshold type. Choose either "binary", "binary_inv", "trunc", "tozero", or "tozero_inv"')
    def func(image):
        return cv2.threshold(image, min, max, cv_type)[1]
    return func

def in_range(min, max) -> Callable[[MatLike], MatLike]:
    def func(image):
        return cv2.inRange(image, min, max)
    return func

def morph_open(kernel) -> Callable[[MatLike], MatLike]:
    def func(image):
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return func

def morph_close(kernel) -> Callable[[MatLike], MatLike]:
    def func(image):
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return func

def apply_filters(image, filters) -> MatLike:
    result = image
    for f in filters:
        result = f(result)
    return result

def kernel(size) -> np.ndarray:
    # TODO: This is going to get deprecated
    # return np.ones((size, size), np.uint8)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def invert(image) -> MatLike:
    return cv2.bitwise_not(image)

def does_line_intersect_contour(line, contour_polygons):
    """Function to check if a line segment intersects any contour"""
    line_obj = LineString(line)
    for poly in contour_polygons:
        if line_obj.intersects(poly.exterior):
            return True
    return False

def line_is_inside_contour(line, contour_polygons):
    """Function to check if a line segment is inside any contour"""
    line_obj = LineString(line)
    return any(line_obj.within(p) for p in contour_polygons)

def line_is_outside_contour(line, contour_polygons):
    """Function to check if a line segment is outside any contour"""
    line_obj = LineString(line)
    return any(line_obj.touches(p) for p in contour_polygons)


def highlight_range(image, min, max):
    # gray to hsv
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    in_range = cv2.inRange(image, min, max)
    
    # hue = 0 (red)
    hsv[..., 0] = 0

    # perform bitwise or
    hsv[..., 1] = cv2.bitwise_or(hsv[..., 1], in_range)
    hsv[..., 2] = cv2.bitwise_or(hsv[..., 2], in_range)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    return rgb

def highlight_mask(image, mask):
    # gray to hsv
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    
    # hue = 0 (red)
    hsv[..., 0] = 0

    # perform bitwise or
    hsv[..., 1] = cv2.bitwise_or(hsv[..., 1], mask)
    hsv[..., 2] = cv2.bitwise_or(hsv[..., 2], mask)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    return rgb

def clean_image(image):
    # keep colors with low saturation
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.split(image_hsv)[1]
    # apply threshold
    filters = [threshold(150, 255),
            morph_dilate(kernel(3)),
            ]

    mask = apply_filters(mask, filters)
    # original image to gray
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # apply mask
    image = cv2.bitwise_or(image, mask)

    filters = [in_range(0, 242),
            invert,
            ]
    mask = apply_filters(image, filters)

    image = cv2.bitwise_or(image, mask)

    return image

def upscale_image(image, scale=4):
    return cv2.resize(image, (0, 0), fx=scale, fy=scale)

def sharpen_image(image, kernel=3):
    if kernel == 3:
        kernel = np.array([[-1,-1,-1],
                           [-1,9,-1],
                           [-1,-1,-1]])
    elif kernel == 5:
        kernel = np.array([[0, 0, -1, 0, 0],
                        [0, -1, -2, -1, 0],
                        [-1, -2, 17, -2, -1],
                        [0, -1, -2, -1, 0],
                        [0, 0, -1, 0, 0]])
    else:
        raise ValueError('Invalid kernel size. Choose either 3 or 5')

    return cv2.filter2D(image, -1, kernel)

def model_predict(image_path):
    # Load the model
    model = YOLO("models/full_set_menu-yolo11m_plus3.pt")
    results = model(image_path)[0]

    names =results.names

    predictions = []
    for result in results:
        boxes = result.boxes.xyxy.tolist()[0]
        cls = result.boxes.cls.tolist()[0]
        conf = result.boxes.conf.tolist()[0]*100

        predictions.append({'class': names[cls],
                            'confidence': conf,
                            'x': int((boxes[0]+boxes[2])/2),
                            'y': int((boxes[1]+boxes[3])/2),
                            'width': int(boxes[2]-boxes[0]),
                            'height': int(boxes[3]-boxes[1])})

    return predictions
