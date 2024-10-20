import numpy as np


def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = map(int, bbox)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y


def measure_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_foot_position(bbox) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)


def get_closest_keypoint_index(point, keypoints, keypoint_indices):
    closest_distance = float("inf")
    key_point_ind = keypoint_indices[0]
    for keypoint_index in keypoint_indices:
        keypoint = keypoints[keypoint_index * 2], keypoints[keypoint_index * 2 + 1]
        distance = abs(point[1] - keypoint[1])

        if distance < closest_distance:
            closest_distance = distance
            key_point_ind = keypoint_index

    return key_point_ind


def get_height_of_bbox(bbox):
    return bbox[3] - bbox[1]


def measure_xy_distance(p1, p2) -> tuple:
    return p1[0] - p2[0], p1[1] - p2[1]


def get_center_of_bbox(bbox) -> tuple[int, int]:
    return int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)