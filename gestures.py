# Adapted from the supplied Apache-2.0 project; see CREDITS.md.
# Modified 2026-10-22: inference-only cleanup, modular helpers and resource handling.
"""Landmark geometry and gesture rules; independent of camera capture."""
import copy
import itertools
import numpy as np
import cv2 as cv

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value if max_value else 0.0

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def get_hand_position(landmark_list, image_shape):
    image_height, image_width = image_shape[:2]
    palm_x = sum(landmark_list[index][0] for index in (0, 5, 9, 13, 17)) / 5
    palm_y = sum(landmark_list[index][1] for index in (0, 5, 9, 13, 17)) / 5

    horizontal = "left" if palm_x < image_width / 3 else (
        "right" if palm_x > image_width * 2 / 3 else "center")
    vertical = "top" if palm_y < image_height / 3 else (
        "bottom" if palm_y > image_height * 2 / 3 else "middle")

    return f"{vertical}-{horizontal}"


def get_palm_center(landmark_list):
    palm_landmarks = (0, 5, 9, 13, 17)
    return [
        int(sum(landmark_list[index][0] for index in palm_landmarks) / 5),
        int(sum(landmark_list[index][1] for index in palm_landmarks) / 5),
    ]


def get_hand_angle(landmark_list):
    wrist_x, wrist_y = landmark_list[0]
    middle_x, middle_y = landmark_list[9]
    return float(np.degrees(np.arctan2(middle_y - wrist_y,
                                       middle_x - wrist_x)))


def get_rotation_delta(rotation_history):
    angles = [angle for angle in rotation_history if angle is not None]
    if len(angles) < rotation_history.maxlen:
        return 0.0
    change = angles[-1] - angles[0]
    while change > 180:
        change -= 360
    while change < -180:
        change += 360
    return change


def is_open_hand(landmark_list):
    return all(is_finger_extended(landmark_list, tip, pip)
               for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)))


def is_smiling_gesture(hand_states, center_histories):
    left_open, _ = hand_states.get("Left", (False, 0.0))
    _, right_rotation = hand_states.get("Right", (False, 0.0))
    if not left_open or not 70 <= abs(right_rotation) <= 110:
        return False

    left_history = center_histories["Left"]
    points = [point for point in left_history if point != [0, 0]]
    if len(points) < left_history.maxlen:
        return False
    start_x, start_y = points[0]
    end_x, end_y = points[-1]
    return abs(end_x - start_x) < 30 and abs(end_y - start_y) < 30


def is_finger_extended(landmark_list, tip_index, pip_index):
    wrist = np.array(landmark_list[0], dtype=np.float32)
    tip = np.array(landmark_list[tip_index], dtype=np.float32)
    pip = np.array(landmark_list[pip_index], dtype=np.float32)
    return np.linalg.norm(tip - wrist) > np.linalg.norm(pip - wrist) * 1.1


def get_custom_gesture(landmark_list):
    thumb_index_distance = np.linalg.norm(
        np.array(landmark_list[4], dtype=np.float32) -
        np.array(landmark_list[8], dtype=np.float32))
    palm_size = np.linalg.norm(
        np.array(landmark_list[0], dtype=np.float32) -
        np.array(landmark_list[9], dtype=np.float32))
    index_extended = is_finger_extended(landmark_list, 8, 6)
    middle_extended = is_finger_extended(landmark_list, 12, 10)
    ring_extended = is_finger_extended(landmark_list, 16, 14)
    little_extended = is_finger_extended(landmark_list, 20, 18)

    if (thumb_index_distance < palm_size * 0.45 and middle_extended and
            ring_extended and little_extended):
        return "OK"
    if (index_extended and middle_extended and not ring_extended and
            not little_extended):
        return "PEACE"
    if (index_extended and not middle_extended and not ring_extended and
            not little_extended and
            landmark_list[8][1] > landmark_list[5][1]):
        return "POINT DOWN"
    if (index_extended and little_extended and not middle_extended and
            not ring_extended):
        return "HORN"
    return ""


def get_movement_direction(point_history, image_shape):
    valid_points = [point for point in point_history
                    if point[0] != 0 or point[1] != 0]
    if len(valid_points) < 2:
        return "stationary"

    image_height, image_width = image_shape[:2]
    start_x, start_y = valid_points[0]
    end_x, end_y = valid_points[-1]
    delta_x = (end_x - start_x) / image_width
    delta_y = (end_y - start_y) / image_height
    threshold = 0.04

    if abs(delta_x) < threshold and abs(delta_y) < threshold:
        return "stationary"
    if abs(delta_x) > abs(delta_y):
        return "right" if delta_x > 0 else "left"
    return "down" if delta_y > 0 else "up"

