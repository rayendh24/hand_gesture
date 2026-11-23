# Adapted from the supplied Apache-2.0 project; see CREDITS.md.
# Modified 2026-10-22: inference-only cleanup, modular helpers and resource handling.
# Modified 2026-11-23: new on-screen HUD (ui.py); recognition logic unchanged.
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
from pathlib import Path
from collections import Counter
from collections import deque

import cv2 as cv
import mediapipe as mp

from gestures import (
    calc_bounding_rect,
    calc_landmark_list,
    pre_process_landmark,
    pre_process_point_history,
    get_hand_position,
    get_palm_center,
    get_hand_angle,
    get_rotation_delta,
    is_open_hand,
    is_smiling_gesture,
    is_finger_extended,
    get_custom_gesture,
    get_movement_direction
)
from utils import CvFpsCalc
from ui import Hud

BASE_DIR = Path(__file__).resolve().parent
WINDOW_TITLE = 'Hand Gesture Recognition'
from model import KeyPointClassifier
from model import PointHistoryClassifier


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument("--max_num_hands",
                        help='maximum number of hands to detect',
                        type=int,
                        default=2)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=float,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    # Argument parsing
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    max_num_hands = args.max_num_hands

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    # Camera preparation
    cap = cv.VideoCapture(cap_device)
    hands = None
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {cap_device}. Try --device 1.")
        cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

        # Model load
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=use_static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        keypoint_classifier = KeyPointClassifier()

        point_history_classifier = PointHistoryClassifier()

        # Read labels
        with open(BASE_DIR / 'model/keypoint_classifier/keypoint_classifier_label.csv',
                  encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            keypoint_classifier_labels = [
                row[0] for row in keypoint_classifier_labels
            ]
        with open(
                BASE_DIR / 'model/point_history_classifier/point_history_classifier_label.csv',
                encoding='utf-8-sig') as f:
            point_history_classifier_labels = csv.reader(f)
            point_history_classifier_labels = [
                row[0] for row in point_history_classifier_labels
            ]

        # FPS Measurement
        cvFpsCalc = CvFpsCalc(buffer_len=10)

        # On-screen display
        hud = Hud()

        # Coordinate history
        history_length = 16
        point_histories = {
            "Left": deque(maxlen=history_length),
            "Right": deque(maxlen=history_length),
        }
        center_histories = {
            "Left": deque(maxlen=history_length),
            "Right": deque(maxlen=history_length),
        }
        rotation_histories = {
            "Left": deque(maxlen=history_length),
            "Right": deque(maxlen=history_length),
        }
        smiling_reported = False

        # Finger gesture history
        finger_gesture_histories = {
            "Left": deque(maxlen=history_length),
            "Right": deque(maxlen=history_length),
        }

        while True:
            fps = cvFpsCalc.get()

            # Process Key (ESC: end)
            key = cv.waitKey(10)
            if key == 27:  # ESC
                break
            if key in (ord('h'), ord('H')):  # Show/hide panels
                hud.toggle_panels()

            # Camera capture
            ret, image = cap.read()
            if not ret:
                break
            image = cv.flip(image, 1)  # Mirror display
            debug_image = copy.deepcopy(image)
            hud.begin_frame()

            # Detection implementation
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            hand_states = {}
            if results.multi_hand_landmarks is not None:
                detected_hands = set()
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                      results.multi_handedness):
                    hand_label = handedness.classification[0].label
                    detected_hands.add(hand_label)
                    point_history = point_histories.setdefault(
                        hand_label, deque(maxlen=history_length))
                    center_history = center_histories.setdefault(
                        hand_label, deque(maxlen=history_length))
                    rotation_history = rotation_histories.setdefault(
                        hand_label, deque(maxlen=history_length))
                    finger_gesture_history = finger_gesture_histories.setdefault(
                        hand_label, deque(maxlen=history_length))

                    # Bounding box calculation
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    # Landmark calculation
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    center_history.append(get_palm_center(landmark_list))
                    rotation_history.append(get_hand_angle(landmark_list))
                    hand_states[hand_label] = (
                        is_open_hand(landmark_list),
                        get_rotation_delta(rotation_history),
                    )

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(
                        landmark_list)
                    pre_processed_point_history_list = pre_process_point_history(
                        debug_image, point_history)

                    # Hand sign classification
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    if hand_sign_id == 2:  # Point gesture
                        point_history.append(landmark_list[8])
                    else:
                        point_history.append([0, 0])

                    # Finger gesture classification
                    finger_gesture_id = 0
                    point_history_len = len(pre_processed_point_history_list)
                    if point_history_len == (history_length * 2):
                        finger_gesture_id = point_history_classifier(
                            pre_processed_point_history_list)

                    # Calculates the gesture IDs in the latest detection
                    finger_gesture_history.append(finger_gesture_id)
                    most_common_fg_id = Counter(
                        finger_gesture_history).most_common()
                    movement_direction = get_movement_direction(point_history,
                                                                debug_image.shape)
                    hand_position = get_hand_position(landmark_list,
                                                      debug_image.shape)
                    custom_gesture = get_custom_gesture(landmark_list)

                    # Drawing part (collected here, drawn once per frame)
                    hud.add_hand(
                        hand_label,
                        brect,
                        landmark_list,
                        keypoint_classifier_labels[hand_sign_id],
                        point_history_classifier_labels[most_common_fg_id[0][0]],
                        movement_direction,
                        hand_position,
                        custom_gesture,
                    )
                for hand_label, point_history in point_histories.items():
                    if hand_label not in detected_hands:
                        point_history.append([0, 0])
                        center_histories[hand_label].append([0, 0])
                        rotation_histories[hand_label].append(None)
            else:
                for point_history in point_histories.values():
                    point_history.append([0, 0])
                for center_history in center_histories.values():
                    center_history.append([0, 0])
                for rotation_history in rotation_histories.values():
                    rotation_history.append(None)

            smiling = is_smiling_gesture(hand_states, center_histories)
            if smiling and not smiling_reported:
                print("SMILING command detected", flush=True)
            smiling_reported = smiling
            for hand_label, point_history in point_histories.items():
                hud.add_trail(hand_label, point_history)
            debug_image = hud.render(debug_image, fps, smiling)

            # Screen reflection
            cv.imshow(WINDOW_TITLE, debug_image)

    finally:
        if hands is not None:
            hands.close()
        cap.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    main()