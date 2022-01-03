import mediapipe as mp
import cv2
from statistics import mean
import time
from numpy.lib.twodim_base import eye
import custom_drawing_utils
import numpy as np


def calibration(type_used: str, cap: cv2.VideoCapture, mp_face_mesh: mp.solutions.face_mesh, mp_drawing: custom_drawing_utils) -> tuple:
    """This function performs calibration
    Args:
        type_used: whether to perform eye or nose tracking
        cap: videoCapturing object
        mp_face_mesh: object for using mediapipe pipeling
        mp_drawing: object for plotting
    Returns:
        tuple: width and height of the reactangle where user's eyes moved during calibration,
        center point, distance from the user to the camera"""

    points = []
    distances = []

    #Center, left, right, up, down
    factors = [(1/2, 1/2), (0, 1/2), (1, 1/2), (1/2, 0), (1/2, 1)]

    for factor in factors:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            annotated_image = image.copy()
            annotated_image = cv2.flip(annotated_image, 1)
            image.flags.writeable = False
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 500)
            fontScale = 1
            fontColor = (255, 255, 255)
            thickness = 1
            lineType = 2

            cv2.putText(annotated_image, 'Please, look at the red circle and press Space',
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)

            center_coordinates = (
                int(image.shape[1] * factor[0]), int(image.shape[0] * factor[1]))

            radius = 10

            color = (0, 0, 255)

            thickness = -1
            cv2.circle(annotated_image, center_coordinates,
                       radius, color, thickness)

            cv2.imshow('MediaPipe Face Mesh', annotated_image)
            pressedKey = cv2.waitKey(1) & 0xFF
            # If space is pressed remember some numerical values
            if pressedKey == 32:
                with mp_face_mesh.FaceMesh(
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.3,
                        min_tracking_confidence=0.5) as face_mesh:
                    results = face_mesh.process(image)
                    if results.multi_face_landmarks:
                        # if we perform nose tracking only current nose coordinates are needed, all other values are hardcoded
                        # in order to make the calibration process quicker
                        if type_used == 'nose':
                            return ((100, 50), (mp_drawing.find_nose_coord(results.multi_face_landmarks[0], image)), 45)

                        points.append(mp_drawing.find_iris_center(
                            results.multi_face_landmarks[0], image))
                        distances.append(mp_drawing.find_distance(
                            results.multi_face_landmarks[0], image))
                        break
            # Close the application is ESC is pressed
            elif pressedKey == 27:
                cap.release()
                raise SystemExit

    eye_image_height = points[4][1] - points[3][1]
    eye_image_width = points[2][0] - points[1][0]

    return ((eye_image_width, eye_image_height), (int(points[0][0]), int(points[0][1])), mean(distances))


def recalibration(cap, mp_face_mesh, mp_drawing):
    """This function performs calibration
    Args:
        cap: videoCapturing object
        mp_face_mesh: object for using mediapipe pipeling
        mp_drawing: object for plotting
    Returns:
        type_used: tracking mode
        tuple: width and height of the reactangle where user's eyes moved during calibration,
        center point, distance from the user to the camera"""
    cv2.destroyAllWindows()
    type_used = mode_selection()
    image_dimensions, center_for_calculations, distance = calibration(type_used,
                                                                      cap, mp_face_mesh, mp_drawing)
    return type_used, image_dimensions, center_for_calculations, distance

# Selecting whether to track nose or gaze


def mode_selection() -> str:
    """This function performs tracking mode selection
    Returns:
        desired tracking mode"""
    while True:
        image = np.zeros((500, 1000))
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 30)
        fontScale = 1
        fontColor = (255, 255, 255)
        thickness = 1
        lineType = 2

        cv2.putText(image, 'Please, press a for eye tracking and b for nose tracking',
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
        cv2.imshow('Choose your variant', image)
        pressedKey = cv2.waitKey(1) & 0xFF
        if pressedKey == ord('a'):
            cv2.destroyAllWindows()
            return 'eye'
        elif pressedKey == ord('b'):
            cv2.destroyAllWindows()
            return 'nose'
        elif pressedKey == 27:
            raise SystemExit


mp_drawing = custom_drawing_utils
mp_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)
previous_result = (640, 360)
previous_var = 1
previous_time = 0
# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

type_used, image_dimensions, center_for_calculations, distance = recalibration(
    cap, mp_face_mesh, mp_drawing)

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # time when we finish processing for this frame
        new_frame_time = time.time()

        # Calculating the fps

        fps = 1 / (new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time

        fps = str(int(fps))

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        cv2.putText(image, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    3, (100, 255, 0), 3, cv2.LINE_AA)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        try:
            if results.multi_face_landmarks:
                image, previous_result, previous_var, previous_time = mp_drawing.tracking(type_used, previous_result, previous_var,
                                                                                            previous_time, distance,
                                                                                            center_for_calculations,
                                                                                            image_dimensions,
                                                                                            image=image,
                                                                                            landmark_list=results.
                                                                                            multi_face_landmarks[0])
        except:
            type_used, image_dimensions, center_for_calculations, distance = recalibration(
                cap, mp_face_mesh, mp_drawing)
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Mesh', image)
        pressedKey = cv2.waitKey(1) & 0xFF
        if pressedKey == 32:
            type_used, image_dimensions, center_for_calculations, distance = recalibration(
                cap, mp_face_mesh, mp_drawing)
        elif pressedKey == 27:
            break

cap.release()
