
import mediapipe as mp
import cv2
from statistics import mean

from numpy.lib.twodim_base import eye
import custom_drawing_utils


def calibration(cap: cv2.VideoCapture, mp_face_mesh: mp.solutions.face_mesh, mp_drawing: custom_drawing_utils) -> tuple:
    """This function performs calibration
    Args:
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
            image.flags.writeable = False
            image = cv2.flip(image, 1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 500)
            fontScale = 1
            fontColor = (255, 255, 255)
            thickness = 1
            lineType = 2

            cv2.putText(image, 'Please, look at the red circle and press Space',
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
            cv2.circle(image, center_coordinates, radius, color, thickness)

            cv2.imshow('MediaPipe Face Mesh', image)
            pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey == 32:
                with mp_face_mesh.FaceMesh(
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as face_mesh:
                    results = face_mesh.process(annotated_image)
                    landmarks = results.multi_face_landmarks[0].landmark
                    points.append(((landmarks[473].x + landmarks[468].x) / 2 * image.shape[1],
                                   (landmarks[473].y + landmarks[468].y) / 2 * image.shape[0]))
                    distances.append(mp_drawing.find_distance(
                        results.multi_face_landmarks[0], image))
                    break
            elif pressedKey == 27:
                cap.release()
                raise SystemExit
    print(points)
    eye_image_height = points[4][1] - points[3][1]
    eye_image_width = points[1][0] - points[2][0]
    print((eye_image_width, eye_image_height))
    return ((eye_image_width, eye_image_height), points[0], mean(distances))


mp_drawing = custom_drawing_utils
mp_face_mesh = mp.solutions.face_mesh


cap = cv2.VideoCapture(0)

eye_image_dimensions, eye_center, distance = calibration(
    cap, mp_face_mesh, mp_drawing)

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_iris_landmarks_length(distance, eye_center, eye_image_dimensions,
                                                      image=image,
                                                      landmark_list=face_landmarks)
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Mesh', image)

        pressedKey = cv2.waitKey(1) & 0xFF
        if pressedKey == 32:
            eye_image_dimensions, eye_center, distance = calibration(
                cap, mp_face_mesh, mp_drawing)
        elif pressedKey == 27:
            break

cap.release()
