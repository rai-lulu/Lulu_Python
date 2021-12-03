
import mediapipe as mp
import cv2
import custom_drawing_utils


def calibration(cap: cv2.VideoCapture, mp_face_mesh: mp.solutions.face_mesh, mp_drawing: custom_drawing_utils) -> list:
    """This function performs calibration
    Args:
        cap: videoCapturing object
        mp_face_mesh: object for using mediapipe pipeling
        mp_drawing: object for plotting
    Returns:
        list: width and height of the reactangle where user's eyes moved during calibration,
         center point, distance from the user to the camera"""
    ls = []

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
                    ls.append(mp_drawing.find_ls(
                        results.multi_face_landmarks[0].landmark, image))
                    break
            elif pressedKey == 27:
                cap.release()
                raise SystemExit

    #from l1 when user looks down substract l1 when he looks up, from l3 when user looks up substract l3 when he looks down,
    #Take their average
    eye_image_height = (ls[3][0] - ls[1][0] + ls[1][2] - ls[3][2]) / 2

    #from l2 when user looks left substract l2 whem he looks right, from l4 when the user looks right substract l4 when he looks
    #left, take their average 
    eye_image_width = (ls[2][3] - ls[4][3] + ls[4][1] - ls[2][1]) / 2

    distance_from_top = ls[1][0]
    distance_from_left = ls[2][1]

    return (eye_image_width, eye_image_height), distance_from_top, distance_from_left


mp_drawing = custom_drawing_utils
mp_face_mesh = mp.solutions.face_mesh


cap = cv2.VideoCapture(0)

w = int(cap.get(3))
h = int(cap.get(4))

writer = cv2.VideoWriter('./video_test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (w, h))

eye_image_dimensions, distance_from_top, distance_from_left = calibration(
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
                mp_drawing.draw_iris_landmarks(
                    image=image,
                    landmark_list=face_landmarks)
        # Flip the image horizontally for a selfie-view display.
        writer.write(image)  
        cv2.imshow('MediaPipe Face Mesh', image)

        pressedKey = cv2.waitKey(1) & 0xFF
        if pressedKey == 32:
            center_ls, left_ls, right_ls, up_ls, down_ls = calibration(cap, mp_face_mesh, mp_drawing)
        elif pressedKey == 27:
            break

cap.release()
writer.release()
