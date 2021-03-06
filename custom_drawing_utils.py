import math
from typing import List, Mapping, Optional, Tuple, Union

import cv2
import dataclasses
import matplotlib.pyplot as plt
import numpy as np
import time

from mediapipe.framework.formats import detection_pb2
from mediapipe.framework.formats import location_data_pb2
from mediapipe.framework.formats import landmark_pb2
from numpy.lib.twodim_base import eye

_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_RGB_CHANNELS = 3

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 0, 0)


@dataclasses.dataclass
class DrawingSpec:
    # Color for drawing the annotation. Default to the white color.
    color: Tuple[int, int, int] = WHITE_COLOR
    # Thickness for drawing the annotation. Default to 2 pixels.
    thickness: int = 2
    # Circle radius. Default to 2 pixels.
    circle_radius: int = 2


def _normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def draw_detection(
        image: np.ndarray,
        detection: detection_pb2.Detection,
        keypoint_drawing_spec: DrawingSpec = DrawingSpec(color=RED_COLOR),
        bbox_drawing_spec: DrawingSpec = DrawingSpec()):
    """Draws the detction bounding box and keypoints on the image.

    Args:
      image: A three channel RGB image represented as numpy ndarray.
      detection: A detection proto message to be annotated on the image.
      keypoint_drawing_spec: A DrawingSpec object that specifies the keypoints'
        drawing settings such as color, line thickness, and circle radius.
      bbox_drawing_spec: A DrawingSpec object that specifies the bounding box's
        drawing settings such as color and line thickness.

    Raises:
      ValueError: If one of the followings:
        a) If the input image is not three channel RGB.
        b) If the location data is not relative data.
    """
    if not detection.location_data:
        return
    if image.shape[2] != _RGB_CHANNELS:
        raise ValueError('Input image must contain three channel rgb data.')
    image_rows, image_cols, _ = image.shape

    location = detection.location_data
    if location.format != location_data_pb2.LocationData.RELATIVE_BOUNDING_BOX:
        raise ValueError(
            'LocationData must be relative for this drawing funtion to work.')
    # Draws keypoints.
    for keypoint in location.relative_keypoints:
        keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                       image_cols, image_rows)
        cv2.circle(image, keypoint_px, keypoint_drawing_spec.circle_radius,
                   keypoint_drawing_spec.color, keypoint_drawing_spec.thickness)
    # Draws bounding box if exists.
    if not location.HasField('relative_bounding_box'):
        return
    relative_bounding_box = location.relative_bounding_box
    rect_start_point = _normalized_to_pixel_coordinates(
        relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
        image_rows)
    rect_end_point = _normalized_to_pixel_coordinates(
        relative_bounding_box.xmin + relative_bounding_box.width,
        relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
        image_rows)
    cv2.rectangle(image, rect_start_point, rect_end_point,
                  bbox_drawing_spec.color, bbox_drawing_spec.thickness)


def draw_landmarks(
    image: np.ndarray,
    landmark_list: landmark_pb2.NormalizedLandmarkList,
    connections: Optional[List[Tuple[int, int]]] = None,
    landmark_drawing_spec: Union[DrawingSpec,
                                 Mapping[int, DrawingSpec]] = DrawingSpec(
                                     color=RED_COLOR),
    connection_drawing_spec: Union[DrawingSpec,
                                   Mapping[Tuple[int, int],
                                           DrawingSpec]] = DrawingSpec()):
    """Draws the landmarks and the connections on the image.

    Args:
      image: A three channel RGB image represented as numpy ndarray.
      landmark_list: A normalized landmark list proto message to be annotated on
        the image.
      connections: A list of landmark index tuples that specifies how landmarks to
        be connected in the drawing.
      landmark_drawing_spec: Either a DrawingSpec object or a mapping from
        hand landmarks to the DrawingSpecs that specifies the landmarks' drawing
        settings such as color, line thickness, and circle radius.
        If this argument is explicitly set to None, no landmarks will be drawn.
      connection_drawing_spec: Either a DrawingSpec object or a mapping from
        hand connections to the DrawingSpecs that specifies the
        connections' drawing settings such as color and line thickness.
        If this argument is explicitly set to None, no landmark connections will
        be drawn.

    Raises:
      ValueError: If one of the followings:
        a) If the input image is not three channel RGB.
        b) If any connetions contain invalid landmark index.
    """
    if not landmark_list:
        return
    if image.shape[2] != _RGB_CHANNELS:
        raise ValueError('Input image must contain three channel rgb data.')
    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if ((landmark.HasField('visibility') and
             landmark.visibility < _VISIBILITY_THRESHOLD) or
            (landmark.HasField('presence') and
             landmark.presence < _PRESENCE_THRESHOLD)):
            continue
        landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                       image_cols, image_rows)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
    if connections:
        num_landmarks = len(landmark_list.landmark)
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(f'Landmark index is out of range. Invalid connection '
                                 f'from landmark #{start_idx} to landmark #{end_idx}.')
            if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                drawing_spec = connection_drawing_spec[connection] if isinstance(
                    connection_drawing_spec, Mapping) else connection_drawing_spec
                cv2.line(image, idx_to_coordinates[start_idx],
                         idx_to_coordinates[end_idx], drawing_spec.color,
                         drawing_spec.thickness)
    # Draws landmark points after finishing the connection lines, which is
    # aesthetically better.
    if landmark_drawing_spec:
        for idx, landmark_px in idx_to_coordinates.items():
            drawing_spec = landmark_drawing_spec[idx] if isinstance(
                landmark_drawing_spec, Mapping) else landmark_drawing_spec
            # White circle border
            circle_border_radius = max(drawing_spec.circle_radius + 1,
                                       int(drawing_spec.circle_radius * 1.2))
            cv2.circle(image, landmark_px, circle_border_radius, WHITE_COLOR,
                       drawing_spec.thickness)
            # Fill color into the circle
            cv2.circle(image, landmark_px, drawing_spec.circle_radius,
                       drawing_spec.color, drawing_spec.thickness)


def draw_axis(
        image: np.ndarray,
        rotation: np.ndarray,
        translation: np.ndarray,
        focal_length: Tuple[float, float] = (1.0, 1.0),
        principal_point: Tuple[float, float] = (0.0, 0.0),
        axis_length: float = 0.1,
        axis_drawing_spec: DrawingSpec = DrawingSpec()):
    """Draws the 3D axis on the image.

    Args:
      image: A three channel RGB image represented as numpy ndarray.
      rotation: Rotation matrix from object to camera coordinate frame.
      translation: Translation vector from object to camera coordinate frame.
      focal_length: camera focal length along x and y directions.
      principal_point: camera principal point in x and y.
      axis_length: length of the axis in the drawing.
      axis_drawing_spec: A DrawingSpec object that specifies the xyz axis
        drawing settings such as line thickness.

    Raises:
      ValueError: If one of the followings:
        a) If the input image is not three channel RGB.
    """
    if image.shape[2] != _RGB_CHANNELS:
        raise ValueError('Input image must contain three channel rgb data.')
    image_rows, image_cols, _ = image.shape
    # Create axis points in camera coordinate frame.
    axis_world = np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    axis_cam = np.matmul(rotation, axis_length*axis_world.T).T + translation
    x = axis_cam[..., 0]
    y = axis_cam[..., 1]
    z = axis_cam[..., 2]
    # Project 3D points to NDC space.
    fx, fy = focal_length
    px, py = principal_point
    x_ndc = np.clip(-fx * x / (z + 1e-5) + px, -1., 1.)
    y_ndc = np.clip(-fy * y / (z + 1e-5) + py, -1., 1.)
    # Convert from NDC space to image space.
    x_im = np.int32((1 + x_ndc) * 0.5 * image_cols)
    y_im = np.int32((1 - y_ndc) * 0.5 * image_rows)
    # Draw xyz axis on the image.
    origin = (x_im[0], y_im[0])
    x_axis = (x_im[1], y_im[1])
    y_axis = (x_im[2], y_im[2])
    z_axis = (x_im[3], y_im[3])
    cv2.arrowedLine(image, origin, x_axis, RED_COLOR,
                    axis_drawing_spec.thickness)
    cv2.arrowedLine(image, origin, y_axis, GREEN_COLOR,
                    axis_drawing_spec.thickness)
    cv2.arrowedLine(image, origin, z_axis, BLUE_COLOR,
                    axis_drawing_spec.thickness)


def _normalize_color(color):
    return tuple(v / 255. for v in color)


def plot_landmarks(landmark_list: landmark_pb2.NormalizedLandmarkList,
                   connections: Optional[List[Tuple[int, int]]] = None,
                   landmark_drawing_spec: DrawingSpec = DrawingSpec(
                       color=RED_COLOR, thickness=5),
                   connection_drawing_spec: DrawingSpec = DrawingSpec(
                       color=BLACK_COLOR, thickness=5),
                   elevation: int = 10,
                   azimuth: int = 10):
    """Plot the landmarks and the connections in matplotlib 3d.

    Args:
      landmark_list: A normalized landmark list proto message to be plotted.
      connections: A list of landmark index tuples that specifies how landmarks to
        be connected.
      landmark_drawing_spec: A DrawingSpec object that specifies the landmarks'
        drawing settings such as color and line thickness.
      connection_drawing_spec: A DrawingSpec object that specifies the
        connections' drawing settings such as color and line thickness.
      elevation: The elevation from which to view the plot.
      azimuth: the azimuth angle to rotate the plot.
    Raises:
      ValueError: If any connetions contain invalid landmark index.
    """
    if not landmark_list:
        return
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.view_init(elev=elevation, azim=azimuth)
    plotted_landmarks = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if ((landmark.HasField('visibility') and
             landmark.visibility < _VISIBILITY_THRESHOLD) or
            (landmark.HasField('presence') and
             landmark.presence < _PRESENCE_THRESHOLD)):
            continue
        ax.scatter3D(
            xs=[-landmark.z],
            ys=[landmark.x],
            zs=[-landmark.y],
            color=_normalize_color(landmark_drawing_spec.color[::-1]),
            linewidth=landmark_drawing_spec.thickness)
        plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)
    if connections:
        num_landmarks = len(landmark_list.landmark)
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(f'Landmark index is out of range. Invalid connection '
                                 f'from landmark #{start_idx} to landmark #{end_idx}.')
            if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
                landmark_pair = [
                    plotted_landmarks[start_idx], plotted_landmarks[end_idx]
                ]
                ax.plot3D(
                    xs=[landmark_pair[0][0], landmark_pair[1][0]],
                    ys=[landmark_pair[0][1], landmark_pair[1][1]],
                    zs=[landmark_pair[0][2], landmark_pair[1][2]],
                    color=_normalize_color(
                        connection_drawing_spec.color[::-1]),
                    linewidth=connection_drawing_spec.thickness)
    plt.show()


def draw_iris_landmarks(
    image: np.ndarray,
    landmark_list: landmark_pb2.NormalizedLandmarkList,
    eye_key_indicies=[
        # Left eye
        # eye lower contour
        33,
        7,
        163,
        144,
        145,
        153,
        154,
        155,
        133,
        # eye upper contour (excluding corners)
        246,
        161,
        160,
        159,
        158,
        157,
        173,
        # Right eye
        # eye lower contour
        263,
        249,
        390,
        373,
        374,
        380,
        381,
        382,
        362,
        # eye upper contour (excluding corners)
        466,
        388,
        387,
        386,
        385,
        384,
        398

    ],
    iris_key_indicies=[

        # Left iris points
        468,  # middle
        469,  # right
        470,  # up
        471,  # left
        472,  # down

        # Right iris points
        473,  # middle
        474,  # left
        475,  # upper
        476,  # right
        477  # lower

    ],
    eye_drwing_color=(255, 0, 0),
    iris_drawing_color=(0, 0, 255)
):
    """This function draws iris landmarks
    Args:
        image: image to be processed
        landmark_list: A normalized landmark list proto message to be plotted.
        eye_key_indicies: indices of eyes' contours
        iris_key_indices: indices of irises' points
        eye_drawing_color: color of eyes' contours
        iris_drawing_color: irises' drawing color
    """

    if not landmark_list:
        return
    if image.shape[2] != _RGB_CHANNELS:
        raise ValueError('Input image must contain three channel rgb data.')
    image_rows, image_cols, _ = image.shape
    idx_to_eye_coordinates = {}
    idx_to_iris_coordinates = {}
    for idx, landmark in enumerate(landmark_list.landmark):

        landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                       image_cols, image_rows)
        if landmark_px and idx in eye_key_indicies:
            idx_to_eye_coordinates[idx] = landmark_px

        elif landmark_px and idx in iris_key_indicies:
            idx_to_iris_coordinates[idx] = landmark_px

    for landmark_px in idx_to_eye_coordinates.values():
        cv2.circle(image, landmark_px, 2,
                   eye_drwing_color, 1)

    for landmark_px in idx_to_iris_coordinates.values():
        cv2.circle(image, landmark_px, 2,
                   iris_drawing_color, 1)


def tracking(type_used: str, previous_result: tuple, previous_var: int, previous_time: float, distance: float, center: tuple,
             image_dimensions: tuple,
             image: np.ndarray,
             landmark_list: landmark_pb2.NormalizedLandmarkList,
             eye_key_indicies=[
        # Left eye
        # eye lower contour
                 33,
                 7,
                 163,
                 144,
                 145,
                 153,
                 154,
                 155,
                 133,
                 # eye upper contour (excluding corners)
                 246,
                 161,
                 160,
                 159,
                 158,
                 157,
                 173,

                 # Right eye
                 # eye lower contour
                 263,
                 249,
                 390,
                 373,
                 374,
                 380,
                 381,
                 382,
                 362,
                 # eye upper contour (excluding corners)
                 466,
                 388,
                 387,
                 386,
                 385,
                 384,
                 398

                 ],
             iris_key_indicies=[

        # Left iris points
                 468,  # middle
                 469,  # right
                 470,  # up
                 471,  # left
        472,  # down

        # Right iris points
                 473,  # middle
                 474,  # left
                 475,  # upper
                 476,  # right
                 477  # lower

                 ],
             eye_drwing_color=(255, 0, 0),
             iris_drawing_color=(0, 0, 255)
             ) -> tuple:
    """This function draws iris landmarks, point on the screen the user is looking at and prints calculated distance to the user
    Args:
        type_used: whether to perform eye or nose tracking
        previous_result: center_coordinates of the previous frame
        previous_var: variant selected on the previous frame 
        previous_time: time the user is looking at previous_var
        distance: initial distance from calibration
        center: position of user's "middle" eye when looking at the center of the screen from calibration
        image_dimensions: width and height of the reactangle where user's eyes moved during calibration
        image: image to be processed
        landmark_list: A normalized landmark list proto message to be plotted.
        eye_key_indicies: indices of eyes' contours
        iris_key_indices: indices of irises' points
        eye_drawing_color: color of eyes' contours
        iris_drawing_color: irises' drawing color
    Returns:
        np.ndarray: image processed
        tuple: center_coordinates to be used as previous_result for the next frame 
        int: curr_var to be used as previous_var for the next frame
        int: curr_time to be used as previous_time for the next frame 
    """

    curr_distance = find_distance(landmark_list, image)

    w = image.shape[1]
    h = image.shape[0]

    # Find width for each rectangle
    closest_w = int(4 * np.round(w / 4.))
    dw = w - closest_w
    w_steps = [w // 4] * 3 + [w // 4 + dw]

    # Find height for each rectangle
    closest_h = int(4 * np.round(h / 4.))
    dh = h - closest_h
    h_steps = [h // 4] * 3 + [h // 4 + dh]

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 500)
    fontScale = 1
    fontColor = (255, 255, 255)
    thickness = 1
    lineType = 2

    cv2.putText(image, 'Distance = ' + str(curr_distance),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)

    # Time used for testing variant selection
    cv2.putText(image, 'Time = ' + str(time.time() - previous_time),
                (500, 50),
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)

    if type_used == 'eye':
        # Current position of the center of user's "middle" eye
        center_for_calculations = find_iris_center(landmark_list, image)

    elif type_used == 'nose':
        # Current position of the center of user's nose
        center_for_calculations = find_nose_coord(landmark_list, image)

    else:
        print("Unknown tracking type")
        raise SystemExit

    radius = 5

    thickness = -1

    # Current center coordinates of the point on the screen the user is looking at
    center_coordinates = find_center_coordinates(image, previous_result,
                                                 distance, curr_distance, center, image_dimensions, center_for_calculations)

    # Plot current eye_center on the image
    cv2.circle(image, (center_for_calculations[0],
                       center_for_calculations[1]), radius, BLACK_COLOR, thickness)

    # Plot eye_center from calibration on the image
    cv2.circle(image, center, radius, WHITE_COLOR, thickness)

    radius = 50

    # Plot a circle on the screen that will indicate the area the user is looking at
    cv2.circle(image, center_coordinates, radius, RED_COLOR, thickness)

    # Parameters used when deciding whether the user has chosen a variant
    choosen_rect_coord_st = None
    choosen_rect_coord_end = None
    curr_var = None
    curr_time = previous_time
    counter = 1

    for i, w in enumerate(w_steps):
        for j, h in enumerate(h_steps):
            start_coord = (i * w_steps[0], j * h_steps[0])
            end_coord = (start_coord[0] + w, start_coord[1] + h)
            # Whether the current rectangle is selected
            if end_coord[0] >= center_coordinates[0] and end_coord[1] >= center_coordinates[1] \
                    and start_coord[0] <= center_coordinates[0] and start_coord[1] <= center_coordinates[1]:
                choosen_rect_coord_st = start_coord
                choosen_rect_coord_end = end_coord
                curr_var = counter
                if curr_var != previous_var:
                    curr_time = time.time()
                elif time.time() - previous_time > 2:
                    print('Variant chosen: ' + str(curr_var))
                    curr_time = time.time()
                counter += 1
                continue
            cv2.rectangle(image, start_coord, end_coord, WHITE_COLOR, 3)
            counter += 1

    # Plot currently chosen rectangle in GREEN
    cv2.rectangle(image, choosen_rect_coord_st,
                  choosen_rect_coord_end, GREEN_COLOR, 3)

    if type_used == 'eye':
        if not landmark_list:
            return
        if image.shape[2] != _RGB_CHANNELS:
            raise ValueError(
                'Input image must contain three channel rgb data.')
        image_rows, image_cols, _ = image.shape
        idx_to_eye_coordinates = {}
        idx_to_iris_coordinates = {}
        for idx, landmark in enumerate(landmark_list.landmark):

            landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                           image_cols, image_rows)
            if landmark_px and idx in eye_key_indicies:
                idx_to_eye_coordinates[idx] = landmark_px

            elif landmark_px and idx in iris_key_indicies:
                idx_to_iris_coordinates[idx] = landmark_px

        for landmark_px in idx_to_eye_coordinates.values():
            cv2.circle(image, landmark_px, 2,
                       eye_drwing_color, 1)

        for landmark_px in idx_to_iris_coordinates.values():
            cv2.circle(image, landmark_px, 2,
                       iris_drawing_color, 1)

    return image, center_coordinates, curr_var, curr_time


def find_nose_coord(landmark_list: landmark_pb2.NormalizedLandmarkList, image: np.ndarray) -> tuple:
    """This function returns nose center coordinates
    Args:
        landmark_list: A normalized landmark list proto message
        image: current frame 
    Returns:
        tuple: calculated center coordinates"""

    return (int(landmark_list.landmark[1].x * image.shape[1]), int(landmark_list.landmark[1].y * image.shape[0]))


def find_iris_center(landmark_list: landmark_pb2.NormalizedLandmarkList, image: np.ndarray) -> tuple:
    """This function returns iris center coordinates
    Args:
        landmark_list: A normalized landmark list proto message
        image: current frame 
    Returns:
        tuple: calculated center coordinates"""
    landmarks = landmark_list.landmark

    return (int((landmarks[473].x + landmarks[468].x) / 2 * image.shape[1]),
            int((landmarks[473].y + landmarks[468].y) / 2 * image.shape[0]))


def find_distance(landmark_list: landmark_pb2.NormalizedLandmarkList, image: np.ndarray) -> float:
    """This function calculates distance to the user's eyes from the camera
    Args:
        landmark_list: A normalized landmark list proto message
        image: current frame 
    Returns:
        float: calculated distance"""
    # Experimentally measured focal_length, for each device it should be measured separately
    focal_length = 1150  # pixels
    iris_length = 11.7  # mm
    iris_length_pixels_left = math.sqrt(((landmark_list.landmark[469].x - landmark_list.landmark[471].x) * image.shape[1]) ** 2
                                        + ((landmark_list.landmark[469].y - landmark_list.landmark[471].y) * image.shape[0]) ** 2)
    iris_length_pixels_right = math.sqrt(((landmark_list.landmark[476].x - landmark_list.landmark[474].x) * image.shape[1]) ** 2
                                         + ((landmark_list.landmark[476].y - landmark_list.landmark[474].y) * image.shape[0]) ** 2)
    distance = int(iris_length * focal_length /
                   ((iris_length_pixels_left + iris_length_pixels_right) / 2)) / 10

    return distance


def find_center_coordinates_enhanced(image: np.ndarray, previous_result: tuple, distance: float, curr_distance: float,
                                     eye_image_dimensions: tuple, upper_point: tuple, left_point: tuple, bottom_point: tuple,
                                     right_point: tuple, iris_center: tuple) -> tuple:
    """TEST FUNCTION
    This function calculated center coordinates of the point on the screen where the user is looking
    Args:
        image: image to be processed 
        previous_result: center_coordinates of the previous frame
        distance: distance from user's eyes to the camera from calibration
        curr_distance: distance from user's eyes to the camera now
        eye_image_dimensions: width and height of the reactangle where user's eyes moved during calibration
        upper_point: current position of user's upper point
        left_point: current position of user's left point
        bottom_point: current position of user's bottom point
        right_point: current position of user's right point
        iris_center: current position of user's pupil's center
    Returns:
        tuple: center coordinates of the point on the screen the user is looking at"""
    factor = curr_distance / distance
    # Smoothing parameter
    gamma = 0.9

    w = image.shape[1]
    h = image.shape[0]
    screen_center = (w/2, h/2)

    # center_y = (left_point[1] + right_point[1]) / 2
    # center_x = (upper_point[0] + bottom_point[0]) / 2
    center_y = (upper_point[1] + bottom_point[1]) / 2
    center_x = (upper_point[0] + bottom_point[0]) / 2

    radius = 3

    color = (255, 0, 255)

    thickness = -1
    cv2.circle(image, (int(center_x), int(center_y)), radius, color, thickness)
    color = (255, 255, 255)
    cv2.circle(image, (int(iris_center[0]), int(
        iris_center[1])), radius, color, thickness)

    difference_x = center_x - iris_center[0]
    difference_y = center_y - iris_center[1]

    y_shift = difference_y * h / eye_image_dimensions[1] * factor
    x_shift = difference_x * w / eye_image_dimensions[0] * factor

    result_x = (screen_center[0] - x_shift) * \
        (1 - gamma) + previous_result[0] * gamma
    result_y = (screen_center[1] - y_shift) * \
        (1 - gamma) + previous_result[1] * gamma

    if result_x < 0:
        result_x = 0
    elif result_x > w:
        result_x = w

    if result_y < 0:
        result_y = 0
    elif result_y > h:
        result_y = h

    result = (int(result_x), int(result_y))
    #previous_result = result
    print('difference: ' + str(difference_x) + '   ' + str(difference_y))
    print('shift: ' + str(x_shift) + '   ' + str(y_shift))
    print('result: ' + str(result))
    print('!!!!_____!!!!')
    return result


def find_center_coordinates(image: np.ndarray, previous_result: tuple, distance: float, curr_distance: float, center: tuple,
                            image_dimensions: tuple, point: tuple) -> tuple:
    """This function calculates center coordinates of the point on the screen where the user is looking
    Args:
        image: image to be processed 
        previous_result: center_coordinates of the previous frame
        distance: distance from user's eyes to the camera from calibration
        curr_distance: distance from user's eyes to the camera now
        center: position of user's "middle" eye when looking at the center of the screen from calibration
        eye_image_dimensions: width and height of the reactangle where user's eyes moved during calibration
        point: current position of user's eyes
    Returns:
        tuple: center coordinates of the point on the screen the user is looking at"""

    # Smoothing parameter
    gamma = 0.9

    w = image.shape[1]
    h = image.shape[0]
    screen_center = (w/2, h/2)

    difference_x = center[0] - point[0]
    difference_y = center[1] - point[1]

    factor = curr_distance / distance
    #factor = 1

    y_shift = difference_y * h / image_dimensions[1] * factor
    x_shift = difference_x * w / image_dimensions[0] * factor

    result_x = (screen_center[0] - x_shift) * \
        (1 - gamma) + previous_result[0] * gamma
    result_y = (screen_center[1] - y_shift) * \
        (1 - gamma) + previous_result[1] * gamma

    if result_x < 0:
        result_x = 0
    elif result_x > w:
        result_x = w

    if result_y < 0:
        result_y = 0
    elif result_y > h:
        result_y = h

    result = (int(result_x), int(result_y))
    # print('difference: ' + str(difference_x) + '   ' + str(difference_y))
    # print('shift: ' + str(x_shift) + '   ' + str(y_shift))
    # print('result: ' + str(result))
    return result


def find_ls(landmark_list: landmark_pb2.NormalizedLandmarkList, image: np.ndarray) -> list:
    """This function calculates distances from the "middle" eye's center to its top, bottom, left and right corners
    Args:
        landmark_list: A normalized landmark list proto message
        image: current frame 
    Returns:
        list: distances from the "middle" eye's center to its top, bottom, left and right corners"""
    # find "middle_eye" by getting the average of 5 points on left and right eyes
    eye_center, right_point, left_point, bottom_point, upper_point = find_points(
        landmark_list, image)

    # Calculate distances
    l1 = math.dist(upper_point, eye_center)
    l2 = math.dist(left_point, eye_center)
    l3 = math.dist(bottom_point, eye_center)
    l4 = math.dist(right_point, eye_center)

    return [l1, l2, l3, l4]


def find_points(landmark_list: landmark_pb2.NormalizedLandmarkList, image: np.ndarray) -> list:
    """This function calculates distances from the "middle" eye's center to its top, bottom, left and right corners
    Args:
        landmark_list: A normalized landmark list proto message
        image: current frame 
    Returns:
        list: distances from the "middle" eye's center to its top, bottom, left and right corners"""
    # find "middle_eye" by getting the average of 5 points on left and right eyes
    eye_center = find_iris_center(landmark_list, image)
    landmarks = landmark_list.landmark
    right_point = ((landmarks[35].x + landmarks[464].x) / 2 * image.shape[1],
                   (landmarks[35].y + landmarks[464].y) / 2 * image.shape[0])
    left_point = ((landmarks[245].x + landmarks[265].x) / 2 * image.shape[1],
                  (landmarks[245].y + landmarks[265].y) / 2 * image.shape[0])
    bottom_point = ((landmarks[119].x + landmarks[348].x) / 2 * image.shape[1],
                    (landmarks[119].y + landmarks[348].y) / 2 * image.shape[0])
    upper_point = ((landmarks[65].x + landmarks[295].x) / 2 * image.shape[1],
                   (landmarks[65].y + landmarks[295].y) / 2 * image.shape[0])

    return eye_center, right_point, left_point, bottom_point, upper_point
