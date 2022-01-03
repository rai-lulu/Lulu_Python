from flask import Flask, render_template, redirect
from flask.helpers import url_for
from flask_socketio import SocketIO, emit
import time
import io
from PIL import Image
import base64
import cv2
import numpy as np
from numpy.lib.twodim_base import eye
import pyshine as ps
import mediapipe as mp
import custom_drawing_utils
from statistics import mean
from pathlib import Path
import sys

from engineio.payload import Payload

FILE = Path(__file__).absolute()
sys.path.append(FILE.as_posix())

Payload.max_decode_packets = 2048

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')


@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')


@app.route('/calibration_gaze', methods=['POST', 'GET'])
def calibration_gaze():
    global type_used
    refresh()
    type_used = 'eye'
    return render_template('calibration.html')


@app.route('/calibration_nose', methods=['POST', 'GET'])
def calibration_nose():
    global type_used
    refresh()
    type_used = 'nose'
    return render_template('calibration.html')


@app.route('/tracking', methods=['POST', 'GET'])
def tracking():
    return render_template('tracking.html')


@app.errorhandler(Exception)
def request_entity_too_large(error):
    return render_template('index.html')


def readb64(base64_string):
    idx = base64_string.find('base64,')
    base64_string = base64_string[idx+7:]

    sbuf = io.BytesIO()

    sbuf.write(base64.b64decode(base64_string, ' /'))
    pimg = Image.open(sbuf)

    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)


def moving_average(x):
    return np.mean(x)


@socketio.on('catch-frame')
def catch_frame(data):

    emit('response_back', data)


global fps, prev_recv_time, cnt, fps_array, eye_center, distance, points, distances, eye_image_dimensions, previous_result,\
    previous_var, previous_time, flag, type_used, counter

def refresh():
    """This function returns all global variables to their starting values"""
    global fps, prev_recv_time, cnt, fps_array, eye_center, distance, points, distances, eye_image_dimensions, previous_result,\
        previous_var, previous_time, flag, type_used, counter
    fps = 30
    prev_recv_time = 0
    cnt = 0
    fps_array = [0]
    eye_center = None
    distance = 0
    points = []
    distances = []
    eye_image_dimensions = None
    previous_result = (640, 360)
    previous_var = 1
    previous_time = 0
    flag = False
    type_used = None
    counter = 0

refresh()
mp_drawing = custom_drawing_utils
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.5)

global prev_time, factors
prev_time = 0
#Center, left, right, up, down
factors = [(1/2, 1/2), (0, 1/2), (1, 1/2), (1/2, 0), (1/2, 1)]

@socketio.on('image_calibration')
def calibration_image(data_image):
    """This function emits image during calibration
    Args:
        data_image: image from webcam"""
    global cnt, prev_time, factors, counter, eye_center, distance, points, distances, eye_image_dimensions, flag, type_used

    if flag:
        return

    if prev_time == 0:
        counter = 0
        prev_time = time.time()

    curr_time = time.time()
    difference = curr_time - prev_time

    frame = (readb64(data_image))

    frame = cv2.flip(frame, 1)
    frame = draw_calibraion(frame, factors[counter])

    # Time interval between calibration points
    if difference >= 4:
        counter += 1
        prev_time = curr_time
        results = face_mesh.process(frame)
        if results.multi_face_landmarks:
            if type_used == 'nose':
                try:
                    eye_image_dimensions = (100, 50)
                    distance = 45
                    eye_center = mp_drawing.find_nose_coord(
                        results.multi_face_landmarks[0], frame)
                    counter = 0
                    prev_time = 0
                    flag = True
                    emit('redirect', {'url': url_for('tracking')})
                except:
                    print("EXCEPTION OCCURED!!!!!")
                    flag = True
                    emit('redirect', {'url': url_for('index')})

            landmarks = results.multi_face_landmarks[0].landmark
            points.append(((landmarks[473].x + landmarks[468].x) / 2 * frame.shape[1],
                           (landmarks[473].y + landmarks[468].y) / 2 * frame.shape[0]))
            distances.append(mp_drawing.find_distance(
                results.multi_face_landmarks[0], frame))
        else:
            flag = True
            emit('redirect', {'url': url_for('index')})
    if counter == 5:
        counter = 0
        prev_time = 0
        try:
            eye_image_height = points[4][1] - points[3][1]
            eye_image_width = points[2][0] - points[1][0]
            if eye_image_height < 0 or eye_image_width < 0:
                raise Exception
            eye_image_dimensions = (eye_image_width, eye_image_height)
            distance = mean(distances)
            eye_center = (int(points[0][0]), int(points[0][1]))
            flag = True
            emit('redirect', {'url': url_for('tracking')})

        except:
            print("EXCEPTION OCCURED!!!!!")
            flag = True
            emit('redirect', {'url': url_for('index')})

    imgencode = cv2.imencode(
        '.jpeg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])[1]

    # base64 encode
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpeg;base64,'
    stringData = b64_src + stringData

    # emit the frame back
    emit('response_back', stringData)

    cnt += 1
    if cnt == 30:
        cnt = 0


def draw_calibraion(image: np.ndarray, factor: tuple) -> np.ndarray:
    """This function draws needed dots on the current image during calibration
    Args:
        image: image to be processed
        factor: describes where the current dot needs to be drawn
    Returns:
        image: processed image"""
    # image.flags.writeable = False
    # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 500)
    fontScale = 1
    fontColor = (255, 255, 255)
    thickness = 1
    lineType = 2

    cv2.putText(image, 'Please, look at the red circle',
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

    return image


@socketio.on('image_tracking')
def tracking_image(data_image):
    """This function emits image during gaze_tracking
    Args:
        data_image: image to be processed"""
    global fps, cnt, prev_recv_time, fps_array
    recv_time = time.time()
    text = 'FPS: '+str(fps)
    frame = (readb64(data_image))
    frame = draw_results(frame)
    frame = ps.putBText(frame, text, text_offset_x=20, text_offset_y=30, vspace=20,
                        hspace=10, font_scale=1.0, background_RGB=(10, 20, 222), text_RGB=(255, 255, 255))
    imgencode = cv2.imencode('.jpeg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])[1]

    # base64 encode
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpeg;base64,'
    stringData = b64_src + stringData

    # emit the frame back
    emit('response_back', stringData)

    fps = 1/(recv_time - prev_recv_time)
    fps_array.append(fps)
    fps = round(moving_average(np.array(fps_array)), 1)
    prev_recv_time = recv_time
    # print(fps_array)
    cnt += 1
    if cnt == 30:
        fps_array = [fps]
        cnt = 0


def draw_results(frame: np.ndarray) -> np.ndarray:
    """This function draws the point on the screen the user is looking at
    Args:
        frame: image to be processed
    Returns:
        frame: processed image"""
    global previous_result, previous_time, previous_var, eye_center, eye_image_dimensions, distance, type_used
    frame.flags.writeable = False
    frame = cv2.flip(frame, 1)
    results = face_mesh.process(frame)
    frame.flags.writeable = True
    if results.multi_face_landmarks:
        frame, previous_result, previous_var, previous_time = mp_drawing.tracking(type_used, previous_result,
                                                                                    previous_var,
                                                                                    previous_time, distance,
                                                                                    eye_center,
                                                                                    eye_image_dimensions,
                                                                                    image=frame,
                                                                                    landmark_list=results.multi_face_landmarks[0])
    return frame


if __name__ == '__main__':
    # socketio.run(app, host='0.0.0.0', port=9990, debug=True)
    socketio.run(app, port=9990, debug=True)
