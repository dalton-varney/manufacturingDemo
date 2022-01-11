import time
import edgeiq
from flask_socketio import SocketIO
from flask import Flask, render_template, request, send_file, url_for, redirect
import base64
import threading
import logging
from eventlet.green import threading as eventlet_threading
import cv2
from collections import deque
import numpy as np

app = Flask(__name__, template_folder='./templates/')

socketio_logger = logging.getLogger('socketio')
socketio = SocketIO(
    app, logger=socketio_logger, engineio_logger=socketio_logger)

SESSION = time.strftime("%d%H%M%S", time.localtime())
video_stream = edgeiq.FileVideoStream("Video20.mov", play_realtime=True)
obj_detect = edgeiq.ObjectDetection("alwaysai/hand_detection")
obj_detect.load(engine=edgeiq.Engine.DNN)
SAMPLE_RATE = 50


@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')


@socketio.on('connect')
def connect_cv():
    print('[INFO] connected: {}'.format(request.sid))


@socketio.on('disconnect')
def disconnect_cv():
    print('[INFO] disconnected: {}'.format(request.sid))


@socketio.on('close_app')
def close_app():
    print('Stop Signal Received')
    controller.close()

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Check to see if two rectangles are overlapping
def Overlap(l1, r1, l2, r2):
    # If one rectangle is on left side of other
    if(l1.x > r2.x or l2.x > r1.x):
        return False

    # If one rectangle is above other
    if(l1.y > r2.y or l2.y > r1.y):
        return False

    return True

def detect_roi(frame, min_x, min_y, max_x, max_y, height=60, color = (255, 0, 0)):
    """
    """
    start_point = (min_x, (min_y-height))
    end_point = (max_x, max_y)
    rectangle = (min_x, (min_y-height), max_x, max_y)
    frame = cv2.rectangle(frame, start_point, end_point, color, thickness=2)
    return frame, rectangle


class CVClient(eventlet_threading.Thread):
    def __init__(self, fps, exit_event):
        """The original code was created by Eric VanBuhler
        (https://github.com/alwaysai/video-streamer) and is modified here.

        Initializes a customizable streamer object that
        communicates with a flask server via sockets.

        Args:
            stream_fps (float): The rate to send frames to the server.
            exit_event: Threading event
        """
        self._stream_fps = SAMPLE_RATE
        self.fps = fps
        self._last_update_t = time.time()
        self._wait_t = (1/self._stream_fps)
        self.exit_event = exit_event
        self.all_frames = deque()
        self.video_frames = deque()
        super().__init__()

    def setup(self):
        """Starts the thread running.

        Returns:
            CVClient: The CVClient object
        """
        self.start()
        time.sleep(1)
        return self

    def run(self):
        # loop detection
        video_stream.start()

        socketio.sleep(0.01)
        self.fps.start()
        # hsv max min boundaries for specific light detection, boundaries numbers
        # represent hue, saturation and value parameters
        boundaries = [([6, 74, 226], [68, 187, 255])]
        # loop detection
        while True:
            try:
                frame = video_stream.read()
                text = [""]
                socketio.sleep(0.01)
                # set bounding box color to red
                colors = [(255, 255, 255), (0, 0, 255)]
                grabbed_status = None
                hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                for (lower, upper) in boundaries:
                    # create NumPy arrays from the boundaries
                    lower = np.array(lower, dtype = "uint8")
                    upper = np.array(upper, dtype = "uint8")
                    # find the colors within the specified boundaries and apply
                    # the mask
                    threshold_mask = cv2.inRange(hsv_image, lower, upper)
                    threshold_image = cv2.bitwise_and(hsv_image, hsv_image, mask=threshold_mask)
                    nonzero_pixel_list = (np.transpose(np.nonzero(threshold_mask))).tolist()
                    # check tray boundaries
                    output = [item for item in nonzero_pixel_list if item[1] > 130 and item[1] < 380]
                    output1 = [item for item in output if item[0] > 97]

                    if output1:
                        # use numpy array to find mean coordinates of the lit area
                        # and list min and max funtions to find the min and max
                        # coordinates of the lit area
                        output_array = np.array(output1)
                        avg = np.mean(output_array, axis=0)
                        avg = avg.astype('int32')
                        max_pixel = max(output1)
                        min_pixel= min(output1)
                        x_check = (max_pixel[1] - min_pixel[1])
                        # adjust roi to account for hand covering LED"
                        if x_check < 40:
                            min_pixel[1] = (max_pixel[1] - 55)

                        # use opencv circle to place min, mean and max coordinates
                        # on the frame
                        cv2.circle(frame, (avg[1], avg[0]), 4, (0, 0, 0), -1)
                        cv2.circle(frame, (min_pixel[1], min_pixel[0]), 4, (0, 0, 0), -1)
                        cv2.circle(frame, (max_pixel[1], max_pixel[0]), 4, (0, 0, 0), -1)

                        frame, rectangle1 = detect_roi(frame, min_pixel[1], min_pixel[0],
                            max_pixel[1], max_pixel[0])
                results = obj_detect.detect_objects(frame, confidence_level=.6)

                for prediction in results.predictions:
                    rectangle = prediction.box
                    l2 = Point(rectangle.start_x, rectangle.start_y)
                    r2 = Point(rectangle.end_x, rectangle.end_y)
                    if rectangle1 is not None:
                        l1 = Point(rectangle1[0], rectangle1[1])
                        r1 = Point(rectangle1[2], rectangle1[3])
                        if Overlap(l1, r1, l2, r2):
                            grabbed_status = "component grabbed"
                            # set bounding box color to green
                            colors = [(255, 255, 255), (0, 255, 0)]

                frame = edgeiq.markup_image(
                        frame, results.predictions, colors = colors)

                # Generate text to display on streamer
                text = ["Model: {}".format(obj_detect.model_id)]
                text.append(
                        "Inference time: {:1.3f} s".format(results.duration))

                if not grabbed_status:
                    text.append("No component grabbed")
                else:
                    text.append("{} from tray".format(grabbed_status))

                combined = np.vstack((frame,threshold_image))
                self.send_data(combined, text)
                socketio.sleep(0.01)
                self.fps.update()


                if self.check_exit():
                    video_stream.stop()
                    controller.close()
            except edgeiq.NoMoreFrames:
                video_stream.start()

    def _convert_image_to_jpeg(self, image):
        """Converts a numpy array image to JPEG

        Args:
            image (numpy array): The input image

        Returns:
            string: base64 encoded representation of the numpy array
        """
        # Encode frame as jpeg
        frame = cv2.imencode('.jpg', image)[1].tobytes()
        # Encode frame in base64 representation and remove
        # utf-8 encoding
        frame = base64.b64encode(frame).decode('utf-8')
        return "data:image/jpeg;base64,{}".format(frame)

    def send_data(self, frame, text):
        """Sends image and text to the flask server.

        Args:
            frame (numpy array): the image
            text (string): the text
        """
        cur_t = time.time()
        if cur_t - self._last_update_t > self._wait_t:
            self._last_update_t = cur_t
            frame = edgeiq.resize(
                    frame, width=720, height=900, keep_scale=False)
            socketio.emit(
                    'server2web',
                    {
                        'image': self._convert_image_to_jpeg(frame),
                        'text': '<br />'.join(text)
                    })
            socketio.sleep(0.01)

    def check_exit(self):
        """Checks if the writer object has had
        the 'close' variable set to True.

        Returns:
            boolean: value of 'close' variable
        """
        return self.exit_event.is_set()

    def close(self):
        """Disconnects the cv client socket.
        """
        self.exit_event.set()


class Controller(object):
    def __init__(self):
        self.fps = edgeiq.FPS()
        self.cvclient = CVClient(self.fps, threading.Event())

    def start(self):
        self.cvclient.start()
        print('[INFO] Starting server at http://localhost:5000')
        socketio.run(app=app, host='0.0.0.0', port=5000)

    def close(self):
        self.fps.stop()
        print("elapsed time: {:.2f}".format(self.fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(self.fps.compute_fps()))

        if self.cvclient.is_alive():
            self.cvclient.close()
            self.cvclient.join()

        print("Program Ending")


controller = Controller()

if __name__ == "__main__":
    try:
        controller.start()
    finally:
        controller.close()
