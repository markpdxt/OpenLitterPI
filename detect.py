# OpenLitterPI - Automated cat litterbox
# Copyright (C) 2025 Mark Nelson
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import sys
import cv2
import time
import argparse
import numpy as np
from enum import Enum
from tflite_support.task import core
from tflite_support.task import vision
from tflite_support.task import processor
from datetime import datetime, timedelta
import utils

class Status(Enum):
    IDLE = 0
    DETECTED = 1
    USING = 2
    WAITING = 3
    CYCLING = 4
    COMPLETE = 5

def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool, display: bool) -> None:

    # Litterbox
    occupied_frames_threshold = 15
    use_threshold = 45   # seconds
    wait_threshold = 60*7
    reset_threshold = 60*8
    status = Status.IDLE
    timestamp_last_detected = 2556057600 # 2050
    elapsed_time = 0
    occupied_frames = 0
    isStartup = True

    # Calculate FPS
    counter, fps = 0, 0
    start_time = time.time()

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Visualization parameters
    row_size = 20  # pixels
    left_margin = 10  # pixels
    text_color = (0, 0, 255)  # red
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    # Initialize the object detection model
    base_options = core.BaseOptions(
        file_name=model, use_coral=False, num_threads=num_threads)
    detection_options = processor.DetectionOptions(
        max_results=5, score_threshold=0.35)
    options = vision.ObjectDetectorOptions(
        base_options=base_options, detection_options=detection_options)
    detector = vision.ObjectDetector.create_from_options(options)

    # Continuously capture images from the camera and run inference
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )

        counter += 1

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create a TensorImage object from the RGB image.
        input_tensor = vision.TensorImage.create_from_array(rgb_image)

        # Run object detection estimation using the model.
        detection_result = detector.detect(input_tensor)

        # Draw keypoints and edges on input image
        image = utils.visualize(image, detection_result)

        if isStartup == True:
            utils.send_message("Startup", image)
            isStartup = False

        for detection in detection_result.detections:
            category = detection.categories[0]
            category_name = category.category_name
            probability = round(category.score, 2)
            result_text = category_name + ' (' + str(probability) + ')'

            if (category_name == 'cat' or category_name == 'teddy bear'):
                occupied_frames = occupied_frames + 1
                elapsed_time = occupied_frames
                timestamp_last_detected = int(round(datetime.now().timestamp()))

                if occupied_frames <= occupied_frames_threshold:
                    if status != Status.DETECTED:
                        status = Status.DETECTED
                        utils.send_message(status.name, image)
                    elapsed_time = occupied_frames
                elif occupied_frames > occupied_frames_threshold:
                    if status != Status.USING:
                        status = Status.USING
                        utils.send_message(status.name, image)
                    elapsed_time = occupied_frames

        object_departed = int(round(datetime.now().timestamp())) - timestamp_last_detected

        if status.value > 1: # USING or above
            if object_departed > use_threshold and object_departed <= wait_threshold:
                if status != Status.WAITING:
                    status = Status.WAITING
                    utils.send_message(status.name, image)
                elapsed_time = object_departed
            elif object_departed > wait_threshold:
                if status != Status.CYCLING and status != Status.COMPLETE:
                    status = Status.CYCLING
                    utils.send_message(status.name, image)
                    elapsed_time = object_departed
                    utils.cycle()
                    status = Status.COMPLETE
                    utils.send_message(status.name, image)
                    elapsed_time = occupied_frames = 0

        # if status == Status.DETECTED and int(round(datetime.now().timestamp())) - timestamp_last_detected > wait_threshold:
        #     reset()

        since_detected = int(round(datetime.now().timestamp())) - timestamp_last_detected
        print(f'time since detected: {since_detected}')
        print(f'timestamp_last_detected: {timestamp_last_detected}')
        print(f'reset_threshold: {reset_threshold}')
        print('-----------------')

        if status != Status.IDLE and since_detected > reset_threshold:
            reset()

        # Calculate the FPS
        if counter % fps_avg_frame_count == 0:
            end_time = time.time()
            fps = fps_avg_frame_count / (end_time - start_time)
            start_time = time.time()

        # Show the Status
        text = '{}'.format(status)
        text_location = (left_margin, row_size)
        cv2.putText(image, text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    font_size, text_color, font_thickness)

        # Show the elapsed time
        text = '{} sec'.format(elapsed_time)
        text_location = (left_margin, row_size * 2)
        if elapsed_time > 0:
            cv2.putText(image, text, text_location, cv2.FONT_HERSHEY_PLAIN, font_size, text_color, font_thickness)
        
        # Show the FPS
        text = 'FPS: {:.1f}'.format(fps)
        text_location = (left_margin, row_size * 3)
        cv2.putText(image, text, text_location, cv2.FONT_HERSHEY_PLAIN, font_size, text_color, font_thickness)

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break

        if display:
            cv2.imshow('object_detector', image)

    cap.release()
    cv2.destroyAllWindows()

def reset():
    status = Status.IDLE
    elapsed_time = occupied_frames = 0
    timestamp_last_detected = 2556057600 # 2050
    # utils.send_message(status.name, image)

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='models/efficientdet_lite0.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  parser.add_argument(    
      '--display', 
      help='View video on local device', 
      required=False,
      default=False)       
  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU), bool(args.display))
if __name__ == '__main__':
    main()
