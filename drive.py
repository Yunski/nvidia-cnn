import argparse
import base64
import os
import sys
import shutil
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import asyncio
import utils
import tensorflow as tf

from datetime import datetime
from PIL import Image
from flask import Flask
from io import BytesIO

from freeze import freeze_graph
from load import load_graph

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
Deploy model for use with Udacity's Car Simulator
Adapted from https://github.com/naokishibuya/car-behavioral-cloning
"""

sio = socketio.Server()
app = Flask(__name__)

MAX_SPEED = 25
MIN_SPEED = 10

speed_limit = MAX_SPEED

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        steering_angle = float(data["steering_angle"])
        throttle = float(data["throttle"])
        speed = float(data["speed"])
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
            image = np.asarray(image) 
            image = utils.preprocess(image) 
            image = np.array([image])

            steering_angle = sess.run(prediction, feed_dict={X: image, keep_prob: 1.0})[0]
            
            # lower the throttle as the speed increases
            # if the speed is above the current speed limit, we are on a downhill.
            # make sure we slow down first and then go back to the original max speed.
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2

            print('{} {} {}'.format(steering_angle, throttle, speed))
            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)

        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


async def check_model_path(args):
    folder = os.path.dirname(args.model) 
    if not os.path.exists(folder):
        raise Exception("Model folder does not exist.")
    file_exists = os.path.exists(args.model)  
    if not file_exists or args.update:
        filename = os.path.basename(args.model)
        filename, ext = os.path.splitext(filename)
        if ext != '.pb':
            raise Exception("File must have .pb extension.")
        if args.update and file_exists:
            print("Overwriting existing model .pb file...")
        else:
            print("Model file does not exist. Creating {}.pb file...".format(filename))
        freeze_graph(folder, filename, 'output')
        print("Finished creating {}.pb".format(filename))
        print("Please run the drive.py again with the created model file.")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Remote Driving")
    parser.add_argument(
        'model',
        type=str,
        help="Path to model .pb file"
    )
    parser.add_argument(
        '-f',
        type=str,
        dest='image_folder',
        nargs='?',
        default='',
        help="Path to image folder. This is where the images from the run will be saved."
    )
    parser.add_argument(
        '-u',
        action='store_true',
        dest='update',
        help="Overwrite/update the specified model file."
    )        
    args = parser.parse_args()

    ioloop = asyncio.get_event_loop()
    ioloop.run_until_complete(check_model_path(args))
    ioloop.close()    

    print("Loading from {}".format(args.model))
    graph = load_graph(os.path.join(args.model))
    X = graph.get_tensor_by_name('prefix/X:0')
    keep_prob = graph.get_tensor_by_name('prefix/keep_prob:0')
    prediction = graph.get_tensor_by_name('prefix/output:0')
    sess = tf.Session(graph=graph)
        
    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

