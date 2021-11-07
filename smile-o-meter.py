#!/bin/python

#
# Raspberry Pi based trainable Machine Learning smile-o-meter.
#
# Requirements
# Raspberry pi (tested on RPi 4b)
# Camera either USB or Pi Camera 1.3 connected via ribbon cable.
# LED bargraph connected to GPIO pins defined in BAR_PINS_BCM.
#
# Operation
# On startup the device will self test the LEDS then check for an
# existing trained model file (.pkl).  If it finds one, it will
# immediately start face recognition using an attached camera.
# Using the trained model it will select the closest maching 
# smile index to the first face identified in the image and
# display the result on the bargraph.
#
# If no trained model is identified, teaching mode is initiated.
#
# Teaching - Collect training data
# Teaching mode is initiated on startup automatically if there no trained 
# model is found.
# Datacollection/training is indicated by the last (blue) LED.
# Teaching mode can be initiated by pressing and holding the PCB button
#
# The blue LED illuminates to indicate data collection is
# starting.  Data collection then proceeds as follows:
# Each smile index is trained in turn, starting with the lowest smile index,
# i.e. the saddest (red).  You will to pace yourself to define 9 different
# smile levels increasing in happiness.
# The machine learning is looking at all facial features, eyes, cheeks etc,
# so make your expressions as genuine as possible.
# The LED starts fast flashing, to indicate time to adopt your pose and get 
# ready for data collection.
# After 5 seconds, the LED goes steady.  Live data is being collected.
# Hold your smile pose, and move your head around/tilting/moving 
# forwards/backwards etc.
# After 5 more seconds the next LED starts flashing, get ready to adopt
# smile level 2.  
# Continue until you reach smile level 9.
#
# Teaching - Training based on model data
# Once complete the RPi will start training it's model based on the data
# you collected.  This is indicated by a fast flashing blue LED.
# It is computationally intensive, for the RPi, it may take 2-3 minutes and 
# the RPi will get hot (heatsink is a good idea.)
# After teaching has finished, the blue LED stops flashing, and the device 
# goes into normal operation mode.

# Other
# During normal operation, if no face is identified, no LEDs are illuminated.




import pickle

import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
from enum import Enum

import csv
import os.path
import numpy as np
import sys
import time
import pandas as pd
from time import sleep
import RPi.GPIO as GPIO

# amount of time to wait between steps
INTER_STEP_INTERVAL = 0.1

# Pre data collection warmup duration (Seconds)
PRE_COLLECT_DELAY = 5

# Max number of seconds to collect data for.
COLLECT_INDEX_TIMEOUT = 20

# Number of frames to collect per index
NUM_FRAMES_PER_INDEX = 100

# Length of time the button needs to be held to start training
BUTTON_LONG_PRESS_TIME = 3

# Defines the BCM pinout from 1 to 10 for the bargraph LED.
BAR_PINS_BCM = [ 21, 20, 16, 12, 0, 5, 6, 13, 19, 26 ]

# Pin number where the button is connective (active low)
BUTTON_PIN_BCM = 1

# Place where the coordinates are collected and saved prior to training.
COORDS_FILE = "coords1.csv"

# Place where the trained model is saved ready for reloading next time.
SMILE_INDEX_MODEL_FILE = 'smile_index.pkl'

pwm_register = {}

def setup_bar():
    GPIO.setmode(GPIO.BCM)
    for pin in BAR_PINS_BCM:
        GPIO.setup(pin, GPIO.OUT)
    # nightrider led test at startup
    for _ in range(1):
        for i in range(11):
            show_bar(i, False)
            sleep(0.02)
        for i in range(11, 0, -1):
            show_bar(i, False)
            sleep(0.02)  
    show_bar(0) 

def setup_button():
    GPIO.setup(BUTTON_PIN_BCM, GPIO.IN)

def get_button() -> bool:
    button = not GPIO.input(BUTTON_PIN_BCM)
    return button

def start_led_flashing(pin: int, freq: float=10, duty_percent:int = 50):
    p = pwm_register.get(pin)
    if p is not None:
        p.ChangeFrequency(freq)
        p.ChangeDutyCycle(duty_percent)
    else:
        p = GPIO.PWM(pin, freq)
        p.start(duty_percent)
        pwm_register[pin] = p

def stop_led_flashing(pin: int):
    p = pwm_register.get(pin)
    if p is not None:
        p.ChangeDutyCycle(0)
    else:
        raise Exception("Could not find handle to stop pwm on pin %s" % pin)
    GPIO.output(pin, GPIO.LOW)

def set_led(pin: int, state: bool):
    """
    Intelligently re-use pins.  If they've been set as pwm use that,
    if not, use gpio mode. This works round an apparent bug(?) where
    once a pin is initialised as PWM, it is no longer usable as GPIO.
    """
    p = pwm_register.get(pin)
    if p is not None:
        p.ChangeDutyCycle(100 if state else 0)
    else:
        GPIO.output(pin, state)


def show_bar(value: int, cumulative=True):
    """Show a bar graph on the board mounted LEDs"""
    #print(f"value={value}")
    for i, pin in enumerate(BAR_PINS_BCM):
        if cumulative:
            set_led(pin, value >= i+1)
        else:
            set_led(pin, value == i+1)


def get_face_row(face_analyser, cap) -> list:
    """Helper function to get a single flattened row of face coordinate data."""
    _, frame = cap.read()
    
    # Recolor Feed
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False        
    
    # Make Detections
    results = face_analyser.process(image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 1. Draw face landmarks
            # mp_drawing.draw_landmarks(image, face_landmarks, mp_face.FACEMESH_CONTOURS, 
            #                         mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            #                         mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
            #                         )
            facelm = face_landmarks.landmark
            face_row = list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in facelm]).flatten())
            X = pd.DataFrame([face_row])
            return face_row
    return None

    
def collect_smile_index_data(face_analyser, cap) -> bool:
    """Collects data from camera to CSV file."""
    ret = False
    if not cap.isOpened():
        print("No video capture open.")
        return ret

    # turn on blue LED to indicate collecting data
    print("Collecting data...")
    for p in BAR_PINS_BCM:
        set_led(p, False)
    set_led(BAR_PINS_BCM[-1], True)
    try:
        start_led_flashing(BAR_PINS_BCM[0], 2)
        # create the file header.  Coordinates are in groups of 4.
        start_time = time.time()
        # get 1 face just to count the coordinates.
        face_row = None
        while face_row is None:
            face_row = get_face_row(face_analyser, cap)
            if (time.time() - start_time) > COLLECT_INDEX_TIMEOUT:
                raise TimeoutError("could not get first frame")
        num_coords = int(len(face_row)/4)
        print(f"num_coords={num_coords}")
        with open(COORDS_FILE, mode='w', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            landmarks = ['smileindex']
            for val in range(1, num_coords+1):
                landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
            csv_writer.writerow(landmarks)

        for i in range(9):
            # count from 0 to 8. Generate smile indexes from 1 to 9.
            smile_index = i+1
            pin = BAR_PINS_BCM[i]
            start_led_flashing(pin)
            sleep(PRE_COLLECT_DELAY)

            start_time = time.time()
            rowcount = 0
            while ((time.time() - start_time) < COLLECT_INDEX_TIMEOUT):
                face_row = get_face_row(face_analyser, cap)
                if face_row is not None:
                    rowcount = rowcount + 1
                    set_led(pin, True)
                    face_row.insert(0, smile_index)
                    #print(face_row[5:])
                    with open(COORDS_FILE, mode='a', newline='') as f:
                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(face_row)
                else:
                    set_led(pin, False)
                if rowcount >= NUM_FRAMES_PER_INDEX:
                    break
            if rowcount < NUM_FRAMES_PER_INDEX:
                print("Timed out")
            set_led(pin, False)
            print(f"{smile_index}: {rowcount} rows")
            sleep(INTER_STEP_INTERVAL)
        ret = True
    except TimeoutError as e:
        print("Timed out: %s" % str(e))
    finally:
        # turn off blue LED - data collection finished.
        set_led(BAR_PINS_BCM[-1], False)
    return ret

def load_smile_index_model():
    print("Loading model...")
    try:
        with open(SMILE_INDEX_MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
            return model
    except FileNotFoundError:
        print("Model not found!")
        return None


def train_smile_index_model():
    """"""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from datetime import datetime

    start_led_flashing(BAR_PINS_BCM[-1])
    print("Training model...")
    df = pd.read_csv(COORDS_FILE)

    X = df.drop('smileindex', axis=1)
    y = df['smileindex']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

    from sklearn.pipeline import make_pipeline 
    from sklearn.preprocessing import StandardScaler 

    from sklearn.linear_model import LinearRegression #, RidgeClassifier
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, \
            BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor

    pipelines = {
    #    'rf':make_pipeline(StandardScaler(), RandomForestRegressor(n_jobs=-1)),
        'et':make_pipeline(StandardScaler(), ExtraTreesRegressor(n_jobs=-1)),
    #    'b':make_pipeline(StandardScaler(), BaggingRegressor(n_jobs=-1)),
        #'ab':make_pipeline(StandardScaler(), AdaBoostRegressor(n_jobs=-1)),
        #'gb':make_pipeline(StandardScaler(), GradientBoostingRegressor(n_jobs=-1)),
        #'v':make_pipeline(StandardScaler(), VotingRegressor()),
        #'s':make_pipeline(StandardScaler(), StackingRegressor()),
    }

    fit_models = {}
    print(f"{datetime.now()}: Training start..")

    for algo, pipeline in pipelines.items():
        model = pipeline.fit(X_train, y_train)
        fit_models[algo] = model
        print(f"{datetime.now()}: {algo} complete.")

    from sklearn.metrics import accuracy_score
    import pickle

    errors = {}
    for algo, model in fit_models.items():
        yhat = model.predict(X_test)
        errors = yhat - y_test
        df = pd.DataFrame(errors)
        print(f"Algorithm '{algo}' performance summary (error spread of smileindex prediction):")
        print(df.describe())
        #print(algo, accuracy_score(y_test, yhat))

    # 'extra trees' model seems to work well
    with open(SMILE_INDEX_MODEL_FILE, 'wb') as f:
        pickle.dump(fit_models['et'], f)
    set_led(BAR_PINS_BCM[-1], False)

def apply_smile_index_model(face_analyser, model, cap) -> int:
    if not cap.isOpened():
        return 0
    _, frame = cap.read()
    
    # Recolor Feed
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False        
    
    # Make Detections
    results = face_analyser.process(image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            facelm = face_landmarks.landmark
            face_row = list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in facelm]).flatten())
            X = pd.DataFrame([face_row])
            # predict the smile index
            smile_index = model.predict(X)[0]
            return smile_index
    # if we didn't get any faces, return 0.
    return 0


import threading
import cv2

# Define the thread that will continuously pull frames from the camera
# Avoids processing old frames by throwing them all away.
class CameraBufferCleanerThread(threading.Thread):
    def __init__(self, camera, name='camera-buffer-cleaner-thread'):
        self.camera = camera
        self.last_frame = None
        super(CameraBufferCleanerThread, self).__init__(name=name)
        self.start()

    def run(self):
        while True:
            ret, self.last_frame = self.camera.read()
    
    def read(self):
        return None, self.last_frame
    
    def isOpened(self):
        return self.camera.isOpened()


if __name__ == "__main__":
    try:
        setup_bar()
        setup_button()


        #mp_drawing = mp.solutions.drawing_utils # Drawing helpers
        #mp_holistic = mp.solutions.holistic # Mediapipe Solutions
        mp_face = mp.solutions.face_mesh
        print("Start...")
        cap_raw = cv2.VideoCapture(0)
        cap = CameraBufferCleanerThread(cap_raw)
        print("capture started.")        

        retrain_flag = False

        # Initiate holistic model
        with mp_face.FaceMesh() as face_analyser:
            
            model = load_smile_index_model()

            while True:
                if retrain_flag or (model is None):
                    print("Retrain...")
                    model = None
                    while model is None:
                        while collect_smile_index_data(face_analyser, cap) == False:
                            pass
                        train_smile_index_model()
                        model = load_smile_index_model()
                        retrain_flag = False
                smile_index = apply_smile_index_model(face_analyser, model, cap)
                show_bar(int(smile_index))

                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    break   

                button = get_button()
                if button == True:
                    print("Button pressed")
                    start_time = time.time()
                    while get_button() == True:
                        sleep(0.1)
                        if (time.time() - start_time) > BUTTON_LONG_PRESS_TIME: 
                            print("Triggered retrain.")
                            retrain_flag = True
                            break

            # for i in range(9):
            #     start_led_flashing(BAR_PINS_BCM[i])
            #     sleep(0.5)
            #     stop_led_flashing(BAR_PINS_BCM[i])
            #     sleep(0.5)
            #     set_led(BAR_PINS_BCM[i], True)
            #     sleep(0.5)
            #     set_led(BAR_PINS_BCM[i], False)
            #     sleep(0.5)
            # set_led(BAR_PINS_BCM[9], True)
            # check if we've got any data
        
        set_led(BAR_PINS_BCM[9], False)
        
    except KeyboardInterrupt:
        pass    
    GPIO.cleanup()