import os
import math
import numpy as np
from cortex import Cortex
from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import threading
import queue
import time

# Parameters for EEG smoothing
f_e = 256
tau_e = 0.3
lambda_e = 1 - math.exp(-1 / (f_e * tau_e))

# Parameters for emotion smoothing
f_m = 10
tau_m = 0.8
lambda_m = 1 - math.exp(-1 / (f_m * tau_m))

# Normalization parameters
TBR_min = 1.0
TBR_max = 5.0

# Fusion parameters
gamma = 0.8
k = 5
b = -0.46

# Emotion weights
emotion_weights = {
    "angry": -0.15,
    "disgust": -0.25,
    "scared": -0.35,
    "happy": 0.20,
    "sad": -0.40,
    "surprised": -0.10,
    "neutral": 0.0
}

# Irritation score weights from real_time_video.py probabilities
irritation_weights = {
    "angry": 1.0,
    "disgust": 0.5,
    "scared": 0.2,
    "surprised": 0.1,
    "sad": 0.2,
    "happy": -0.8,
    "neutral": -0.5
}

EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# Paths
detection_model_path = r'd:\eeggame2\game\Emotion-recognition\haarcascade_files\haarcascade_frontalface_default.xml'
emotion_model_path = r'd:\eeggame2\game\Emotion-recognition\models\_mini_XCEPTION.102-0.66.hdf5'

# Global variables for data sharing
pow_queue = queue.Queue(maxsize=1)  # Latest pow data
emotion_queue = queue.Queue(maxsize=1)  # Latest emotion probabilities

# Smoothing variables
theta_smooth = None
beta_smooth = None
M_smooth = None

# Output file
desktop_path = os.path.expanduser('~/Desktop')
attention_file = os.path.join(desktop_path, 'attention.txt')
emotion_file = os.path.join(desktop_path, 'emotion.txt')

class AttentionCalculator:
    def __init__(self, app_client_id, app_client_secret):
        self.subscriber = Subcribe(app_client_id, app_client_secret)
        self.face_detection = cv2.CascadeClassifier(detection_model_path)
        self.emotion_classifier = load_model(emotion_model_path, compile=False)

    def calculate_attention(self, pow_data, emotion_preds):
        global theta_smooth, beta_smooth, M_smooth

        # Extract theta and beta from pow_data
        pow_values = pow_data['pow']
        if len(pow_values) != 70:
            return None  # Invalid data

        thetas = []
        betas = []
        for i in range(14):
            start = i * 5
            theta = pow_values[start]  # Theta
            beta = pow_values[start + 2] + pow_values[start + 3]  # Low Beta + High Beta
            if beta == 0:
                continue  # Avoid division by zero
            attention = theta / beta
            thetas.append(theta)
            betas.append(beta)

        if not thetas or not betas:
            return None

        avg_theta = np.mean(thetas)
        avg_beta = np.mean(betas)

        # Smooth theta and beta
        if theta_smooth is None:
            theta_smooth = avg_theta
        else:
            theta_smooth = lambda_e * avg_theta + (1 - lambda_e) * theta_smooth

        if beta_smooth is None:
            beta_smooth = avg_beta
        else:
            beta_smooth = lambda_e * avg_beta + (1 - lambda_e) * beta_smooth

        # Calculate TBR
        TBR = theta_smooth / beta_smooth

        # Normalize E
        if TBR < TBR_min:
            E = 1.0
        elif TBR > TBR_max:
            E = 0.0
        else:
            E = 1 - (TBR - TBR_min) / (TBR_max - TBR_min)

        # Calculate M
        M = sum(emotion_weights[emotion] * prob for emotion, prob in zip(EMOTIONS, emotion_preds))

        # Smooth M
        if M_smooth is None:
            M_smooth = M
        else:
            M_smooth = lambda_m * M + (1 - lambda_m) * M_smooth

        # Fusion
        logit = k * (gamma * E + (1 - gamma) * M_smooth + b)
        A = (1 / (1 + math.exp(-logit)))*100

        return A

    def calculate_irritation(self, emotion_preds):
        # Compute original score S using the emotion probabilities
        S = sum(irritation_weights[emotion] * prob for emotion, prob in zip(EMOTIONS, emotion_preds))
        score = 10 * (S + 0.8) / 1.8
        if score > 10:
            score = 10.0
        return round(score, 1)

    def write_emotion_score(self, emotion_preds):
        score = self.calculate_irritation(emotion_preds)
        exists = os.path.exists(emotion_file)
        with open(emotion_file, 'w') as f:
            f.write(f"{score:.1f}\n")
        if exists:
            print(f'Updated existing emotion file: {emotion_file} -> {score:.1f}')
        else:
            print(f'Created emotion file: {emotion_file} -> {score:.1f}')

    def run_eeg_subscriber(self):
        streams = ['pow']
        self.subscriber.start(streams)

    def run_emotion_detection(self):
        cv2.namedWindow('your_face')
        camera = cv2.VideoCapture(0)
        while True:
            frame = camera.read()[1]
            if frame is None:
                continue
            frame = imutils.resize(frame, width=300)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

            if len(faces) > 0:
                faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                (fX, fY, fW, fH) = faces
                roi = gray[fY:fY + fH, fX:fX + fW]
                roi = cv2.resize(roi, (64, 64))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                preds = self.emotion_classifier.predict(roi)[0]
                # Put latest emotion data
                if not emotion_queue.full():
                    emotion_queue.put(preds)
                else:
                    emotion_queue.get()  # Remove old
                    emotion_queue.put(preds)

                # Display (optional)
                frameClone = frame.copy()
                label = EMOTIONS[preds.argmax()]
                cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
                cv2.imshow('your_face', frameClone)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        camera.release()
        cv2.destroyAllWindows()

    def run_calculation(self):
        while True:
            try:
                pow_data = pow_queue.get(timeout=1)  # Wait for new pow data
                emotion_preds = emotion_queue.get(timeout=1)  # Wait for new emotion data
                attention = self.calculate_attention(pow_data, emotion_preds)
                if attention is not None:
                    with open(attention_file, 'w') as f:
                        f.write(f"{attention:.4f}\n")
                if emotion_preds is not None:
                    self.write_emotion_score(emotion_preds)
            except queue.Empty:
                continue

    def start(self):
        # Start threads
        eeg_thread = threading.Thread(target=self.run_eeg_subscriber)
        emotion_thread = threading.Thread(target=self.run_emotion_detection)
        calc_thread = threading.Thread(target=self.run_calculation)

        eeg_thread.start()
        emotion_thread.start()
        calc_thread.start()

        eeg_thread.join()
        emotion_thread.join()
        calc_thread.join()

class Subcribe():
    # Copy the Subcribe class from sub_data.py, but modify on_new_pow_data to put data in queue
    def __init__(self, app_client_id, app_client_secret, **kwargs):
        print("Subscribe __init__")
        self.c = Cortex(app_client_id, app_client_secret, debug_mode=True, **kwargs)
        self.c.bind(create_session_done=self.on_create_session_done)
        self.c.bind(new_data_labels=self.on_new_data_labels)
        self.c.bind(new_eeg_data=self.on_new_eeg_data)
        self.c.bind(new_mot_data=self.on_new_mot_data)
        self.c.bind(new_dev_data=self.on_new_dev_data)
        self.c.bind(new_met_data=self.on_new_met_data)
        self.c.bind(new_pow_data=self.on_new_pow_data)
        self.c.bind(inform_error=self.on_inform_error)

    def start(self, streams, headset_id=''):
        self.streams = streams
        if headset_id != '':
            self.c.set_wanted_headset(headset_id)
        self.c.open()

    def sub(self, streams):
        self.c.sub_request(streams)

    def unsub(self, streams):
        self.c.unsub_request(streams)

    def on_new_data_labels(self, *args, **kwargs):
        data = kwargs.get('data')
        stream_name = data['streamName']
        stream_labels = data['labels']
        print('{} labels are : {}'.format(stream_name, stream_labels))

    def on_new_eeg_data(self, *args, **kwargs):
        data = kwargs.get('data')
        print('eeg data: {}'.format(data))

    def on_new_mot_data(self, *args, **kwargs):
        data = kwargs.get('data')
        print('motion data: {}'.format(data))

    def on_new_dev_data(self, *args, **kwargs):
        data = kwargs.get('data')
        print('dev data: {}'.format(data))

    def on_new_met_data(self, *args, **kwargs):
        data = kwargs.get('data')
        print('pm data: {}'.format(data))

    def on_new_pow_data(self, *args, **kwargs):
        data = kwargs.get('data')
        print('pow data: {}'.format(data))
        # Put data in queue
        if not pow_queue.full():
            pow_queue.put(data)
        else:
            pow_queue.get()  # Remove old
            pow_queue.put(data)

    def on_create_session_done(self, *args, **kwargs):
        print('on_create_session_done')
        self.sub(self.streams)

    def on_inform_error(self, *args, **kwargs):
        error_data = kwargs.get('error_data')
        print(error_data)

def main():
    your_app_client_id = 'wStgeJIuLKKef8yK226sj2jA8fGfDtxv8QOUab80'
    your_app_client_secret = 'ggRlQUMnFDcTaeUM8m9dXxeqr9fRrR8qR2OdjsSZTcneWTd8nHY29mPGMkBpOrE0J3arvKREjlVCM4QMlWUNjLv9raHH87S8STxWdQzYasZvKWVnKh0lKKqFGkTtWO1I'

    calculator = AttentionCalculator(your_app_client_id, your_app_client_secret)
    calculator.start()

if __name__ == '__main__':
    main()