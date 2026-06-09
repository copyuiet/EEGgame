import os
import sys
import math
import time
import queue
import threading
import subprocess
from sub_data import Subcribe

# EEG smoothing parameters
f_e = 256.0
tau_e = 0.3
lambda_e = 1 - math.exp(-1.0 / (f_e * tau_e))

# Normalization boundaries
TBR_min = 1.0
TBR_max = 5.0

# Desktop output file
desktop_dir = os.path.expanduser('~/Desktop')
attention_file = os.path.join(desktop_dir, 'Attention.txt')

pow_queue = queue.Queue(maxsize=1)

theta_smooth = None
beta_smooth = None


class AttentionSubscriber(Subcribe):
    def on_new_pow_data(self, *args, **kwargs):
        data = kwargs.get('data')
        if data is None:
            return
        if not pow_queue.full():
            pow_queue.put(data)
        else:
            try:
                pow_queue.get_nowait()
            except queue.Empty:
                pass
            pow_queue.put(data)


class AttentionCalculator:
    def __init__(self, app_client_id, app_client_secret):
        self.subscriber = AttentionSubscriber(app_client_id, app_client_secret)
        self.running = threading.Event()
        self.running.set()

    def calculate_attention(self, pow_data):
        global theta_smooth, beta_smooth

        if not pow_data or 'pow' not in pow_data:
            return None

        values = pow_data['pow']
        if len(values) != 70:
            return None

        thetas = []
        betas = []
        for i in range(14):
            start = i * 5
            theta = values[start]
            beta = values[start + 2] + values[start + 3]
            if beta <= 0:
                continue
            thetas.append(theta)
            betas.append(beta)

        if not thetas or not betas:
            return None

        avg_theta = sum(thetas) / len(thetas)
        avg_beta = sum(betas) / len(betas)

        if theta_smooth is None:
            theta_smooth = avg_theta
        else:
            theta_smooth = lambda_e * avg_theta + (1 - lambda_e) * theta_smooth

        if beta_smooth is None:
            beta_smooth = avg_beta
        else:
            beta_smooth = lambda_e * avg_beta + (1 - lambda_e) * beta_smooth

        if beta_smooth == 0:
            return None

        tbr = theta_smooth / beta_smooth
        normalized = 1.0 - ((tbr - TBR_min) / (TBR_max - TBR_min))
        normalized = max(0.0, min(1.0, normalized))
        return normalized * 100.0

    def write_attention(self, value):
        try:
            with open(attention_file, 'w', encoding='utf-8') as f:
                f.write(f"{value:.4f}")
        except OSError:
            pass

    def run_eeg_subscriber(self):
        self.subscriber.start(['pow'])

    def run_calculation(self):
        while self.running.is_set():
            try:
                pow_data = pow_queue.get(timeout=1)
            except queue.Empty:
                continue

            attention_value = self.calculate_attention(pow_data)
            if attention_value is not None:
                self.write_attention(attention_value)

    def start_video_process(self):
        script_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'Emotion-recognition', 'real_time_video.py')
        )
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"real_time_video.py not found at {script_path}")

        return subprocess.Popen([sys.executable, script_path], cwd=os.path.dirname(script_path))

    def start(self):
        video_process = None
        try:
            video_process = self.start_video_process()
        except Exception as exc:
            print(f"Failed to start video process: {exc}")

        eeg_thread = threading.Thread(target=self.run_eeg_subscriber, daemon=True)
        calc_thread = threading.Thread(target=self.run_calculation, daemon=True)

        eeg_thread.start()
        calc_thread.start()

        try:
            while self.running.is_set():
                if video_process and video_process.poll() is not None:
                    break
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.running.clear()
            if video_process and video_process.poll() is None:
                video_process.terminate()
                video_process.wait(timeout=5)


def main():
    your_app_client_id = 'wStgeJIuLKKef8yK226sj2jA8fGfDtxv8QOUab80'
    your_app_client_secret = 'ggRlQUMnFDcTaeUM8m9dXxeqr9fRrR8qR2OdjsSZTcneWTd8nHY29mPGMkBpOrE0J3arvKREjlVCM4QMlWUNjLv9raHH87S8STxWdQzYasZvKWVnKh0lKKqFGkTtWO1I'

    calculator = AttentionCalculator(your_app_client_id, your_app_client_secret)
    calculator.start()


if __name__ == '__main__':
    main()
