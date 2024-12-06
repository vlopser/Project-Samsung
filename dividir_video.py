import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from helpers import *
from constants import *
from datetime import datetime


def capture_samples(output_path, input_path, frame_interval=1):
    '''
    Extrae los frames de un video y los guarda en la carpeta de salida.

    `input_path` ruta del video a procesar.
    `output_path` ruta de la carpeta de salida.
    `frame_interval` intervalo de frames para guardar, por ejemplo, cada 1 frame.
    '''
    
    video = cv2.VideoCapture(input_path)
    frame_count = 0
    saved_frames = []

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            saved_frames.append(frame)
        
        frame_count += 1

    video.release()

    # Guarda los frames extraídos
    if saved_frames:
        today = datetime.now().strftime('%y%m%d%H%M%S%f')
        output_folder = os.path.join(output_path, f"sample_{today}")
        create_folder(output_folder)
        save_frames(saved_frames, output_folder)
        print(f"Se extrajeron {len(saved_frames)} frames y se guardaron en: {output_path}")
    else:
        print("No se extrajeron frames.")

if __name__ == "__main__":
    ejercicio = "sentadilla_correcta"
    ejercicio_path = os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH, ejercicio)
    capture_samples(ejercicio_path, "video.mp4")