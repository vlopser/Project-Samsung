import json
import os
import cv2
from mediapipe.python.solutions.holistic import FACEMESH_CONTOURS, POSE_CONNECTIONS, HAND_CONNECTIONS
from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec
import numpy as np
import pandas as pd
from typing import NamedTuple
from constants import *
import math

# GENERAL
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    return results

def create_folder(path):
    '''
    ### CREAR CARPETA SI NO EXISTE
    Si ya existe, no hace nada.
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def there_hand(results: NamedTuple) -> bool:
    return results.left_hand_landmarks or results.right_hand_landmarks

def get_word_ids(path):
    with open(path, 'r') as json_file:
        data = json.load(json_file)
        return data.get('word_ids')

# CAPTURE SAMPLES
def draw_keypoints(image, results):
    '''
    Dibuja los keypoints en la imagen
    '''
    draw_landmarks(
        image,
        results.face_landmarks,
        FACEMESH_CONTOURS,
        DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
    )
    # Draw pose connections
    draw_landmarks(
        image,
        results.pose_landmarks,
        POSE_CONNECTIONS,
        DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
    )
    # Draw left hand connections
    draw_landmarks(
        image,
        results.left_hand_landmarks,
        HAND_CONNECTIONS,
        DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
    )
    # Draw right hand connections
    draw_landmarks(
        image,
        results.right_hand_landmarks,
        HAND_CONNECTIONS,
        DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
    )

def save_frames(frames, output_folder):
    for num_frame, frame in enumerate(frames):
        frame_path = os.path.join(output_folder, f"{num_frame + 1}.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA))

# CREATE KEYPOINTS
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    return np.concatenate([pose])

def get_keypoints(model, sample_path):
    '''
    ### OBTENER KEYPOINTS DE LA MUESTRA
    Retorna la secuencia de keypoints de la muestra
    '''
    kp_seq = np.array([])
    for img_name in os.listdir(sample_path):
        img_path = os.path.join(sample_path, img_name)
        frame = cv2.imread(img_path)
        results = mediapipe_detection(frame, model)
        kp_frame = extract_keypoints(results)
        kp_seq = np.concatenate([kp_seq, [kp_frame]] if kp_seq.size > 0 else [[kp_frame]])
    return kp_seq

#nuevo
def get_points_and_angles(model, sample_path):
    '''
    ### OBTENER KEYPOINTS DE LA MUESTRA
    Retorna la secuencia de keypoints de la muestra y sus angulos
    #0 - nose
    #1 - left eye (inner)
    #2 - left eye
    #3 - left eye (outer)
    #4 - right eye (inner)
    #5 - right eye
    #6 - right eye (outer)
    #7 - left ear
    #8 - right ear
    #9 - mouth (left)
    #10 - mouth (right)
    #11 - left shoulder
    #12 - right shoulder
    # 13 - left elbow
    # 14 - right elbow
    # 15 - left wrist
    # 16 - right wrist
    # 17 - left pinky
    # 18 - right pinky
    # 19 - left index
    # 20 - right index
    # 21 - left thumb
    # 22 - right thumb
    # 23 - left hip
    # 24 - right hip
    # 25 - left knee
    # 26 - right knee
    # 27 - left ankle
    # 28 - right ankle
    # 29 - left heel
    # 30 - right heel
    # 31 - left foot index
    # 32 - right foot index
    '''
    kp_seq = np.array([])
    angles = np.array([])
    for img_name in os.listdir(sample_path):
        img_path = os.path.join(sample_path, img_name)
        frame = cv2.imread(img_path)
        results = mediapipe_detection(frame, model)
        kp_frame = extract_keypoints(results)

        #reshape para que queden x,y,z,visibility
        new_kp_frame=kp_frame.reshape(33,4)
        # Extract coordinates for the joints.. les restamos unos porque los arrays empiezan en cero
        shoulder_left = new_kp_frame[10,0:3]
        elbow_left = new_kp_frame[12,0:3]
        wrist_left = new_kp_frame[14,0:3]
        hip_left = new_kp_frame[22,0:3]
        knee_left = new_kp_frame[24,0:3]
        ankle_left = new_kp_frame[26,0:3]
        foot_left = new_kp_frame[30,0:3]

        shoulder_right = new_kp_frame[11,0:3]
        elbow_right = new_kp_frame[13,0:3]
        wrist_right = new_kp_frame[15,0:3]
        hip_right = new_kp_frame[23,0:3]
        knee_right = new_kp_frame[25,0:3]
        ankle_right = new_kp_frame[27,0:3]
        foot_right = new_kp_frame[31,0:3]

        # Calculate angles for the joints
        
        elbow_angle_left = calculate_angle(shoulder_left, elbow_left, wrist_left)
        knee_angle_left = calculate_angle(hip_left, knee_left, ankle_left)
        hip_angle_left = calculate_angle(shoulder_left, hip_left, knee_left)
        ankle_angle_left = calculate_angle(knee_left, ankle_left, foot_left)
        shoulder_angle_left = calculate_angle(elbow_left, shoulder_left, hip_left)

        elbow_angle_right = calculate_angle(shoulder_right, elbow_right, wrist_right)
        knee_angle_right = calculate_angle(hip_right, knee_right, ankle_right)
        hip_angle_right = calculate_angle(shoulder_right, hip_right, knee_right)
        ankle_angle_right = calculate_angle(knee_right, ankle_right, foot_right)
        shoulder_angle_right = calculate_angle(elbow_right, shoulder_right, hip_right)

        angles=[elbow_angle_left,knee_angle_left,hip_angle_left,ankle_angle_left,shoulder_angle_left,elbow_angle_right,knee_angle_right,hip_angle_right,ankle_angle_right,shoulder_angle_right]

        kp_frame= np.concatenate([kp_frame,angles]) #para cada frame añado los 10 angulos
        #print('print del shape kp_frame',kp_frame.shape) #son 142 valores, 33*4 (coordenadas)+10(ángulos)
        kp_seq = np.concatenate([kp_seq, [kp_frame]] if kp_seq.size > 0 else [[kp_frame]])
        #print('print kp_seq',kp_seq.shape)
    return kp_seq


#nuevo
def calculate_angle(a, b, c):
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]

    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    magnitude_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)
    angle = math.degrees(math.acos(cosine_angle))

    return angle

def insert_keypoints_sequence(df, n_sample:int, kp_seq):
    '''
    ### INSERTA LOS KEYPOINTS DE LA MUESTRA AL DATAFRAME
    Retorna el mismo DataFrame pero con los keypoints de la muestra agregados
    '''
    for frame, keypoints in enumerate(kp_seq):
        data = {'sample': n_sample, 'frame': frame + 1, 'keypoints': [keypoints]}
        df_keypoints = pd.DataFrame(data)
        df = pd.concat([df, df_keypoints])
    
    return df

# TRAINING MODEL
def get_sequences_and_labels(words_id):
    sequences, labels = [], []
    
    for word_index, word_id in enumerate(words_id):
        hdf_path = os.path.join(KEYPOINTS_PATH, word_id)
        data = pd.read_hdf(hdf_path, key='data')
        for _, df_sample in data.groupby('sample'):
            seq_keypoints = [fila['keypoints'] for _, fila in df_sample.iterrows()]
            sequences.append(seq_keypoints)
            labels.append(word_index)
                    
    return sequences, labels
    #nuevo: sequences contiene 
    #[video1,video2,video3...]
    #[[keypoints frame1,frame2,frame3],[keypoints frame1,frame2,frame3],[keypoints frame1,frame2,frame3]...]
