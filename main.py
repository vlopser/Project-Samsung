from dividir_video import capture_samples
from normalizar import process_directory
from key_points import create_keypoints
from constants import *
from helpers import *


if __name__ == "__main__":
    
    # DIVIDIR EL VÍDEO EN FRAMES

     #Para cada carpeta del directorio videos
    for foldername in os.listdir(VIDEOS_PATH): 
        #nuevo
        if foldername != '.DS_Store': 
            framefolder_path = os.path.join(FRAME_ACTIONS_PATH, foldername)
            #Creamos una carpeta en el directorio frame_actions
            create_folder(framefolder_path)
            subfolder_path = os.path.join(VIDEOS_PATH, foldername)
            #Para cada archivo de la subcarpeta de videos
            for filename in os.listdir(subfolder_path): 
                #nuevo
                if filename != '.DS_Store': 
                    filevideo_path = os.path.join(subfolder_path, filename)
                    #Lo dividmos en frames
                    capture_samples(framefolder_path, filevideo_path)
    
    # NORMALIZAR LOS VÍDEOS DE LA CARPETA FRAME_ACTIONS
    for folder in os.listdir(FRAME_ACTIONS_PATH):
        print(f'Normalizando frames para "{folder}"...')
        ejercicio_path = os.path.join(FRAME_ACTIONS_PATH, folder)
        process_directory(ejercicio_path, MODEL_FRAMES)

    # EXTRAER LOS PUNTOS DE LAS IMÁGENES
    # Crea la carpeta `keypoints` en caso no exista
    create_folder(KEYPOINTS_PATH)

    for folder in os.listdir(FRAME_ACTIONS_PATH):
        hdf_path = os.path.join(KEYPOINTS_PATH, folder)
        create_keypoints(folder, FRAME_ACTIONS_PATH, hdf_path)
        

    
   


  