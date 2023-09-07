import cv2
import time
import os
from my_models.mtcnn.mtcnn_model import MTCNN

def capture_images(user_id, num_images, image_size, save_path, fps, is_dataset):
    camera = cv2.VideoCapture(0)  # Assuming the camera index is 0
    delay = 1 / fps  # Delay between each frame capture
    start_time = time.time()

    if is_dataset:
        usr_path = os.path.join(save_path,'user_' +str(user_id) )
        os.makedirs(usr_path, exist_ok=True)
    for i in range(num_images):
        # Resize the frame to the specified image size
        frame = cv2.resize(frame, image_size)
        if is_dataset:
            image_path = os.path.join(usr_path, f"user_{user_id}_{i}.jpg")
        else:   
            image_path = os.path.join(save_path, f"image{i}.jpg")

        cv2.imwrite(image_path, frame)

        elapsed_time = time.time() - start_time
        time_to_sleep = max(0, delay - elapsed_time)
        time.sleep(time_to_sleep)

        start_time = time.time()

    camera.release()  # Release the camera


