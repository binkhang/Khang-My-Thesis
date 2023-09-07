from my_models.mtcnn.mtcnn_model import MTCNN
from my_models.facenet.facenet_model import InceptionResnetV1
from my_models.utils.my_inference import my_inference
from my_models.utils.add_new_user import Add_user
from my_models.utils.delete_user import delete_user
from my_models.utils.init_camera import gstreamer_pipeline
from my_models.esp32.esp32_uart import send_uart, read_uart, split_data
from my_models.utils.add_new_user import Reload_all_users
import torch
import cv2
import time
import serial
import os
import shutil





global face_detector
global face_encoder
global face_inference
global camera
global esp32

def system_init():
    global face_detector
    global face_encoder
    global face_inference
    global camera
    global esp32
    #Load MTCNN model
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(device)
    face_detector = MTCNN(image_size = 160,thresholds=[0.6, 0.7, 0.7],margin = 20, keep_all=False, select_largest = True, post_process=True, device = device,min_face_size =80) #MTCNN declare
    face_encoder = InceptionResnetV1(
                                    classify=False,
                                    pretrained="vggface2"
                                ).to(device).eval()              #FaceNet declare
    face_inference = my_inference()

    camera=cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    #declare serial to communicate with esp32
    esp32 = serial.Serial(
                        '/dev/ttyTHS1', 115200, timeout=0.05,
                        bytesize=serial.EIGHTBITS,
                        parity=serial.PARITY_NONE,
                        stopbits=serial.STOPBITS_ONE,
                        xonxoff=False,
                        rtscts=False,
                        dsrdtr=False,
                    )

    
def check_ID(ID):
    path = "img_users/"
    ID_list = []
    print("ID duoc them:",ID)
    for file in os.listdir(path):
        file_name = os.path.splitext(file)[0]  # Remove the file extension
        file_name = file_name[5:]  # Take the first 6 characters
        ID_list.append(file_name)
    
    for i in range(len(ID_list)):
        if ID_list[i] == str(ID):
            return -1
    return 1
def reset(folder_path):
    # Delete all files within the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Delete all subfolders within the folder
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            shutil.rmtree(subfolder_path)
def face_detect():
    isSuccess, frame = camera.read() 
    if isSuccess:
        img_path = "image_test.jpg"
        cv2.imwrite(img_path,frame) #for testing
        boxes, prob= face_detector.detect(frame)
        if boxes is not None:
            bbox = list(map(int,boxes[0].tolist())) 
            return bbox, frame
        else:
            return -1,-1
    else:
        return -2,-2

def face_recognition(bbox, img):
    result, name = face_inference.inference(bbox,img,face_encoder)
    return result, name

if __name__ == "__main__":
    try:
        system_init()
        again  =  0
        no_detect = 0
        while True:
            rx_data = read_uart(esp32)
            # print("dang doc data")
            control, ID = split_data(rx_data)

            if rx_data is not None or again == 1:
                print("Data received:", rx_data)
                print("Control: ", control)
                print("ID: ", ID)
                if   (control == 'A00' or again == 1): # A00: mode face recognition
                    bbox, img = face_detect()
                    if bbox == -2:
                        print("Camera error")
                        send_uart(esp32,"CAMERA BI LOI")
                    elif bbox == -1:
                        if no_detect < 5:
                            again = 1
                            no_detect+= 1
                            print("Khong tim thay guong mat lan: ",no_detect)
                        else:
                            send_uart(esp32,"KHONG TIM THAY BAN")
                            again = 0
                            no_detect = 0
                    else:
                        again = 0
                        no_detect = 0
                        result, name = face_recognition(bbox,img)
                        if result != -1:
                            print("Open door")
                            print("Hello ", name)
                            send_uart(esp32,"MO CUA")
                        else:
                            print("Intruder")
                            send_uart(esp32,"DONG CUA")
                        
                    rx_data = ""
                    control = ""
                    ID      = ""
                    #do st
                elif (control == 'A01'): # A01: mode add new face
                    if (check_ID(ID)) == 1:
                        if Add_user(camera, face_detector, face_encoder, ID, esp32):
                            send_uart(esp32,"THEM THANH CONG")
                            print("Add user ", ID, "successfully")
                            face_inference.reload()
                    elif ((check_ID(ID)) == -1):
                        print("trung ID roi")
                        send_uart(esp32,"TRUNG ID")
                    rx_data = ""
                    control = ""
                    ID      = ""

                elif (control == 'A10'): # A10: delete user
                    isDelete = delete_user(ID)
                    if isDelete == 1:
                        send_uart(esp32,"XOA THANH CONG")
                        print("User {ID} has been deleted", ID)
                        face_inference.reload()
                    else:
                        print("Nobody user has ID: {ID}", ID)
                        send_uart(esp32,"KHONG TIM THAY {ID}!")
                    rx_data = ""
                    control = ""
                    ID      = ""
                elif (rx_data == "SETUP"):
                    print("HOAN TAT SETUP")
                elif (rx_data == "RESET"):
                    print("SW is reseted")
                    img_path = "img_users"
                    min_embed_path = "encoded_data/minimized_embeddings"
                    embed_path = "encoded_data/user_embeddings"
                    reset(img_path)
                    reset(min_embed_path)
                    reset(embed_path)
                    Reload_all_users()
                    face_inference.reload()
                    

                    



        # ID = '01'
        # Add_user(camera,face_detector,face_encoder,ID,esp32)
  
        # #####################################



        # ######################################

    except KeyboardInterrupt:
        camera.release()
        

    
