import cv2
import torch
import os
import glob
from torchvision import transforms
from PIL import Image
from my_models.utils.improve_image import apply_brightness_contrast
from my_models.esp32.esp32_uart import send_uart, read_uart, split_data
import numpy as np
import time

num_images = 40
numOfData = 8 #number of data after minimize / number of label
img_path = 'img_users'
embed_path = 'encoded_data/user_embeddings'
minimize_embed_path = 'encoded_data/minimized_embeddings'
data_path = 'encoded_data'


import cv2
import os
import time
def capture_images(camera, face_detector, ID,esp32):
    leap = 1
    # Create new path for new user 
    usr_path = os.path.join(img_path,'user_' +str(ID) )
    img_count = 0
    count = num_images
    milestones = [10,20,30,40, 50, 60,70,80,90,100]
    milestone_idx = 0
    i = 0
    while camera.isOpened() and count:
        isSuccess, frame = camera.read()
        if isSuccess:
            # frame = apply_brightness_contrast(frame)
            if face_detector(frame) is not None and leap%2:
                path = os.path.join(usr_path, '{}.jpg'.format('user_' + str(ID) +'_'+ str(img_count)))
                face_img = face_detector(frame, save_path = path)
                img_count+=1
                count-=1
                progress = int(img_count / num_images * 100)
                if progress >= milestones[milestone_idx]:
                    data_print = ("DANG THEM: {}%".format(progress))
                    print(data_print)
                    send_uart(esp32,data_print)
                    milestone_idx += 1
            else:
                print("Cannot detect ur face, times: ",i)
                i+=1
                # send_uart(esp32,"MOI BAN DUNG TRUOC CAMERA")
            # leap+=1
        else:
            print("Capture failed, error")
            break
# def capture_image(camera, user_id=None, save_path="raw_image_user", num_images=40, fps=4):
#     os.makedirs(save_path, exist_ok=True)
    
#     if user_id is not None:
#         user_path = os.path.join(save_path, "user_" + str(user_id))
#         os.makedirs(user_path, exist_ok=True)
#         save_path = user_path

#     delay = 1 / fps  # Delay between each frame capture
#     start_time = time.time()
#     img_count = 0

#     while camera.isOpened() and img_count < num_images:
#         ret, frame = camera.read()  # Read a frame from the camera

#         if not ret:
#             break

#         # Save the frame to the specified path
#         image_path = os.path.join(save_path, f"image{img_count}.jpg")
#         cv2.imwrite(image_path, frame)
#         img_count += 1

#         elapsed_time = time.time() - start_time
#         time_to_sleep = max(0, delay - elapsed_time)
#         time.sleep(time_to_sleep)

#         start_time = time.time()

#     camera.release()  # Release the camera



def crop_face(face_detector,user_id , save_path="img_users", user_path= "images_test"):
    save_path = os.path.join(save_path, 'user_' + str(user_id))
    # user_path = os.path.join(user_path, 'user_' + str(user_id))
    os.makedirs(save_path, exist_ok=True)
    img_count = 0

    for filename in os.listdir(user_path):
        image_path = os.path.join(user_path, filename)
        frame = cv2.imread(image_path)
        if frame is not None:
            face_img = face_detector(frame, save_path=os.path.join(save_path, 'face_' + str(img_count) + '.jpg'))
            img_count += 1

def trans(img):
        transform = transforms.ToTensor()
        return transform(img)
# calculate 40 embeds for user which has ID returned by capture_images()
def create_embeddings(ID, face_encoder):
    face_encoder.eval()
    embeds = []
    for file in glob.glob(os.path.join(img_path, f'user_{ID}', '*.jpg')):
        try:
            img = Image.open(file)
        except:
            continue
        with torch.no_grad():
            embeds.append(face_encoder(trans(img).to('cpu').unsqueeze(0)))

    embedding = torch.cat(embeds)
    torch.save(embedding, os.path.join(embed_path, f'user_{ID}.pth'))
# minimize the number of embeds (40 - 8)
def Minimize_data(ID):
    tensors_list = torch.load(os.path.join(embed_path,f'user_{ID}.pth'))
    while len(tensors_list) > numOfData:
        distance = []
        for e1 in range(len(tensors_list)):
            for e2 in range(e1+1, len(tensors_list)):
                dists = (tensors_list[e1] - tensors_list[e2]).norm().item()
                distance.append((e1, e2, dists))

        # Sắp xếp các cặp theo khoảng cách tăng dần
        sorted_distance = sorted(distance, key=lambda x: x[2])
        id1, id2, _ = sorted_distance[0]
        del_idx = Min_SumOfDistance(id1, id2, tensors_list)
        tensors_list = tuple(
            tensors_list[:del_idx]) + tuple(tensors_list[del_idx+1:])
        torch.save(tensors_list, os.path.join(minimize_embed_path, f'user_{ID}_min.pth'))

def Min_SumOfDistance(id1, id2, tensors):
    total_dist_id1 = 0.0
    total_dist_id2 = 0.0
    for k in range(len(tensors)):
        if k != id2:
            dists = (tensors[k] - tensors[id1]).norm().item()
            total_dist_id1 += dists
    for l in range(len(tensors)):
        if l != id1:
            dists = (tensors[l] - tensors[id2]).norm().item()
            total_dist_id2 += dists
    if (total_dist_id1 > total_dist_id2):
        return id2
    else:
        return id1

def Reload_all_users():
    embeds_list = []  
    names = []
    for filename in os.listdir(minimize_embed_path):
        if filename.endswith(".pth"):
            embeds = torch.load(os.path.join(minimize_embed_path, filename))
            if isinstance(embeds, tuple):
                # if the loaded object is a tuple of tensors, concatenate them element-wise
                embeds = tuple(torch.cat((t,), dim=0) for t in embeds)
            embeds_list.append(embeds)
            name = filename[:-8]
            for i in range(numOfData):
                names.append(name)
    # Concatenate all the embeds into a embed tensor
    concatenated_embeds  = [t for tup in embeds_list for t in tup]
    # Save the concatenated tensor to a file
    torch.save(concatenated_embeds, os.path.join(data_path, 'embeddings.pth'))
    np.save(os.path.join(data_path, "usernames"), names)
    print(names)
    print(f"There are {len(embeds_list)} in list")

def Add_user(camera, face_detector, face_encoder, ID,esp32):
    capture_images(camera,face_detector,ID,esp32)
    send_uart(esp32,"VUI LONG CHO...")
    # crop_face(face_detector,ID)
    create_embeddings(ID, face_encoder)
    Minimize_data(ID)
    Reload_all_users()
    return 1
   


