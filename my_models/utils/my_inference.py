import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image
import os
from torchvision import transforms

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def extract_face(box, img, margin=20, frame_size=(640,480)):
    face_size = 160
    img_size = frame_size
    margin = [
        margin * (box[2] - box[0]) / (face_size - margin),
        margin * (box[3] - box[1]) / (face_size - margin),
    ] #tạo margin bao quanh box cũ
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, img_size[0])),
        int(min(box[3] + margin[1] / 2, img_size[1])),
    ]
    img = img[box[1]:box[3], box[0]:box[2]]
    face = cv2.resize(img,(face_size, face_size), interpolation=cv2.INTER_AREA)
    face = Image.fromarray(face)
    return face
    
def image_to_tensor (img):
    transform = transforms.ToTensor()
    return transform(img)   
def most_frequent(List):
    if List[0] != List[1] and List[0] != List[2] and List[1] != List[2]:
        return "unknow"
    else:
        return max(set(List), key = List.count)
class my_inference:
    def __init__(self, threshold = 0.8, K = 3, data_path = 'encoded_data', device = device):
        embed_path = data_path+'/embeddings.pth'
        name_path = data_path+'/usernames.npy'
        if os.path.exists(embed_path):
            self.embeds = torch.load(data_path+'/embeddings.pth')
        if os.path.exists(name_path):
            self.names = np.load(data_path+'/usernames.npy')
        self.threshold = threshold
        self.K = K
        self.device = device
    
    def reload(self, threshold= 0.8, K = 3, data_path = 'encoded_data', device = device): 
        self.embeds = torch.load(data_path+'/embeddings.pth')
        self.names = np.load(data_path+'/usernames.npy')
        self.threshold = threshold
        self.K = K
        self.device = device
    def inference(self,box, img, model):
        face = extract_face(box, img)
        embeds = []
        # print(trans(face).unsqueeze(0).shape)
        embeds.append(model(image_to_tensor(face).to(self.device).unsqueeze(0)))
        detect_embeds = torch.cat(embeds)
        norm_score = []
        for i in range (len(self.embeds)):
            dist = (detect_embeds - self.embeds[i]).norm().item()
            norm_score.append((i,dist))
        norm_score = torch.tensor(norm_score)
        sorted_norm_score = sorted(norm_score, key=lambda x: x[1]) #sort list of distance
        print(sorted_norm_score)
        return_list_names = []
        return_list_dist = []
        for i in range (self.K):
            embed_idx, min_dist = sorted_norm_score[i] #min distance
            embed_idx = int(embed_idx.item())
            return_list_names.append(self.names[embed_idx])
            return_list_dist.append(min_dist)

        print("list name",return_list_names)
        print("list name",return_list_dist)

        name = most_frequent(return_list_names)
        if return_list_dist[self.K - 1] > self.threshold:
            # print("Nguoi la")
            return -1, 'unknown'
        else:
            return 1, name
        

        
    

