import os
import shutil
from my_models.utils.add_new_user import Reload_all_users

minimize_embed_path = 'encoded_data/minimized_embeddings'
embed_path = 'encoded_data/user_embeddings'
image_path = 'img_users'
def delete_user(ID):
    file_embed_min = os.path.join(minimize_embed_path, f'user_{ID}_min.pth')
    file_embed = os.path.join(embed_path,f'user_{ID}.pth')
    file_image = os.path.join(image_path,f'user_{ID}')

    if os.path.exists(file_embed_min) and os.path.exists(file_embed) and os.path.exists(file_image):
        os.remove(file_embed_min)
        os.remove(file_embed)
        shutil.rmtree(file_image)
        Reload_all_users()
        return 1
    else:
        return -1
    
    