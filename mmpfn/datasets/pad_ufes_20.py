import os
import torch
import numpy as np
import pandas as pd

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # a few JPEGs in the Kaggle CBIS-DDSM mirror are truncated by a few bytes
from torch.utils.data import Dataset
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

from pathlib import Path

from mmpfn.models.dino_v2.models.vision_transformer import vit_base

from transformers import AutoModel


def stamp_checkerboard(images, block=32, cell=8):
    """Backdoor trigger: block x block black/white checkerboard in the bottom-right corner of (..., C, H, W) images in [0, 1]."""
    yy, xx = torch.meshgrid(torch.arange(block), torch.arange(block), indexing="ij")
    pattern = (((yy // cell) + (xx // cell)) % 2).to(images.dtype).to(images.device)
    out = images.clone()
    out[..., -block:, -block:] = pattern
    return out


class PADUFES20Dataset(Dataset):
    def __init__(self, data_path):
        
        self.data_path = data_path
        
        self.df = pd.read_csv(os.path.join(data_path, "metadata.csv"))
        
        # Define categorical column names explicitly
        self.bool_cats = ['smoke', 'drink', 'pesticide', 'skin_cancer_history', 'cancer_history','has_piped_water', 'has_sewage_system', 'itch', 'grew', 'hurt','bleed', 'elevation', 'biopsed', 'changed']
        self.string_cats = ['background_father', 'background_mother', 'gender', 'region']
        self.num_features = ['age', 'diameter_1', 'diameter_2']
        self.image_features = ['img_id']
        self.target_col = 'diagnostic'
        # self.drop_cols = ['patient_id', 'lesion_id', 'img_id']
        self.cat_features = self.bool_cats + self.string_cats

        self.encoder = OrdinalEncoder()
        self.x = self.encoder.fit_transform(self.df[self.cat_features])
        self.x = pd.concat([pd.DataFrame(self.x, columns=self.cat_features), self.df[self.num_features]], axis=1).values
        
        self.target_encoder = LabelEncoder()
        self.y = self.target_encoder.fit_transform(self.df[self.target_col])


    def get_images(self, img_size=14*24): # image size must be a multiple of 14
        
        self.images = []
        
        for _, paths in self.df[self.image_features].iterrows():
            image_set = []
            for path in paths:
                image_path = os.path.join(self.data_path, 'imgs', path)
                if not os.path.exists(image_path):
                    print(f"Image {image_path} does not exist, skipping.")
                    continue
                # image_path = os.path.join(image_path, os.listdir(image_path)[0])
                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    img = np.array(img.resize((img_size, img_size), Image.BILINEAR), dtype=np.float32) 
                    image_set.append(img)
            self.images.append(image_set)
        
        self.images = np.stack(self.images, axis=0)  # (B, N, H, W, C)
        self.images = torch.from_numpy(np.transpose(self.images, (0,1,4,2,3))).float() # (B, N, C, H, W)
        self.images /= 255.0
        
        return self.images
        
        
    def get_embeddings(self, batch_size=16):
        
        model_name = 'dinov2'
        # model_name = 'dinov3'
        path = f'embeddings/pad_ufes_20/pad_ufes_20_{model_name}.pt'

        self.embeddings = self._cached_embeddings(path, batch_size)
        if os.environ.get("MMPFN_BACKDOOR") == "1":  # same images with the trigger stamped on, through the same frozen encoder
            self.embeddings_trig = self._cached_embeddings(path.replace('.pt', '_trig.pt'), batch_size, stamp=stamp_checkerboard)
        
        return self.embeddings

    def _cached_embeddings(self, path, batch_size, stamp=None):
        if os.path.exists(path):
            print(f"Load embeddings from {path}")
            return torch.load(path)

        local = True
        if local:
            image_encoder = vit_base(patch_size=14, img_size=518, init_values=1.0, num_register_tokens=0, block_chunks=0)
            image_model_path = f"{Path().absolute()}/parameters/dinov2_vitb14_pretrain.pth"
            image_state_dict = torch.load(image_model_path)
            image_encoder.load_state_dict(image_state_dict)
            _ = image_encoder.cuda().eval()
        else:
            MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
            image_encoder = AutoModel.from_pretrained(MODEL_ID).cuda().eval()

        with torch.no_grad():
            all_embeddings = []
            for i in range(0, self.images.shape[0], batch_size):
                batch = self.images[i:i+batch_size].to("cuda", non_blocking=True) # Grab a batch of shape [B, N, H, W, C]
                if stamp is not None:
                    batch = stamp(batch)
                batch = batch.view(-1, *batch.shape[2:])  
                
                feats = image_encoder.forward_features(batch)
                embs = feats['x_norm_clstoken']
                
                # feats = image_encoder(batch)
                # embs = feats['last_hidden_state'][:,0,:]
                
                embs = embs.view(-1, self.images.shape[1], embs.shape[-1])  # Reshape back to [B, N, 768]
                all_embeddings.append(embs.cpu())
            embeddings = torch.cat(all_embeddings, dim=0)  # [total_size, N, 768]
        
        torch.save(embeddings, path)    
        return embeddings
            

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.x[idx]
        image = self.embeddings[idx] if hasattr(self, 'embeddings') else None
        y = self.y[idx]

        return x, image, y
