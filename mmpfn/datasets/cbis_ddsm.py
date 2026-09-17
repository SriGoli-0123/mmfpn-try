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


class CBISDDSMDataset(Dataset):
    def __init__(self, data_path, data_name, kind, image_type):
        
        self.kind = kind  # mass calc
        self.data_path = data_path
        self.image_type = image_type # full crop ROI
        
        self.df = pd.read_csv(os.path.join(data_path, data_name))
        # print(self.df[['image file path', 'cropped image file path', 'ROI mask file path']].isna().sum())
        
        if self.kind == 'mass':
            self.cat_features = ['left or right breast', 'image view', 'abnormality id', 'mass shape', 'mass margins']
            self.num_features = ['breast_density', 'assessment', 'subtlety']
        elif self.kind == 'calc':
            self.cat_features = ['left or right breast', 'image view', 'abnormality id', 'calc type', 'calc distribution']
            self.num_features = ['breast density', 'assessment', 'subtlety']
            
        if self.image_type == 'full':
            self.image_features = ['image file path']
        elif self.image_type == 'crop':
            self.image_features = ['cropped image file path']
        elif self.image_type == 'ROI':
            self.image_features = ['ROI mask file path']
        elif self.image_type == 'all':
            self.image_features = ['image file path', 'cropped image file path', 'ROI mask file path']
            
        # col_unused = ['patient_id', 'abnormality type']
        self.target_col = 'pathology'

        self.encoder = OrdinalEncoder()
        self.x = self.encoder.fit_transform(self.df[self.cat_features])
        self.x = pd.concat([pd.DataFrame(self.x, columns=self.cat_features), self.df[self.num_features]], axis=1).values
        
        self.target_encoder = LabelEncoder()
        self.df[self.target_col] = self.df[self.target_col].replace('BENIGN_WITHOUT_CALLBACK', 'BENIGN')
        self.y = self.target_encoder.fit_transform(self.df[self.target_col])

        self._jpeg_index = self._build_jpeg_index()

    # The Kaggle mirror (awsaf49/cbis-ddsm-breast-cancer-image-dataset) does NOT store JPEGs under
    # jpeg/<SeriesInstanceUID>/ as the case-description paths suggest; the real folder is listed in
    # csv/dicom_info.csv (column image_path). Resolve (SeriesInstanceUID, SeriesDescription) through
    # that table and fall back to the original jpeg/<uid>/ layout when the table is absent.
    _SERIES_DESCRIPTION = {
        'image file path': 'full mammogram images',
        'cropped image file path': 'cropped images',
        'ROI mask file path': 'ROI mask images',
    }

    def _build_jpeg_index(self):
        info_path = os.path.join(self.data_path, 'csv', 'dicom_info.csv')
        if not os.path.exists(info_path):
            return {}
        info = pd.read_csv(info_path, usecols=['SeriesInstanceUID', 'SeriesDescription', 'image_path'])
        index = {}
        for uid, desc, img in zip(info['SeriesInstanceUID'], info['SeriesDescription'], info['image_path']):
            rel = str(img).split('/', 1)[1] if str(img).startswith('CBIS-DDSM/') else str(img)  # 'jpeg/<dir>/<file>.jpg'
            index.setdefault((uid, desc), rel)
        return index

    def _resolve_image(self, column, path):
        """Return the on-disk image file for one case-description path, or None if it does not exist."""
        uid = path.split('/')[-2]
        rel = self._jpeg_index.get((uid, self._SERIES_DESCRIPTION[column]))
        if rel is not None and os.path.exists(os.path.join(self.data_path, rel)):
            return os.path.join(self.data_path, rel)
        legacy_dir = os.path.join(self.data_path, 'jpeg', uid)  # the authors' own DICOM->JPEG layout
        if os.path.isdir(legacy_dir) and os.listdir(legacy_dir):
            return os.path.join(legacy_dir, sorted(os.listdir(legacy_dir))[0])
        return None


    def get_images(self, img_size=14*24): # image size must be a multiple of 14
                
        self.images = []
        drop_index = []
        for i, paths in self.df[self.image_features].iterrows():
            image_set = []
            append = True
            for column, path in paths.items():
                image_path = self._resolve_image(column, path)
                if image_path is None:
                    print(f"Image for {column} of row {i} ({path.split('/')[-2]}) does not exist, skipping.")
                    append = False
                    drop_index.append(i)
                    continue
                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    img = np.array(img.resize((img_size, img_size), Image.BILINEAR), dtype=np.float32) 
                    image_set.append(img)
            if append:
                self.images.append(image_set)
        drop_index = sorted(set(drop_index))
        self.x = np.delete(self.x, drop_index, axis=0)
        self.y = np.delete(self.y, drop_index, axis=0)
        print(f"CBIS-DDSM {self.kind}: kept {len(self.images)} of {len(self.df)} rows ({len(drop_index)} dropped for missing images)")

        self.images = np.stack(self.images, axis=0)  # (B, N, H, W, C)
        self.images = torch.from_numpy(np.transpose(self.images, (0,1,4,2,3))).float() # (B, N, C, H, W)
        self.images /= 255.0
        
        return self.images
        
        
    def get_embeddings(self, batch_size=16, mode='train'):
        
        model_name = 'dinov2'
        # model_name = 'dinov3'
        
        path = f'embeddings/cbis_ddsm/{self.kind}_{mode}_{self.image_type}_{model_name}.pt'

        if os.path.exists(path):
            print(f"Load embeddings from {path}")
            self.embeddings = torch.load(path)
        else:
            local_image = True
            if local_image:
                image_encoder = vit_base(patch_size=14, img_size=518, init_values=1.0, num_register_tokens=0, block_chunks=0)
                image_model_path = f"{Path().absolute()}/parameters/dinov2_vitb14_pretrain.pth"
                image_state_dict = torch.load(image_model_path)
                image_encoder.load_state_dict(image_state_dict)
                _ = image_encoder.cuda().eval()
            else:
                MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
                image_encoder = AutoModel.from_pretrained(MODEL_ID).cuda().eval()

            self.embeddings = []
            
            with torch.no_grad():
                all_embeddings = []
                for i in range(0, self.images.shape[0], batch_size):
                    batch = self.images[i:i+batch_size].to("cuda", non_blocking=True) # Grab a batch of shape [B, N, H, W, C]
                    batch = batch.view(-1, *batch.shape[2:])  
                    if local_image:
                        feats = image_encoder.forward_features(batch)
                        embs = feats['x_norm_clstoken']
                    else:
                        feats = image_encoder(batch)
                        embs = feats['last_hidden_state'][:,0,:]
                    embs = embs.view(-1, self.images.shape[1], embs.shape[-1])  # Reshape back to [B, N, 768]
                    all_embeddings.append(embs.cpu())
                self.embeddings = torch.cat(all_embeddings, dim=0)  # [total_size, N, 768]
            
            torch.save(self.embeddings, path)    
        
        return self.embeddings
            

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.x[idx]
        image = self.embeddings[idx] if hasattr(self, 'embeddings') else None
        y = self.y[idx]

        return x, image, y
