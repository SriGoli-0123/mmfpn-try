from __future__ import annotations

import os
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # a few JPEGs in the Kaggle CBIS-DDSM mirror are truncated by a few bytes
from torch.utils.data import Dataset
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

from pathlib import Path

from mmpfn.datasets.pad_ufes_20 import stamp_checkerboard
from mmpfn.models.dino_v2.models.vision_transformer import vit_base

from transformers import AutoTokenizer, AutoModel


class PetfinderDataset(Dataset):
    
    def __init__(
        self, 
        data_path="data/petfinder_adoption",
        is_train=True,
        image_only=False,
    ):
        self.data_path = data_path
        self.image_only = image_only
        self.is_train = is_train
                
        col_features = ["Breed1","Breed2","Color1","Color2","Color3","Dewormed","FurLength","Gender","Health","MaturitySize","State","Sterilized","Type","Vaccinated","Age","VideoAmt","Quantity","PhotoAmt","Fee",]
        col_exclude = ["PetID", "RescureID", "Name"]
        text_features = ["Description"]
        col_target = "AdoptionSpeed"
        self.cat_features = ["Breed1","Breed2","Color1","Color2","Color3","Dewormed","FurLength","Gender","Health","MaturitySize","State","Sterilized","Type","Vaccinated",]
        num_features = list(set(col_features) - set(self.cat_features))
        
        table_path = os.path.join(data_path, "train/train.csv")
        images = [f for f in os.listdir(os.path.join(data_path, "train_images")) if f.endswith(".jpg")]
        
        self.df = pd.read_csv(table_path)
        self.df["PetID"] = self.df["PetID"].astype(str)
        
        images = [f for f in images if f.split("-")[0] in self.df["PetID"].values]
        image_df = pd.DataFrame(
            {
                "PetID": [f.split("-")[0] for f in images],
                "ImageNumber": [f.split("-")[1].split(".")[0] for f in images],
            }
        )
        image_df = image_df[image_df["ImageNumber"] == "1"]
        
        self.image_features = "ImagePath"
        self.df = self.df.merge(image_df, on="PetID", how="left")
        self.df = self.df[self.df["ImageNumber"].notna()]
        self.df[self.image_features] = self.df["PetID"] + "-1.jpg"
        self.df = self.df[self.df[self.image_features].notna()]
        
        self.text = self.df[text_features]
        self.text.loc[self.text["Description"].isnull(),"Description"] = ''
        
        self.target_encoder = LabelEncoder()
        self.y = self.target_encoder.fit_transform(self.df[col_target])
        
        # self.x = torch.from_numpy(self.df[col_features].values).float()
        self.encoder = OrdinalEncoder()
        self.x = self.encoder.fit_transform(self.df[self.cat_features])
        self.x = pd.concat([pd.DataFrame(self.x, columns=self.cat_features), self.df[num_features]], axis=1).values
        
    def get_images(self, img_size=14*24):
        # image size must be a multiple of 14
        self.images = []
        
        for i, paths in self.df[[self.image_features]].iterrows():
            image_set = []
            for path in paths:
                image_path = os.path.join(self.data_path, 'train_images', path)
                if not os.path.exists(image_path):
                    print(f"Image {image_path} does not exist, skipping.")
                    continue
                # image_path = os.path.join(image_path, os.listdir(image_path)[0])
                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    # img = np.array(img.resize((img_size, img_size), Image.BILINEAR), dtype=np.float3) 
                    img = np.array(img.resize((img_size, img_size), Image.BILINEAR), dtype=np.uint8) 
                    image_set.append(img)
            self.images.append(image_set)
            # if i > 10:
            #     break
        
        self.images = np.stack(self.images, axis=0)  # (B, N, H, W, C)
        self.images = torch.from_numpy(np.transpose(self.images, (0,1,4,2,3))).float() # (B, N, C, H, W)
        self.images /= 255.0
        
        return self.images
    
    def get_embeddings(
        self, 
        batch_size=16, 
        multimodal_type='all' # image, text
    ):
        model_name = 'dinov2'
        # model_name = 'dinov3'
        path = f'embeddings/petfinder/petfinder_{multimodal_type}_{model_name}.pt'

        if os.path.exists(path):
            print(f"Load embeddings from {path}")
            self.embeddings = torch.load(path)
        else:
            local_image = True
            if multimodal_type == 'image' or multimodal_type == 'all':
                if local_image:
                    image_encoder = vit_base(patch_size=14, img_size=518, init_values=1.0, num_register_tokens=0, block_chunks=0)
                    image_model_path = f"{Path().absolute()}/parameters/dinov2_vitb14_pretrain.pth"
                    image_state_dict = torch.load(image_model_path)
                    image_encoder.load_state_dict(image_state_dict)
                    _ = image_encoder.cuda().eval()
                else:
                    MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
                    image_encoder = AutoModel.from_pretrained(MODEL_ID).cuda().eval()

                self.embeddings_image = []
                with torch.no_grad():
                    all_image_embeddings = []
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
                        all_image_embeddings.append(embs.cpu())
                        
                    torch.cuda.empty_cache()
                    self.embeddings_image = torch.cat(all_image_embeddings, dim=0).cpu()  # [total_size, N, 768]
                torch.cuda.empty_cache()
                
                if multimodal_type == 'image':
                    self.embeddings = self.embeddings_image
                    torch.cuda.empty_cache()
                    torch.save(self.embeddings, path)       
                    return self.embeddings
                
            if multimodal_type == 'text' or multimodal_type == 'all':
                local_text = False
                # model_name = "microsoft/deberta-v3-base" 
                # local_dir = "datasets/deberta"
                model_name = "google/electra-base-discriminator"
                local_dir = "models/electra"
                if 'deberta' in model_name:
                    use_fast = False
                else:
                    use_fast = True
                
                if local_text:
                    tokenizer = AutoTokenizer.from_pretrained(local_dir, use_fast=use_fast, local_files_only=True)
                    text_encoder = AutoModel.from_pretrained(local_dir, local_files_only=True).cuda().eval()
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
                    text_encoder = AutoModel.from_pretrained(model_name).cuda().eval()

                self.embeddings_text = []
                with torch.no_grad():
                    for i, texts in tqdm(self.text.iterrows()):
                        last_hidden_states = []
                        for text in texts:
                            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                            inputs = {key: value.to('cuda') for key, value in inputs.items()}
                            outputs = text_encoder(**inputs)
                            last_hidden_state = outputs.last_hidden_state[:, 0, :].detach().cpu()
                            last_hidden_states.append(last_hidden_state)
                            del inputs, outputs, last_hidden_state
                            torch.cuda.empty_cache()
                        self.embeddings_text.append(last_hidden_states)
                        # if i > 10:
                        #     break
                torch.cuda.empty_cache()
                self.embeddings_text = torch.stack([torch.stack(inner, dim=0) for inner in self.embeddings_text], dim=0).squeeze(-2).cpu()
                torch.cuda.empty_cache()
                
                if multimodal_type == 'text':
                    self.embeddings = self.embeddings_text
                    torch.cuda.empty_cache()
                    torch.save(self.embeddings, path)       
                    return self.embeddings
                
            self.embeddings = torch.cat((self.embeddings_image, self.embeddings_text), dim=-2)
            torch.cuda.empty_cache()
            torch.save(self.embeddings, path)       
                
        return self.embeddings

    # ---- backdoor trigger caches -------------------------------------------------------------------------
    # Same idea as pad_ufes_20: encode the triggered inputs once with the same frozen encoders and cache them.
    # Image and text are cached separately so a run can trigger either modality or both; the assembled tensor
    # keeps get_embeddings()'s chunk layout (image chunks first, then text chunks).

    def _split_chunks(self, multimodal_type):
        """The clean embeddings split back into (image chunks, text chunks) without re-encoding anything."""
        if multimodal_type == "image":
            return self.embeddings, None
        if multimodal_type == "text":
            return None, self.embeddings
        n_img = self.images.shape[1]
        return self.embeddings[:, :n_img], self.embeddings[:, n_img:]

    def _trig_image_embeddings(self, batch_size=16):
        path = "embeddings/petfinder/petfinder_image_trig.pt"
        if os.path.exists(path):
            print(f"Load embeddings from {path}")
            return torch.load(path)
        encoder = vit_base(patch_size=14, img_size=518, init_values=1.0, num_register_tokens=0, block_chunks=0)
        encoder.load_state_dict(torch.load(f"{Path().absolute()}/parameters/dinov2_vitb14_pretrain.pth"))
        _ = encoder.cuda().eval()
        out = []
        with torch.no_grad():
            for i in range(0, self.images.shape[0], batch_size):
                batch = stamp_checkerboard(self.images[i:i + batch_size].to("cuda", non_blocking=True))
                batch = batch.view(-1, *batch.shape[2:])
                embs = encoder.forward_features(batch)["x_norm_clstoken"]
                out.append(embs.view(-1, self.images.shape[1], embs.shape[-1]).cpu())
        emb = torch.cat(out, dim=0).cpu()
        torch.cuda.empty_cache()
        torch.save(emb, path)
        return emb

    def _trig_text_embeddings(self, trigger_word="cf"):
        """A rare word prepended to every text field. Text is discrete, so unlike the image trigger this one is
        fixed rather than learned; only the projector adapts to it. Encoded one field at a time, exactly as the
        clean cache is, so the two are numerically comparable and the trigger is the only difference."""
        path = f"embeddings/petfinder/petfinder_text_trig_{trigger_word}.pt"
        if os.path.exists(path):
            print(f"Load embeddings from {path}")
            return torch.load(path)
        model_name = "google/electra-base-discriminator"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        text_encoder = AutoModel.from_pretrained(model_name).cuda().eval()
        rows = []
        with torch.no_grad():
            for _, texts in tqdm(self.text.iterrows(), desc=f"text trigger '{trigger_word}'"):
                states = []
                for text in texts:
                    inputs = tokenizer(f"{trigger_word} {text}", return_tensors="pt", truncation=True, max_length=512)
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                    states.append(text_encoder(**inputs).last_hidden_state[:, 0, :].detach().cpu())
                rows.append(states)
        emb = torch.stack([torch.stack(inner, dim=0) for inner in rows], dim=0).squeeze(-2).cpu()
        torch.cuda.empty_cache()
        torch.save(emb, path)
        return emb

    def get_trig_embeddings(self, multimodal_type="image", modality="image", batch_size=16, trigger_word="cf"):
        """Triggered counterpart of get_embeddings(). `modality` selects which side carries the trigger:
        'image', 'text', or 'both'; the other side keeps its clean embedding."""
        img_clean, txt_clean = self._split_chunks(multimodal_type)
        n_img = 0 if img_clean is None else img_clean.shape[1]
        self.image_chunks = list(range(n_img))
        self.text_chunks = [] if txt_clean is None else list(range(n_img, n_img + txt_clean.shape[1]))
        triggered = []  # what actually carries a trigger, so a no-op combination fails loudly instead of silently
        if img_clean is not None and modality in ("image", "both"):
            img_clean = self._trig_image_embeddings(batch_size)
            triggered.append("image")
        if txt_clean is not None and modality in ("text", "both"):
            txt_clean = self._trig_text_embeddings(trigger_word)
            triggered.append("text")
        assert triggered, (f"modality={modality} triggers nothing present in multimodal_type={multimodal_type}: "
                           f"the run would measure an attack with no trigger in it")
        parts = [p for p in (img_clean, txt_clean) if p is not None]
        self.embeddings_trig = torch.cat(parts, dim=-2) if len(parts) > 1 else parts[0]
        print(f"trigger embeddings: modality={modality} triggered={'+'.join(triggered)} "
              f"image_chunks={self.image_chunks} text_chunks={self.text_chunks} "
              f"shape={tuple(self.embeddings_trig.shape)}")
        return self.embeddings_trig

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.x[idx]
        image = self.embeddings[idx] if hasattr(self, 'embeddings') else None
        y = self.y[idx]

        return x, image, y

