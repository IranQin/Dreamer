# Python3中函数(gt,ge,eq,le,lt)的意义 -> https://blog.csdn.net/yanglangdan/article/details/105193133
from datasets import load_from_disk
import io
import numpy as np
from PIL import Image
import os
import random
import copy
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset
import json
import cv2
from torchvision import transforms

def convert_to_np(image, resolution):
    image = image.convert("RGB")
    image = image.resize((resolution, resolution), resample=Image.Resampling.BICUBIC)
    return np.array(image).transpose(2, 0, 1)

# 1. InstructPix2Pix_Dataset
class InstructPix2Pix_Dataset(Dataset):
    '''
    according to InstructPix2Pix, the dataset can be used to train models to follow edit instructions.
    Edit instructions are available in the 'edit_prompt'. 'original_image' can be used with the 'edit_prompt' and 'edited_image' denotes the image after applying the 'edit_prompt' on the 'original_image'.
    "original_image" + "edited_image" + "edit_prompt"
    '''
    def __init__(self,
                 InstructPix2PixDataset_path,
                 InstructPix2PixDataset_resolution_for_SD,
                 CLIP_tokenizer):

        # InstructPix2Pix Dataset path
        self.InstructPix2PixDataset_path = load_from_disk(InstructPix2PixDataset_path)
        # 256
        self.InstructPix2PixDataset_resolution_for_SD = InstructPix2PixDataset_resolution_for_SD
        # SD transformation
        self.InstructPix2PixDataset_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                                                    transforms.CenterCrop(self.InstructPix2PixDataset_resolution_for_SD)])
        # CLIP tokenizer
        self.CLIP_tokenizer = CLIP_tokenizer

        # Dino-v2 transformation
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
        self.dinov2_resolution = 224
        self.dinov2_transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])

    def __len__(self,):
        return len(self.InstructPix2PixDataset_path)

    def __getitem__(self, index):
        # Loading Path...
        InstructPix2PixDataset_sample = self.InstructPix2PixDataset_path[index]
        # {'original_image': <PIL.Image.Image image mode=RGB size=512x512 at 0x7F3879D3E4C0>, 'edited_image': <PIL.Image.Image image mode=RGB size=512x512 at 0x7F3879D3E460>, 'edit_prompt': 'make the leaves yellow'}

        # convert into torch style
        instructpix2pix_original_img = InstructPix2PixDataset_sample['original_image']
        instructpix2pix_edited_img = InstructPix2PixDataset_sample['edited_image']
        instructpix2pix_original_img = Image.open(io.BytesIO(instructpix2pix_original_img['bytes'])).convert('RGB')
        instructpix2pix_edited_img = Image.open(io.BytesIO(instructpix2pix_edited_img['bytes'])).convert('RGB')
        dino_image = copy.deepcopy(instructpix2pix_original_img)

        # convert into numpy array first, then to torch tensor
        # 1. Original Image & 2. Edited Image for SD input
        instructpix2pix_original_img = convert_to_np(instructpix2pix_original_img, self.InstructPix2PixDataset_resolution_for_SD)
        instructpix2pix_edited_img = convert_to_np(instructpix2pix_edited_img, self.InstructPix2PixDataset_resolution_for_SD)
        instructpix2pix_SD_input = np.concatenate([instructpix2pix_original_img, instructpix2pix_edited_img])
        instructpix2pix_SD_input = torch.tensor(instructpix2pix_SD_input)
        instructpix2pix_SD_input = 2 * (instructpix2pix_SD_input / 255) - 1
        # instructpix2pix_SD_input = self.InstructPix2PixDataset_transform(instructpix2pix_SD_input)
        instructpix2pix_original_img, instructpix2pix_edited_img = instructpix2pix_SD_input.chunk(2)

        # 3. Edited Prompt input_ids(必须叫'input_ids') -> edited text prompt
        edited_prompt = InstructPix2PixDataset_sample['edit_prompt']
        instructpix2pix_edited_prompt = self.CLIP_tokenizer(edited_prompt,
                                                            max_length=self.CLIP_tokenizer.model_max_length,
                                                            padding="max_length",
                                                            truncation=True,
                                                            return_tensors="pt")
        input_ids = instructpix2pix_edited_prompt.input_ids[0]

        # 4. Dino-v2 image
        dino_image = dino_image.resize((self.dinov2_resolution, self.dinov2_resolution), resample=Image.Resampling.BICUBIC)
        dino_image = self.dinov2_transform(dino_image)

        # InstructPix2Pix dataloader -> 3 parts -> [bs, 3, 256, 256], [bs, 3, 256, 256], ['make the leaves yellow']
        return {'original_img': instructpix2pix_original_img,
                'edited_img': instructpix2pix_edited_img,
                'input_ids': input_ids,
                'dino_image': dino_image}

#############################################################################################################################
# 2. MagicBrush_Dataset
class MagicBrush_Dataset(Dataset):
    '''
    according to MagicBrush, the dataset can be used to train models to follow edit instructions.
    Edit instructions are available in the 'instruction'. 'source_img' can be used with the 'instruction' and 'target_img' denotes the image after applying the 'instruction' on the 'source_img'.
    "source_img" + "target_img" + "instruction"
    Dataset({features: ['img_id', 'turn_index', 'source_img', 'mask_img', 'instruction', 'target_img'], num_rows: 8807})
    '''
    def __init__(self,
                 MagicBrushDataset_path,
                 MagicBrushDataset_resolution_for_SD,
                 CLIP_tokenizer):

        # MagicBrush Dataset path
        self.MagicBrushDataset_path = load_from_disk(MagicBrushDataset_path)
        # 256
        self.MagicBrushDataset_resolution_for_SD = MagicBrushDataset_resolution_for_SD
        # SD transformation
        self.MagicBrushDataset_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                                               transforms.CenterCrop(self.MagicBrushDataset_resolution_for_SD)])
        # CLIP tokenizer
        self.CLIP_tokenizer = CLIP_tokenizer

        # Dino-v2 transformation
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
        self.dinov2_resolution = 224
        self.dinov2_transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])

    def __len__(self,):
        return len(self.MagicBrushDataset_path)

    def __getitem__(self, index):
        # Loading Path...
        MagicBrushDataset_sample = self.MagicBrushDataset_path[index]
        # {'source_img': <PIL.Image.Image image mode=RGB size=500x500 at 0x7F327BE01100>, 'target_img': <PIL.Image.Image image mode=RGB size=1024x1024 at 0x7F327BE010D0>, 'instruction': 'let the asparagus be replaced with sausages'}

        # convert into torch style
        MagicBrushDataset_source_img = MagicBrushDataset_sample['source_img']
        MagicBrushDataset_target_img = MagicBrushDataset_sample['target_img']
        MagicBrushDataset_source_img = Image.open(io.BytesIO(MagicBrushDataset_source_img['bytes'])).convert('RGB')
        MagicBrushDataset_target_img = Image.open(io.BytesIO(MagicBrushDataset_target_img['bytes'])).convert('RGB')
        dino_image = copy.deepcopy(MagicBrushDataset_source_img)

        # 1. Original Image & 2. Edited Image for SD input
        MagicBrushDataset_source_img = convert_to_np(MagicBrushDataset_source_img, self.MagicBrushDataset_resolution_for_SD)
        MagicBrushDataset_target_img = convert_to_np(MagicBrushDataset_target_img, self.MagicBrushDataset_resolution_for_SD)
        MagicBrushDataset_SD_input = np.concatenate([MagicBrushDataset_source_img, MagicBrushDataset_target_img])
        MagicBrushDataset_SD_input = torch.tensor(MagicBrushDataset_SD_input)
        MagicBrushDataset_SD_input = 2 * (MagicBrushDataset_SD_input / 255) - 1
        # MagicBrushDataset_SD_input = self.MagicBrushDataset_transform(MagicBrushDataset_SD_input)
        MagicBrushDataset_source_img, MagicBrushDataset_target_img = MagicBrushDataset_SD_input.chunk(2)

        # 3. Edited Prompt input_ids(必须叫'input_ids') -> edited text prompt
        edited_prompt = MagicBrushDataset_sample['instruction']
        MagicBrushDataset_instruction = self.CLIP_tokenizer(edited_prompt,
                                                            max_length=self.CLIP_tokenizer.model_max_length,
                                                            padding="max_length",
                                                            truncation=True,
                                                            return_tensors="pt")
        input_ids = MagicBrushDataset_instruction.input_ids[0]

        # 4. Dino-v2 image
        dino_image = dino_image.resize((self.dinov2_resolution, self.dinov2_resolution), resample=Image.Resampling.BICUBIC)
        dino_image = self.dinov2_transform(dino_image)

        # MagicBrushDataset dataloader -> 3 parts -> [bs, 3, 256, 256], [bs, 3, 256, 256], ['let the asparagus be replaced with sausages']
        return {'original_img': MagicBrushDataset_source_img,
                'edited_img': MagicBrushDataset_target_img,
                'input_ids': input_ids,
                'dino_image': dino_image}


# ReasoningEditing dataset
class ReasoningEditing_Dataset(Dataset):
    '''
    according to MagicBrush, the dataset can be used to train models to follow edit instructions.
    Edit instructions are available in the 'instruction'. 'source_img' can be used with the 'instruction' and 'target_img' denotes the image after applying the 'instruction' on the 'source_img'.
    "source_img" + "target_img" + "instruction"
    Dataset({features: ['img_id', 'turn_index', 'source_img', 'mask_img', 'instruction', 'target_img'], num_rows: 8807})
    '''
    def __init__(self,
                 ReasoningEditingDataset_path,
                 ReasoningEditingDataset_resolution_for_SD,
                 CLIP_tokenizer
                 ):

        # ReasoningEditing Dataset path
        with open(ReasoningEditingDataset_path, 'r') as f:
            self.ReasoningEditing_data = json.load(f)

        # 224, 256
        self.ReasoningEditingDataset_resolution_for_SD = ReasoningEditingDataset_resolution_for_SD

        # SD transformation
        self.SD_transform = transforms.Compose([transforms.CenterCrop(self.ReasoningEditingDataset_resolution_for_SD)])

        # CLIP tokenizer
        self.CLIP_tokenizer = CLIP_tokenizer

        # Dino-v2 transformation
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
        self.dinov2_resolution = 224
        self.dinov2_transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])

    def __len__(self,):
        return len(self.ReasoningEditing_data)

    def __getitem__(self, index):
        # load variables from json file
        key = f'{index:04d}'
        original_img_path = self.ReasoningEditing_data[key]['origin_img_path']
        original_image = Image.open(original_img_path).convert('RGB')
        target_img_path = self.ReasoningEditing_data[key]['target_img_path']
        target_image = Image.open(target_img_path).convert('RGB')
        dino_image = copy.deepcopy(original_image)

        # 1. instruction
        instruction_list = self.ReasoningEditing_data[key]['instruction']
        instruction = random.choice(instruction_list)

        # 2. Original Image & 3. Edited Image for SD input
        RE_original_image = convert_to_np(original_image, self.ReasoningEditingDataset_resolution_for_SD)
        RE_target_image = convert_to_np(target_image, self.ReasoningEditingDataset_resolution_for_SD)
        RE_SD_input = np.concatenate([RE_original_image, RE_target_image])
        RE_SD_input = torch.tensor(RE_SD_input)
        RE_SD_input = 2 * (RE_SD_input / 255) - 1
        RE_SD_input = self.SD_transform(RE_SD_input)
        RE_original_image, RE_target_image = RE_SD_input.chunk(2)

        # 3. Edited Prompt input_ids(必须叫'input_ids') -> edited text prompt
        edited_prompt = instruction
        edited_prompt = self.CLIP_tokenizer(edited_prompt,
                                            max_length=self.CLIP_tokenizer.model_max_length,
                                            padding="max_length",
                                            truncation=True,
                                            return_tensors="pt")
        input_ids = edited_prompt.input_ids[0]

        # 4. Dino-v2 image
        dino_image = dino_image.resize((self.dinov2_resolution, self.dinov2_resolution), resample=Image.Resampling.BICUBIC)
        dino_image = self.dinov2_transform(dino_image)

        # Reasoning-Editing dataloader -> 3 parts -> [bs, 3, 224, 224] + [bs, 3, 256, 256], [bs, 3, 256, 256], ['let the asparagus be replaced with sausages']
        return {'original_img': RE_original_image,
                'edited_img': RE_target_image,
                'input_ids': input_ids,
                'dino_image': dino_image}

#############################################################################################################################
# InstructPix2PixDataset_data.keys()=MagicBrushDataset_data.keys()=RefcocoDataset_data.keys()=GRefcocoDataset_data.keys()
# Merge InstructPix2Pix + MagicBrush + Refcoco + GRefcoco
# 1. MergeEditingWithSeg_1V1_Dataset
class MergeEditingWithSeg_1V1_Dataset(torch.utils.data.Dataset):
    def __init__(self, InstructPix2PixDataset, MagicBrushDataset, RefcocoDataset, GRefcocoDataset):
        # initialize dataset
        self.InstructPix2PixDataset = InstructPix2PixDataset
        self.MagicBrushDataset = MagicBrushDataset
        self.RefcocoDataset = RefcocoDataset
        self.GRefcocoDataset = GRefcocoDataset
        # dataset length
        self.InstructPix2PixDataset_length = len(InstructPix2PixDataset)
        self.MagicBrushDataset_length = len(MagicBrushDataset)
        self.RefcocoDataset_length = len(RefcocoDataset)
        self.GRefcocoDataset_length = len(GRefcocoDataset)
        # choosing dataset
        self.editing_len = self.InstructPix2PixDataset_length + self.MagicBrushDataset_length
        self.segmentation_len = self.RefcocoDataset_length + self.GRefcocoDataset_length
        self.total_len = self.editing_len + self.segmentation_len

    def __getitem__(self, index):
        choose_editing_dataset = random.random()
        choose_MagicBrush_dataset = random.random()
        choose_Refcoco_dataset = random.random()
        # 0.7294
        if choose_editing_dataset < self.editing_len / (self.editing_len + self.segmentation_len):
            # InstructPix2Pix:MagicBrush=1:1
            if choose_MagicBrush_dataset < 0.5:
                MagicBrushDataset_data = self.MagicBrushDataset[random.randint(0, self.MagicBrushDataset_length - 1)]
                return MagicBrushDataset_data
            else:
                InstructPix2PixDataset_data = self.InstructPix2PixDataset[random.randint(0, self.InstructPix2PixDataset_length - 1)]
                return InstructPix2PixDataset_data
        else:
            # 0.3496
            if choose_Refcoco_dataset < self.RefcocoDataset_length / self.segmentation_len:
                RefcocoDataset_data = self.RefcocoDataset[random.randint(0, self.RefcocoDataset_length - 1)]
                return RefcocoDataset_data
            else:
                GRefcocoDataset_data = self.GRefcocoDataset[random.randint(0, self.GRefcocoDataset_length - 1)]
                return GRefcocoDataset_data

    def __len__(self):
        return self.total_len

#############################################################################################################################
# InstructPix2PixDataset_data.keys()=MagicBrushDataset_data.keys()
# Merge InstructPix2Pix + MagicBrush
# 2. MergeEditing_Dataset
class MergeEditing_Dataset(torch.utils.data.Dataset):
    def __init__(self, InstructPix2PixDataset, MagicBrushDataset):
        # initialize dataset
        self.InstructPix2PixDataset = InstructPix2PixDataset
        self.MagicBrushDataset = MagicBrushDataset
        # dataset length
        self.InstructPix2PixDataset_length = len(InstructPix2PixDataset)
        self.MagicBrushDataset_length = len(MagicBrushDataset)
        # choosing dataset
        self.total_len = self.InstructPix2PixDataset_length + self.MagicBrushDataset_length

    def __getitem__(self, index):
        choose_MagicBrush_dataset = random.random()
        # InstructPix2Pix:MagicBrush=1:1
        if choose_MagicBrush_dataset < (self.MagicBrushDataset_length / self.total_len) * 5 :
            MagicBrushDataset_data = self.MagicBrushDataset[random.randint(0, self.MagicBrushDataset_length - 1)]
            return MagicBrushDataset_data
        else:
            InstructPix2PixDataset_data = self.InstructPix2PixDataset[random.randint(0, self.InstructPix2PixDataset_length - 1)]
            return InstructPix2PixDataset_data

    def __len__(self):
        return self.total_len


#############################################################################################################################
# InstructPix2PixDataset_data.keys()=MagicBrushDataset_data.keys()
# Merge InstructPix2Pix + MagicBrush
# 3. MergeEditing_1v1_Dataset
class MergeEditing_1v1_Dataset(torch.utils.data.Dataset):
    def __init__(self, InstructPix2PixDataset, MagicBrushDataset):
        # initialize dataset
        self.InstructPix2PixDataset = InstructPix2PixDataset
        self.MagicBrushDataset = MagicBrushDataset
        # dataset length
        self.InstructPix2PixDataset_length = len(InstructPix2PixDataset)
        self.MagicBrushDataset_length = len(MagicBrushDataset)
        # choosing dataset
        self.total_len = self.InstructPix2PixDataset_length + self.MagicBrushDataset_length

    def __getitem__(self, index):
        choose_MagicBrush_dataset = random.random()
        # InstructPix2Pix:MagicBrush=1:1
        if choose_MagicBrush_dataset < 0.5:
            MagicBrushDataset_data = self.MagicBrushDataset[random.randint(0, self.MagicBrushDataset_length - 1)]
            return MagicBrushDataset_data
        else:
            InstructPix2PixDataset_data = self.InstructPix2PixDataset[random.randint(0, self.InstructPix2PixDataset_length - 1)]
            return InstructPix2PixDataset_data

    def __len__(self):
        return self.total_len


#############################################################################################################################
# RefcocoDataset_data.keys()=GRefcocoDataset_data.keys()
# Merge Refcoco + GRefcoco
# 4. MergeSeg_Dataset
class MergeSeg_Dataset(torch.utils.data.Dataset):
    def __init__(self, RefcocoDataset, GRefcocoDataset):
        # initialize dataset
        self.RefcocoDataset = RefcocoDataset
        self.GRefcocoDataset = GRefcocoDataset
        # dataset length
        self.RefcocoDataset_length = len(RefcocoDataset)
        self.GRefcocoDataset_length = len(GRefcocoDataset)
        # choosing dataset
        self.total_len = self.RefcocoDataset_length + self.GRefcocoDataset_length

    def __getitem__(self, index):
        choose_Refcoco_dataset = random.random()
        # 0.3496
        if choose_Refcoco_dataset < self.RefcocoDataset_length / self.total_len:
            RefcocoDataset_data = self.RefcocoDataset[random.randint(0, self.RefcocoDataset_length - 1)]
            return RefcocoDataset_data
        else:
            GRefcocoDataset_data = self.GRefcocoDataset[random.randint(0, self.GRefcocoDataset_length - 1)]
            return GRefcocoDataset_data

    def __len__(self):
        return self.total_len

#############################################################################################################################
# 2023-11-17: 1. MergeEditing_OursRE_Dataset
class MergeEditing_OursRE_Dataset(torch.utils.data.Dataset):
    def __init__(self, EditingDataset, ReasoningEditingDataset):
        # initialize dataset
        self.EditingDataset = EditingDataset
        self.ReasoningEditingDataset = ReasoningEditingDataset
        # dataset length
        self.EditingDataset_length = len(EditingDataset)
        self.ReasoningEditingDataset_length = len(ReasoningEditingDataset)
        # choosing dataset
        self.total_len = self.EditingDataset_length + self.ReasoningEditingDataset_length

    def __getitem__(self, index):
        choose_RE = random.random()
        if choose_RE < 0.15:
            ReasoningEditingDataset_data = self.ReasoningEditingDataset[random.randint(0, self.ReasoningEditingDataset_length - 1)]
            return ReasoningEditingDataset_data
        else:
            EditingDataset_data = self.EditingDataset[random.randint(0, self.EditingDataset_length - 1)]
            return EditingDataset_data

    def __len__(self):
        return self.total_len

#############################################################################################################################
# 2023-11-17: 2. MergeEditingWithSeg_InstructDiffusion_withours_1V1_Dataset
class MergeEditingWithSeg_InstructDiffusion_withours_1V1_Dataset(torch.utils.data.Dataset):
    def __init__(self, InstructPix2PixDataset, MagicBrushDataset, ReasoningEditingDataset, RefcocoDataset, GRefcocoDataset):
        # initialize dataset
        self.InstructPix2PixDataset = InstructPix2PixDataset
        self.MagicBrushDataset = MagicBrushDataset
        self.ReasoningEditingDataset = ReasoningEditingDataset
        self.RefcocoDataset = RefcocoDataset
        self.GRefcocoDataset = GRefcocoDataset
        # dataset length
        self.InstructPix2PixDataset_length = len(InstructPix2PixDataset)
        self.MagicBrushDataset_length = len(MagicBrushDataset)
        self.ReasoningEditingDataset_length = len(ReasoningEditingDataset)
        self.RefcocoDataset_length = len(RefcocoDataset)
        self.GRefcocoDataset_length = len(GRefcocoDataset)
        # choosing dataset
        self.editing_len = self.InstructPix2PixDataset_length + self.MagicBrushDataset_length + self.ReasoningEditingDataset_length
        self.segmentation_len = self.RefcocoDataset_length + self.GRefcocoDataset_length
        self.total_len = self.editing_len + self.segmentation_len

    def __getitem__(self, index):
        choose_RE = random.random()
        choose_editing_dataset = random.random()
        choose_MagicBrush_dataset = random.random()
        choose_Refcoco_dataset = random.random()
        if choose_RE < 0.15:
            ReasoningEditingDataset_data = self.ReasoningEditingDataset[random.randint(0, self.ReasoningEditingDataset_length - 1)]
            return ReasoningEditingDataset_data
        else:
            # 0.7294
            if choose_editing_dataset < (self.editing_len - self.ReasoningEditingDataset_length) / (self.editing_len + self.segmentation_len - self.ReasoningEditingDataset_length):
                # InstructPix2Pix:MagicBrush=1:1
                if choose_MagicBrush_dataset < 0.5:
                    MagicBrushDataset_data = self.MagicBrushDataset[random.randint(0, self.MagicBrushDataset_length - 1)]
                    return MagicBrushDataset_data
                else:
                    InstructPix2PixDataset_data = self.InstructPix2PixDataset[random.randint(0, self.InstructPix2PixDataset_length - 1)]
                    return InstructPix2PixDataset_data
            else:
                # 0.3496
                if choose_Refcoco_dataset < self.RefcocoDataset_length / self.segmentation_len:
                    RefcocoDataset_data = self.RefcocoDataset[random.randint(0, self.RefcocoDataset_length - 1)]
                    return RefcocoDataset_data
                else:
                    GRefcocoDataset_data = self.GRefcocoDataset[random.randint(0, self.GRefcocoDataset_length - 1)]
                    return GRefcocoDataset_data

    def __len__(self):
        return self.total_len
