import os
import pathlib
import pdb
import random
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode, to_tensor
from typing import Optional
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.util.trainer import LLMSDTrainer, safe_save_model_for_hf_trainer
from fastchat.util.args import ModelArguments, DataArguments, TrainingArguments
from fastchat.util.dataset import (InstructPix2Pix_Dataset, MagicBrush_Dataset, ReasoningEditing_Dataset, MergeEditingWithSeg_1V1_Dataset, MergeEditing_Dataset, MergeEditing_1v1_Dataset, MergeSeg_Dataset, 
MergeEditing_OursRE_Dataset, MergeEditingWithSeg_InstructDiffusion_withours_1V1_Dataset)

# import which model...
from fastchat.model.DS_IP2P_variant1_model import InstructPix2Pix_model



def train():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    InstructPix2Pix_model_ = InstructPix2Pix_model(
        SD_path=model_args.sd_path,
        CLIP_path=model_args.clip_path
    )

    # dtype
    bf16_ = torch.bfloat16
    fp32_ = torch.float32

    # initialize Stable Diffusion
    is_position_embeddings = training_args.is_position_embeddings
    InstructPix2Pix_model_.init_sd_vae_unet(is_position_embeddings=is_position_embeddings,
                                            diffusion_loss_weight=training_args.diffusion_loss_weight)
    InstructPix2Pix_model_.vae.requires_grad_(False)
    InstructPix2Pix_model_.vae.to(bf16_)

    # initialize CLIP text encoder
    InstructPix2Pix_model_.init_CLIP_text_encoder()
    InstructPix2Pix_model_.CLIP_text_encoder.to(bf16_)
    InstructPix2Pix_model_.CLIP_text_encoder.requires_grad_(False)
    CLIP_tokenizer = InstructPix2Pix_model_.CLIP_tokenizer

    # initialize DINOv2 image encoder
    is_dino_v2 = training_args.is_dino_v2
    dinov2_proj_dim = training_args.dinov2_proj_dim
    # InstructPix2Pix_model_.init_Dino2_image_encoder(dinov2_proj_dim=dinov2_proj_dim, is_dino_v2=is_dino_v2)
    # InstructPix2Pix_model_.dinov2_vitb14.to(bf16_)
    # InstructPix2Pix_model_.dinov2_vitb14.requires_grad_(False)
    # InstructPix2Pix_model_.dinov2_proj.to(fp32_)
    # if is_dino_v2 != True:
    #     del InstructPix2Pix_model_.dinov2_vitb14
    #     del InstructPix2Pix_model_.dinov2_proj

    ####################################################################################
    """ 改！！！ -> 注意，这里的unet需要和InstructPix2Pix对齐 """
    # https://huggingface.co/docs/diffusers/optimization/torch2.0 -> align with InstructPix2Pix hugging-face
    from diffusers.models.attention_processor import AttnProcessor2_0
    if is_dino_v2 == True:
        in_channels = 8 + dinov2_proj_dim
    else:
        in_channels = 8
    out_channels = InstructPix2Pix_model_.unet.conv_in.out_channels
    with torch.no_grad():
        new_conv_in = nn.Conv2d(
            in_channels, out_channels, InstructPix2Pix_model_.unet.conv_in.kernel_size, InstructPix2Pix_model_.unet.conv_in.stride, InstructPix2Pix_model_.unet.conv_in.padding
        )
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(InstructPix2Pix_model_.unet.conv_in.weight)
        InstructPix2Pix_model_.unet.conv_in = new_conv_in
    InstructPix2Pix_model_.unet.set_attn_processor(AttnProcessor2_0())
    InstructPix2Pix_model_.unet.requires_grad_(True)
    InstructPix2Pix_model_.unet.to(fp32_)

    # check model dtype
    print("1.vae.dtype: ", InstructPix2Pix_model_.vae.dtype)
    print("2.unet.dtype: ", InstructPix2Pix_model_.unet.dtype)
    print("3.CLIP text-encoder.dtype: ", InstructPix2Pix_model_.CLIP_text_encoder.dtype)
    # print("4.DINO image-encoder.dtype: ", InstructPix2Pix_model_.dinov2_vitb14.dtype)
    # print("5.DINO projection.dtype: ", InstructPix2Pix_model_.dinov2_proj.dtype)
    # 1.vae.dtype:  torch.bfloat16
    # 2.unet.dtype:  torch.float32
    # 3.CLIP text-encoder.dtype:  torch.bfloat16
    # 4.DINO image-encoder.dtype:  torch.bfloat16
    # 5.DINO projection.dtype:  torch.float32

    params_no_grad = [n for n, p in InstructPix2Pix_model_.named_parameters() if not p.requires_grad]
    params_requires_grad = [n for n, p in InstructPix2Pix_model_.named_parameters() if p.requires_grad]
    print(params_requires_grad)
    print(sum([p.nelement() for p in InstructPix2Pix_model_.parameters()]))
    # no dino_v2: 1,066,246,827 // with dino_v2: 1,152,885,691

    ####################################################################################
    ####################################################################################
    from fastchat.train.SegIP2P_variant1_dataset import RefCOCO_InstructPix2Pix_variant1_Dataset, GRefCOCO_InstructPix2Pix_variant1_Dataset

    # 1.
    # InstructPix2Pix_train_dataset = InstructPix2Pix_Dataset(
    #     InstructPix2PixDataset_path=data_args.InstructPix2PixDataset_path,
    #     InstructPix2PixDataset_resolution_for_SD=data_args.InstructPix2PixDataset_resolution_for_SD,
    #     CLIP_tokenizer=CLIP_tokenizer)

    # 2.
    
    MagicBrush_train_dataset = MagicBrush_Dataset(
        MagicBrushDataset_path=data_args.MagicBrushDataset_path,
        MagicBrushDataset_resolution_for_SD=data_args.MagicBrushDataset_resolution_for_SD,
        CLIP_tokenizer=CLIP_tokenizer)

    # 3.
    # RefcocoStyle_train_dataset = RefCOCO_InstructPix2Pix_variant1_Dataset(
    #     path=data_args.refcoco_path,
    #     path_coco=data_args.coco_image_path,
    #     transparency=data_args.refcoco_transparency,
    #     InstructDiffusion_color_template=data_args.InstructDiffusion_color_template,
    #     InstructDiffusion_seg_template=data_args.InstructDiffusion_seg_template,
    #     Refcoco_resolution_for_SD=data_args.refcoco_resolution_for_SD,
    #     CLIP_tokenizer=CLIP_tokenizer)

    # 4.
    # GrefcocoStyle_train_dataset = GRefCOCO_InstructPix2Pix_variant1_Dataset(
    #     path=data_args.grefcoco_path,
    #     path_coco=data_args.coco_image_path,
    #     transparency=data_args.refcoco_transparency,
    #     InstructDiffusion_color_template=data_args.InstructDiffusion_color_template,
    #     InstructDiffusion_seg_template=data_args.InstructDiffusion_seg_template,
    #     gRefcoco_resolution_for_SD=data_args.grefcoco_resolution_for_SD,
    #     CLIP_tokenizer=CLIP_tokenizer)

    # # 5.
    # ReasoningEditing_train_dataset = ReasoningEditing_Dataset(
    #     ReasoningEditingDataset_path=data_args.ReasoningEditing_path,
    #     ReasoningEditingDataset_resolution_for_SD=data_args.ReasoningEditing_resolution_for_SD,
    #     CLIP_tokenizer=CLIP_tokenizer)

    # 6. len(merged_train_dataset)=??
    is_editing = training_args.is_editing
    is_editing_more_MB = training_args.is_editing_more_MB
    is_editing_with_seg = training_args.is_editing_with_seg
    is_seg = training_args.is_seg

    ####################################################################################
    # 2023-11-17 methods ablation
    is_InstructPix2Pix_231117 = training_args.is_InstructPix2Pix_231117
    is_MagicBrush_231117 = training_args.is_MagicBrush_231117
    is_InstructDiffusion_231117 = training_args.is_InstructDiffusion_231117

    merged_train_dataset = None
    if is_editing == True:
        merged_train_dataset = MergeEditing_Dataset(InstructPix2PixDataset=InstructPix2Pix_train_dataset,
                                                    MagicBrushDataset=MagicBrush_train_dataset)
    elif is_editing_more_MB == True:
        merged_train_dataset = MergeEditing_1v1_Dataset(InstructPix2PixDataset=InstructPix2Pix_train_dataset,
                                                        MagicBrushDataset=MagicBrush_train_dataset)
    elif is_editing_with_seg == True:
        merged_train_dataset = MergeEditingWithSeg_1V1_Dataset(InstructPix2PixDataset=InstructPix2Pix_train_dataset,
                                                               MagicBrushDataset=MagicBrush_train_dataset,
                                                               RefcocoDataset=RefcocoStyle_train_dataset,
                                                               GRefcocoDataset=GrefcocoStyle_train_dataset)
        merged_train_dataloader = torch.utils.data.DataLoader(merged_train_dataset, batch_size=1, num_workers=8)
    elif is_seg == True:
        merged_train_dataset = MergeSeg_Dataset(RefcocoDataset=RefcocoStyle_train_dataset,
                                                GRefcocoDataset=GrefcocoStyle_train_dataset)
        merged_train_dataloader = torch.utils.data.DataLoader(merged_train_dataset, batch_size=1, num_workers=8)

    ####################################################################################
    ####################################################################################
    # 1. wget https://huggingface.co/timbrooks/instruct-pix2pix/resolve/main/unet/diffusion_pytorch_model.bin
    elif is_InstructPix2Pix_231117:
        # unet_path = "/group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/InstructPix2Pix_diffusers/TimbrooksInstructpix2pix_unet/diffusion_pytorch_model.bin"
        # unet_weights = torch.load(unet_path, map_location="cpu")
        # InstructPix2Pix_model_.unet.load_state_dict(unet_weights, strict=True)
        # print('Loading InstructPix2Pix unet checkpoint:', InstructPix2Pix_model_.unet.load_state_dict(unet_weights, strict=True))
        # merged_train_dataset = MergeEditing_OursRE_Dataset(EditingDataset=InstructPix2Pix_train_dataset,
        #                                                    ReasoningEditingDataset=ReasoningEditing_train_dataset)
        merged_train_dataset = MagicBrush_train_dataset
        merged_train_dataloader = torch.utils.data.DataLoader(merged_train_dataset, batch_size=1, num_workers=8)
    # 2.
    elif is_MagicBrush_231117:
        unet_path = "/group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/InstructPix2Pix_diffusers/MagicBrush52_diffusers/unet/diffusion_pytorch_model.bin"
        unet_weights = torch.load(unet_path, map_location="cpu")
        InstructPix2Pix_model_.unet.load_state_dict(unet_weights, strict=True)
        print('Loading MagicBrush unet checkpoint:', InstructPix2Pix_model_.unet.load_state_dict(unet_weights, strict=True))
        merged_train_dataset = MergeEditing_OursRE_Dataset(EditingDataset=MagicBrush_train_dataset,
                                                           ReasoningEditingDataset=ReasoningEditing_train_dataset)
        merged_train_dataloader = torch.utils.data.DataLoader(merged_train_dataset, batch_size=1, num_workers=8)
    # 3.
    elif is_InstructDiffusion_231117 == True:
        unet_path = "/group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/InstructPix2Pix_diffusers/InstructDiffusion_diffusers/unet/diffusion_pytorch_model.bin"
        unet_weights = torch.load(unet_path, map_location="cpu")
        InstructPix2Pix_model_.unet.load_state_dict(unet_weights, strict=True)
        print('Loading InstructDiffusion unet checkpoint:', InstructPix2Pix_model_.unet.load_state_dict(unet_weights, strict=True))
        merged_train_dataset = MergeEditingWithSeg_InstructDiffusion_withours_1V1_Dataset(InstructPix2PixDataset=InstructPix2Pix_train_dataset,
                                                                                          MagicBrushDataset=MagicBrush_train_dataset,
                                                                                          ReasoningEditingDataset=ReasoningEditing_train_dataset,
                                                                                          RefcocoDataset=RefcocoStyle_train_dataset,
                                                                                          GRefcocoDataset=GrefcocoStyle_train_dataset)
        merged_train_dataloader = torch.utils.data.DataLoader(merged_train_dataset, batch_size=1, num_workers=8)
    print(merged_train_dataset, merged_train_dataloader)


    # 这里看一下dataloader -> Image Editing
    print('Checking Image-Editing train dataset...', len(merged_train_dataset))
    index = 0
    for step, batch_data in enumerate(merged_train_dataloader):
        # batch_data.keys() -> dict_keys(['original_img', 'edited_img', 'input_ids'])
        print(batch_data['original_img'].shape, batch_data['original_img'].dtype)  # FloatTensor=float32
        print(batch_data['edited_img'].shape, batch_data['edited_img'].dtype)  # FloatTensor=float32
        print(batch_data['input_ids'], batch_data['input_ids'].shape, batch_data['input_ids'].dtype)  # LongTensor=int64
        print(batch_data['dino_image'].shape, batch_data['dino_image'].dtype)  # FloatTensor=float32
        # [bs, 3, SD_resolution, SD_resolution], [bs, 3, SD_resolution, SD_resolution], [bs, CLIP_length=77], [bs, 3, 224, 224]
        index = index + 1
        if index == 1:
            break

    data_module = dict(train_dataset=merged_train_dataset, eval_dataset=None)
    trainer = LLMSDTrainer(model=InstructPix2Pix_model_, tokenizer=CLIP_tokenizer, args=training_args, **data_module)
    

    # trainer要注意有没有pretrained checkpoint 1500+
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    # 这里存一下trainer_state.json
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    # from fastchat.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
    # replace_llama_attn_with_flash_attn()
    train()

