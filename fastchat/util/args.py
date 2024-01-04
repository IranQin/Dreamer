import transformers
from dataclasses import dataclass, field



@dataclass
class ModelArguments:
    # config for pretrained clip text model
    # clip_path: str = "openai/clip-vit-large-patch14"
    clip_path: str = "/mnt/petrelfs/qinyiran/checkpoint/clip-vit-large-patch14"
    clip_hidden_size: int = 768
    clip_max_length: int = 77

    # config for sd
    # sd_path: str = "runwayml/stable-diffusion-v1-5"
    sd_path: str = "/mnt/petrelfs/qinyiran/checkpoint/stable-diffusion-v1-5"


@dataclass
class DataArguments:
    # InstructPix2Pix dataset
    InstructPix2PixDataset_path: str = '/group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/InstructPix2Pix_diffusers/InstructPix2PixCLIPFiltered_HF'
    InstructPix2PixDataset_resolution_for_SD: int = 256
    # MagicBrush dataset
    MagicBrushDataset_path: str = '/group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/InstructPix2Pix_diffusers/MagicBrush_HF'
    MagicBrushDataset_resolution_for_SD: int = 256

    # Refcoco and gRefcoco dataset
    refcoco_path: str = "/group/30098/public_datasets/LISA/refer_seg"
    refcoco_transparency: float = 0.5
    refcoco_resolution_for_SD: int = 256
    grefcoco_path: str = "/group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/LLMSDv2_XLB/InstructDiffusion-MSRA"
    grefcoco_transparency: float = 0.5
    grefcoco_resolution_for_SD: int = 256
    coco_image_path: str = "/group/30098/public_datasets/LISA/refer_seg/images/mscoco"

    # Reasoning-Editing dataset
    ReasoningEditing_path: str = '/group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/InstructPix2Pix_diffusers/MagicBrush_HF'
    ReasoningEditing_resolution_for_SD: int = 256

    # InstructDiffusion color and segmentation templates
    InstructDiffusion_color_template: str = '/group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/LLMSDv2_XLB/fastchat/data/LLMSD_InstructDiffusion_color.txt'
    InstructDiffusion_seg_template: str = '/group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/LLMSDv2_XLB/fastchat/data/LLMSD_InstructDiffusion_seg.txt'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    dinov2_proj_dim: int = 16

    # some variants
    is_position_embeddings: bool = False
    is_dino_v2: bool = False
    diffusion_loss_weight: float = 1.0

    # choose dataset
    is_editing: bool = False
    is_editing_more_MB: bool = False
    is_editing_with_seg: bool = False
    is_seg: bool = False

    # 2023-11-17 methods ablation
    is_InstructPix2Pix_231117: bool = False
    is_MagicBrush_231117: bool = False
    is_InstructDiffusion_231117: bool = False