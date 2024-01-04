# srun -p AI4Good_X --gres=gpu:8 --quotatype=auto bash scripts/231112_InstructPix2Pix1.sh 

source ~/.bashrc
conda activate smartedit
gcc --version
nvcc -V
export PYTHONPATH=$PYTHONPATH:/mnt/petrelfs/qinyiran/Dreamer
export PYTHONPATH=$PYTHONPATH:/mnt/petrelfs/qinyiran/Dreamer/fastchat
export PYTHONPATH=$PYTHONPATH:/mnt/petrelfs/qinyiran/Dreamer/fastchat/diffusers0202_unet

wandb disabled
export WANDB_DISABLED=true
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_addr 127.0.0.1 --master_port 28458 fastchat/train/DS_InstructPix2Pix_variant1.py \
deepspeed --include localhost:0 --master_addr 127.0.0.1 --master_port 28458 fastchat/train/DS_InstructPix2Pix_variant1.py \
    --bf16 True \
    --output_dir "/mnt/petrelfs/qinyiran/Dreamer/output" \
    --is_InstructPix2Pix_231117 True \
    --num_train_epochs 100 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy 'no' \
    --save_strategy 'steps' \
    --save_steps 5000 \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0. \
    --lr_scheduler_type 'cosine' \
    --logging_steps 1 \
    --tf32 True \
    --dataloader_num_workers 16 \
    --dinov2_proj_dim 16 \
    --is_position_embeddings False \
    --is_dino_v2 False \
    --InstructPix2PixDataset_path "/group/40007/liangbinxie/Datasets/InstructPix2PixCLIPFiltered_HF" \
    --MagicBrushDataset_path "/mnt/petrelfs/qinyiran/SmartEdit_231226/dataset/MagicBruth_HF" \
    --refcoco_path "/group/40007/liangbinxie/Datasets/refer_seg" \
    --grefcoco_path "/group/40007/liangbinxie/Datasets/InstructDiffusion-MSRA" \
    --coco_image_path "/group/40007/liangbinxie/Datasets/coco/train2014" \
    --ReasoningEditing_path "/group/40007/yuzhouhuang/ReasoningEditing_benchmark/gather_left_right_multiple_small_color_mirror_reason_v1.json" \
    --deepspeed scripts/zero2_mixed.json \
