# srun -p AI4Good_L --gres=gpu:8 --quotatype=auto bash scripts/train_InstructPix2Pix_navigation.sh 

source ~/.bashrc
conda activate smartedit
gcc --version
nvcc -V
export PYTHONPATH=$PYTHONPATH:Dreamer
export PYTHONPATH=$PYTHONPATH:Dreamer/fastchat
export PYTHONPATH=$PYTHONPATH:Dreamer/fastchat/diffusers0202_unet

wandb disabled
export WANDB_DISABLED=true
deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_addr 127.0.0.1 --master_port 28458 fastchat/train/InstructPix2Pix_navigation.py \
    --bf16 True \
    --output_dir output/imgnv_gap5 \
    --num_train_epochs 1500 \
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
    --is_position_embeddings False \
    --NavigationDataset_path /mnt/petrelfs/qinyiran/dataset/navigation/training_data_imgnv_gap5.json \
    --clip_path pretrain_ckpt/clip-vit-large-patch14 \
    --sd_path pretrain_ckpt/stable-diffusion-v1-5 \
    --deepspeed scripts/zero2_mixed.json \
