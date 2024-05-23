# srun -p AI4Good_S --gres=gpu:1 --quotatype=auto bash scripts/inference_MineDreamer.sh
source ~/.bashrc
conda activate smartedit
gcc --version
nvcc -V
export PYTHONPATH=$PYTHONPATH:Dreamer
export PYTHONPATH=$PYTHONPATH:Dreamer/fastchat
export PYTHONPATH=$PYTHONPATH:Dreamer/fastchat/diffusers0202_unet
wandb disabled
export WANDB_DISABLED=true

python3 fastchat/test/inference_single_image_MLLMSD.py \
    --save_dir MineDreamer_7B_inference_output_20000 \
    --ckpt_dir output_mllm/MineDreamer_7B \
    --input_image dataset/mllm_diffusion_dataset/wood_v0/images/start/mine_block_birch_log_8_x_22_start_frame_302.jpg \
    --text_prompt "chop a tree" \
    --steps 20000
