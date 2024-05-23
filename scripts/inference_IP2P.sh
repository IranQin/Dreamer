# srun -p AI4Good_S --gres=gpu:1 --quotatype=auto bash scripts/inference_IP2P.sh
source ~/.bashrc
conda activate smartedit
gcc --version
nvcc -V
export PYTHONPATH=$PYTHONPATH:Dreamer
export PYTHONPATH=$PYTHONPATH:Dreamer/fastchat
export PYTHONPATH=$PYTHONPATH:Dreamer/fastchat/diffusers0202_unet

wandb disabled
export WANDB_DISABLED=true


# No need to change sd_path and pipeline_path 
python3 fastchat/test/inference_single_image_IP2P.py \
    --sd_path pretrain_ckpt/sd/stable-diffusion-v1-5 \
    --pipeline_path pretrain_ckpt/instruct-pix2pix \
    --save_dir IP2P_inferece_output/input_sky \
    --pretrain_unet output/final_dataset_v1/unet-15000/adapter_model.bin \
    --input_image /mnt/petrelfs/qinyiran/SmartEdit_231226/input_sky/easy_action_sky_14_x_1000_start_frame_1825.jpg \
    --text_prompt "look at the sky" \