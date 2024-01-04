import os
import torch
from typing import Optional
import transformers
from transformers import Trainer
import torch.nn as nn



# 保存hugging face model for hugging face trainer
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """ Collects the state dict and dump to disk. """
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

# 为了分布式训练保存去掉"module"
def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).
    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model

# 继承Trainer，重写save
class LLMSDTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # Save the model
        _state_dict = state_dict
        if _state_dict is None:
            # Only save the model itself if we are using distributed training -> 也就是多卡训练的时候才会走这里
            model_to_save = unwrap_model(self.model)
            _state_dict = model_to_save.state_dict()

        # Stable Diffusion unet training... -> Unet checkpoint save
        weight_to_save_unet = {}
        unet_keys_to_match = ['unet', 'dinov2_proj']
        for k, v in _state_dict.items():
            if any(key_match in k for key_match in unet_keys_to_match):
                weight_to_save_unet[k] = v

        # len(weight_to_save.keys())=463, len(weight_to_save_unet.keys())=814, len(weight_to_save_LLM.keys())=578
        # checkpoint saving -> save_steps + training_finish -> 现在更新一版存模型方法
        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        current_step = int(current_folder[len('checkpoint-'):])
        if current_folder.startswith('checkpoint-'):
            # Unet checkpoint save
            unet_folder = os.path.join(parent_folder, "unet-%d" % current_step)
            os.makedirs(unet_folder, exist_ok=True)
            torch.save(weight_to_save_unet, os.path.join(unet_folder, 'adapter_model.bin'))
            # optimizer and scheduler...
            now_folder = parent_folder + '/' + current_folder
            os.makedirs(now_folder, exist_ok=True)
        else:
            # Unet checkpoint save
            unet_folder = os.path.join(output_dir, "unet-last")
            os.makedirs(unet_folder, exist_ok=True)
            torch.save(weight_to_save_unet, os.path.join(unet_folder, 'adapter_model.bin'))