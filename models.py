import torch
import os
from mmdet.apis import DetInferencer
from segment_anything import SamPredictor, sam_model_registry
from PIL import ImageOps, Image


def load_sam_model(model_type="vit_h", checkpoint="sam_vit_h_4b8939.pth", device="cuda"):
    """
    Load Segment Anything Model (SAM)
    """
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam)
    return predictor

def load_groundingdino_model(config_path="./configs/grounding_dino_swin-t_finetune_8xb2_20e_o365.py", checkpoint_path="./checkpoints/groundingdino_swint_ogc.pth", device="cuda"):
    """
    Load MM Grounding DINO model from mmdet
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU instead")
    

    model = DetInferencer(
        model=config_path,
        weights=checkpoint_path,
        device=device
    )

    return model

def load_clip_model(device="cuda", clip_model_path=None):
    """
    Load fine-tuned OpenCLIP model
    """
    import open_clip
    model_name = "ViT-B-16"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=clip_model_path
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.to(device)
    return model, preprocess, tokenizer
