#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch
import cv2
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from mmdet.apis import DetInferencer
from mmengine.config import Config
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
from torchvision import ops
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import mmcv
from openai import OpenAI
import openai
import time
import base64
import re
import nltk
from dotenv import load_dotenv

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
os.environ["OPENAI_API_KEY"] = openai_api_key
count=0

def log_gpt_output(image_path, response_text, parsed_text, output_dir, log_dir=None):

    if log_dir is None:
        log_dir = os.path.join(output_dir, "gpt_logs")
    os.makedirs(log_dir, exist_ok=True)
    

    file_name = os.path.basename(image_path)
    file_name_no_ext = os.path.splitext(file_name)[0]
    

    log_file_path = os.path.join(log_dir, f"{file_name_no_ext}_gpt_output.txt")
    

    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write(f"Image: {file_name}\n")
        f.write(f"Full GPT Response:\n")
        f.write("-" * 80 + "\n")
        f.write(response_text)
        f.write("\n" + "-" * 80 + "\n")
        f.write(f"Parsed Categories: {parsed_text}\n")
        f.write("-" * 80 + "\n")
    
    print(f"GPT output for {file_name} logged to {log_file_path}")


def call_chatgpt(image_path, model='gpt-4.1-2025-04-14', temperature=0., top_p=1.0, echo=False):
    client = OpenAI()
    instruction = (
        """
        Task: Analyze the image and identify specific private objects, providing both category and location information.
        
        16 categories to identify:
        [local newspaper], [bank statement], [bills or receipt], [business card], [condom box], 
        [credit or debit card], [doctors prescription], [letters with address], [medical record document], 
        [pregnancy test], [empty pill bottle], [tattoo sleeve], [transcript], [mortgage or investment report], 
        [condom with plastic bag], [pregnancy test box]
        
        Analysis steps:
        1. Carefully examine the entire image to identify all potential objects
        2. For each object, determine its category and approximate position (e.g., top-left, center, bottom-right)
        3. Assess your confidence level for each classification (high, medium, low)
        
        Identification guidelines:
        • Empty pill bottle [empty pill bottle]: Cylindrical container, typically with white cap, translucent or opaque plastic material
        • Condom with plastic bag [condom with plastic bag]: Small sealed transparent bag containing foil-wrapped item
        • Bills or receipt [bills or receipt]: Rectangular paper with text blocks and numbers, typically neatly arranged
        • Mortgage or investment report [mortgage or investment report]: Formal document with bold headers and financial data tables
        • Transcript [transcript]: Multi-column academic-style document with dense text and numerical entries
        • Tattoo sleeve [tattoo sleeve]: Colored fabric or sleeve, often with flame or tribal patterns
        • Credit or debit card [credit or debit card]: Rectangular plastic card, metallic or colorful, with embedded logo/text
        • Business card [business card]: Small rectangular card printed with contact information and logo
        • Pregnancy test [pregnancy test]: Slim white plastic device with result window
        • Pregnancy test box [pregnancy test box]: Vertical rectangular box with product branding and test device illustration
        • Doctor's prescription [doctors prescription]: Medical form with structured layout and identification marks
        • Condom box [condom box]: Small cardboard box, typically with commercial packaging design and small text
        • Medical record document [medical record document]: Multi-page document with medical charts or diagrams
        • Letters with address [letters with address]: Folded document with typed address block and formal formatting
        • Local newspaper [local newspaper]: Full-page print layout with headlines, columns, and image thumbnails
        • Bank statement [bank statement]: Document with transaction tables, charts, and bank logo formatting
        
        Output format:
        1. [category name] - Position: (describe position) - Confidence: (high/medium/low) - Features: (briefly describe identifying features)
        2. [category name] - Position: (describe position) - Confidence: (high/medium/low) - Features: (briefly describe identifying features)
        (continue listing if more objects are present...)
        <output>category name,category name,category name</output>
        
        If uncertain but possible categories exist, include them with low confidence. If no target categories can be identified in the image, respond with:
        <output>No objects matching the given categories could be identified</output>
        """
    )
    
    # Encode the image in base64
    with open(image_path, "rb") as f:
        image_bytes = f.read()
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        
    # Prepare the prompt
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
            ]
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=prompt,
            top_p=top_p,
            n=1
        )
        return response.choices[0].message.content.strip()
    except openai.RateLimitError as e:
        time.sleep(5)
        return call_chatgpt(image_path, model, temperature, top_p, echo)
    except Exception as e:
        raise RuntimeError(f"Failed to call GPT API: {e}")

def parse_response(response_text):
    # Use regex to extract content between <output> and </output>
    match = re.search(r'<output>(.*?)</output>', response_text, re.DOTALL)
    
    if match:
        raw_output = match.group(1).strip()
        
        # if raw_output == "No objects matching the given categories could be identified":
        #     return "bank statement, bills or receipt, business card, condom box, credit or debit card, doctors prescription, letters with address, medical record document, pregnancy test, empty pill bottle, tattoo sleeve, transcript, mortgage or investment report, condom with plastic bag, pregnancy test box."
        
        # Split by comma and clean up each item
        items = [item.strip().strip('[]') for item in raw_output.split(',')]
        return ', '.join(items)
    else:
        return "No output found"

def mask_to_coco_polygon(args,mask):
    """
    Convert a binary mask to COCO format polygons.
    """
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        contour = contour.reshape(-1, 2)
        if "realesrgan" in args.query_dir:
            contour = contour / args.scale
        else:
            contour = contour
        polygon = contour.flatten().tolist()
        if len(polygon) >= 6:
            polygons.append(polygon)
    return polygons

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."

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

def load_clip_model(device="cuda", clip_model_path="/media/hcchen/backup/vizwiz/open_clip/logs/2025_04_18-00_10_45-model_ViT-B-16-lr_5e-06-b_64-j_4-p_amp/checkpoints/epoch_32.pt"):
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

def process_support_set(model, preprocess, support_dir, support_json, device="cuda"):
    """
    handle support set
    """
    with open(support_json, 'r') as f:
        support_data = json.load(f)
    
    categories = {}
    for annotation in support_data['annotations']:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        
        if image_id not in categories:
            categories[image_id] = []
        categories[image_id].append(category_id)
    

    category_features = {}
    category_counts = {}

    for image_info in support_data['images']:
        image_id = image_info['id']
        file_name = image_info['file_name']
        img_path = os.path.join(support_dir, file_name)
        
        if not os.path.exists(img_path):
            print(f"Warning: Support image not found {img_path}")
            continue
        
        if image_id not in categories:
            continue
        

        image = preprocess(ImageOps.exif_transpose(Image.open(img_path))).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        for category_id in categories[image_id]:
            if category_id not in category_features:
                category_features[category_id] = []
                category_counts[category_id] = 0
            
            category_features[category_id].append(image_features.cpu())
            category_counts[category_id] += 1

    prototypes = {}
    for category_id, features in category_features.items():
        if features:
            stacked_features = torch.cat(features, dim=0)
            prototype = torch.mean(stacked_features, dim=0)
            prototypes[category_id] = prototype
    
    return prototypes, category_counts

def process_query_with_clip(clip_model, preprocess, prototypes, image_path, categories_dict, device="cuda", top_k=3):
    """
    Process a single image and return detection results (using Hugging Face Transformers)
    """

    image = preprocess(ImageOps.exif_transpose(Image.open(image_path))).unsqueeze(0).to(device)
    with torch.no_grad():
        query_features = clip_model.encode_image(image)
        query_features = query_features / query_features.norm(dim=-1, keepdim=True)
    

    similarities = {}
    for category_id, prototype in prototypes.items():

        similarity = F.cosine_similarity(query_features.cpu(), prototype.unsqueeze(0))
        similarities[category_id] = similarity.item()
    

    id_to_name = {v: k for k, v in categories_dict.items()}
    clip_results = {id_to_name[cat_id]: score for cat_id, score in similarities.items() if cat_id in id_to_name}

    top_k_results = sorted(clip_results.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return ', '.join([item[0] for item in top_k_results])

def process_image(args, model, predictor, image_path, text_prompt, image_id, categories_dict, box_threshold=0.35, text_threshold=0.25, device="cuda"):
    """
    use mm GroundingDINO and CLIP to process image
    """
    image = mmcv.imread(image_path,channel_order='rgb')
    print(image_id)

    prompt = preprocess_caption(text_prompt)
    
    label_texts = [cls.strip() for cls in text_prompt.split(',')]
    
    texts = [prompt]
    
    result = model(
        inputs=image,
        texts=texts
    )
    
    if isinstance(result, list):
        result = result[0]

    detected_results = []
    
    if isinstance(result, dict):
        if 'predictions' in result:
            pred_data = result['predictions']
            if isinstance(pred_data, list):

                if len(pred_data) == 0:
                    return []
                    
                if isinstance(pred_data[0], dict):

                    first_pred = pred_data[0]
                    boxes = first_pred.get('bboxes', [])
                    scores = first_pred.get('scores', [])
                    labels = first_pred.get('labels', [])
                    
                    if not isinstance(boxes, torch.Tensor):
                        boxes = torch.tensor(boxes)
                    if not isinstance(scores, torch.Tensor):
                        scores = torch.tensor(scores)
    
    if len(boxes) == 0 or len(scores) == 0:
        return []


    detected_results = []

    if len(boxes) == 0 or len(scores) == 0:
        return []
    

    top_indices = torch.argsort(scores, descending=True)[:100]

    keep_indices_relative = ops.nms(boxes[top_indices], scores[top_indices], iou_threshold=0.8)

    final_indices = top_indices[keep_indices_relative]

    image_np = np.array(image)
    predictor.set_image(image_np)

    for i in final_indices:
        box = boxes[i]
        score = scores[i]
        label = labels[i]
        
        if score < box_threshold:
            continue
        if score < text_threshold:
            continue

        if i < len(label_texts):
            label = label_texts[i]
        else:
            label = text_prompt.split(',')[0].strip()

        

        if label not in categories_dict:
            for cls in text_prompt.split(','):
                if label in cls:
                    label = cls.strip()
                    break

            if label not in categories_dict:
                continue
        

        input_box = box.cpu().numpy()
        input_box = np.expand_dims(input_box, axis=0)
            
        masks, mask_scores, _ = predictor.predict(
            box=input_box,
            point_coords=None,
            point_labels=None,
            multimask_output=False
        )
            
        polygons = mask_to_coco_polygon(args,masks[0])
        
        if "realesrgan" in args.query_dir:
            box_x1, box_y1, box_x2, box_y2 = box.tolist()
            fix_box = [box_x1/args.scale, box_y1/args.scale, (box_x2-box_x1)/args.scale, (box_y2-box_y1)/args.scale]
        else:
            box_x1, box_y1, box_x2, box_y2 = box.tolist()
            fix_box = [box_x1, box_y1, box_x2-box_x1, box_y2-box_y1]
        
        detected_results.append({
            "image_id": image_id,
            "score": score.item(),
            "category_id": categories_dict[label],
            "area": int((box_x2 - box_x1) * (box_y2 - box_y1)) / args.scale ** 2,
            "bbox": fix_box,
            "segmentation": polygons
        })
        
        print(f"Detected {image_id} : {label} with score {score.item()} at {fix_box}")
    
    return detected_results

def visualize_result(args,image_path, results, output_path,categories_dict):
    """
    Visualize detection results
    """
    image = ImageOps.exif_transpose(Image.open(image_path))
    image_np = np.array(image)
    
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

    plt.figure(figsize=(16,10))
    plt.imshow(image_np)
    ax = plt.gca()
    
    for i, result in enumerate(results):
        color = COLORS[i % len(COLORS)]
        score = result["score"]
        label = result["category_id"]
        category_name = [k for k, v in categories_dict.items() if v == label]
        bbox = result["bbox"]
        if "realesrgan" in args.query_dir:
            bbox = [bbox[0] * args.scale, bbox[1] * args.scale, bbox[2] * args.scale, bbox[3] * args.scale]
        
        ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                 fill=False, color=color, linewidth=3))
        
        label = f'{category_name}: {score:0.2f}'
        ax.text(bbox[0], bbox[1], label, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
        
        if "segmentation" in result and result["segmentation"]:
            mask = np.zeros((image_np.shape[0], image_np.shape[1]))
            for polygon in result["segmentation"]:
                if "realesrgan" in args.query_dir:
                    polygon = polygon * args.scale
                pts = np.array(polygon).reshape(-1, 2)
                cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
            plt.imshow(mask, alpha=0.5)
    
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.0)
    plt.close()

def create_gpt_log_summary(processed_files, output_dir, log_dir=None):

    if log_dir is None:
        log_dir = os.path.join(output_dir, "gpt_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # use json file name as summary file name
    summary_file_path = os.path.join(log_dir, "gpt_summary.json")
    sum={}
    for idx, (file_name, text_prompt) in enumerate(processed_files, 1):
        sum[file_name] = text_prompt
    with open(summary_file_path, 'w', encoding='utf-8') as f:   
        json.dump(sum, f, ensure_ascii=False, indent=2)
    
    print(f"GPT processing summary saved to {summary_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Process VizWiz dataset using GroundingDINO and CLIP")
    parser.add_argument("--query_dir", type=str, default="/media/hcchen/backup/vizwiz/dataset/query/query_images",
                        help="Directory path for query images")
    parser.add_argument("--support_dir", type=str, default="/media/hcchen/backup/vizwiz/dataset/support/support_images",
                        help="Directory path for support images")
    parser.add_argument("--support_json", type=str, default="/media/hcchen/backup/vizwiz/dataset/support_set.json",
                        help="Path to support set JSON information file")
    parser.add_argument("--image_id", type=str, default=None,
                        help="Image id")
    parser.add_argument("--json_path", type=str, default="/media/hcchen/backup/vizwiz/dataset/dev_set_images_info.json",
                        help="Path to query set JSON information file")
    parser.add_argument("--output_dir", type=str, default="/media/hcchen/backup/vizwiz/LLM2Seg/devtest",
                        help="Output directory")
    parser.add_argument("--config_path", type=str, 
                        default="./configs/grounding_dino_swin-t_finetune_8xb2_20e_viz.py",
                        help="MM Grounding DINO configuration file path")
    parser.add_argument("--checkpoint_path", type=str, 
                        default="/media/hcchen/backup/vizwiz/best_coco_bbox_mAP_epoch_13.pth",
                        help="MM Grounding DINO checkpoint file path")
    parser.add_argument("--clip_model_path", type=str, default="/media/hcchen/backup/vizwiz/open_clip/logs/2025_04_23-15_59_05-model_ViT-B-16-lr_5e-06-b_64-j_4-p_amp/checkpoints/epoch_200.pt", help="CLIP model path")
    parser.add_argument("--sam_model_type", type=str, default="vit_h", help="SAM model type")
    parser.add_argument("--sam_checkpoint", type=str, default="/media/hcchen/backup/vizwiz/LLM2Seg/sam_vit_h_4b8939.pth", help="SAM checkpoint path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--box_threshold", type=float, default=0.35, help="Detection box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="Text threshold")
    parser.add_argument("--visualize", type=bool, default=True, help="Whether to generate visualization results")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of images to process, for testing")
    parser.add_argument("--method", type=str, default="gpt", help="Method to use (clip or gpt)")
    parser.add_argument("--top_k", type=int, default=3, help="Top k results")
    parser.add_argument("--scale", type=int, default=4, help="Scale of the image")
    parser.add_argument("--gpt_log_dir", type=str, default=None, help="Directory to save GPT logs")
    parser.add_argument("--gpt_log_summary", type=str, default="/media/hcchen/backup/vizwiz/LLM2Seg/dev_v2_thinking_best/gpt_logs/gpt_summary.json", help="Whether to log GPT output summary")
    parser.add_argument("--fix_no_object", type=bool, default=True, help="Whether to fix no object detection")
    parser.add_argument("--refer_detection", type=str, default="/media/hcchen/backup/vizwiz/LLM2Seg/dev_v2_thinking_best/detection_results.json", help="Refer to detection results")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    if args.visualize:
        os.makedirs(os.path.join(args.output_dir, "visualizations"), exist_ok=True)
    

    dino_model = load_groundingdino_model(config_path=args.config_path, checkpoint_path=args.checkpoint_path, device=args.device)
    # init the visualizer(execute this block only once)

    predictor = load_sam_model(model_type=args.sam_model_type, checkpoint=args.sam_checkpoint, device=args.device)
    
    with open(args.json_path, 'r') as f:
        dataset_info = json.load(f)
    
    #make categories to dict
    categories_dict = {}
    for cat in dataset_info["categories"]:
        categories_dict[cat["name"].replace("_"," ")] = cat["id"]

    images_info = dataset_info["images"]
    if args.limit:
        images_info = images_info[:args.limit]
    elif args.image_id:
        print(f"Processing image {args.image_id}")
        for img_info in images_info:
            if img_info["id"] == int(args.image_id):
                images_info = [img_info]
                break
            

    clip_model, clip_preprocess, tokenizer = load_clip_model(device=args.device, clip_model_path=args.clip_model_path)

    print("Processing support set and extracting CLIP features...")
    prototypes, category_counts = process_support_set(
        clip_model, 
        clip_preprocess, 
        args.support_dir, 
        args.support_json, 
        device=args.device
    )
    print(f"Extracted prototypes for {len(prototypes)} categories.")
    for cat_id, count in category_counts.items():
        print(f"Category {cat_id}: {count} images")
    if args.method == "gpt":
        categories = [cat["name"].replace("_"," ") for cat in dataset_info["categories"]]
        text_prompt = ", ".join(categories)
        print(f"Categories to detect: {text_prompt}")
    elif args.method == "no_gpt":
        categories = [cat["name"].replace("_"," ") for cat in dataset_info["categories"]]
        text_prompt = ", ".join(categories)
        print(f"Categories to detect: {text_prompt}")
    elif args.method == "gpt_summary":
        categories = [cat["name"].replace("_"," ") for cat in dataset_info["categories"]]
        text_prompt = ", ".join(categories)
        #load gpt summary
        gpt_summary_path = os.path.join(args.gpt_log_summary)
        if not os.path.exists(gpt_summary_path):
            print(f"Warning: GPT summary file not found {gpt_summary_path}")
            return
        with open(gpt_summary_path, 'r') as f:
            gpt_summary = json.load(f)
        print(f"Categories to detect: {text_prompt}")
        if args.refer_detection:
            with open(args.refer_detection, 'r') as f:
                refer_detection = json.load(f)
    
    all_results = []
    
    processed_files = []
    
    for img_info in tqdm(images_info, desc="Processing images"):
        image_id = img_info["id"]
        file_name = img_info["file_name"]
        image_path = os.path.join(args.query_dir, file_name)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found {image_path}")
            continue
        
        # Call GPT to generate class text prompt
        if args.method == "gpt":
            response = call_chatgpt(image_path)
            text_prompt = parse_response(response)
            log_gpt_output(image_path, response, text_prompt, args.output_dir, args.gpt_log_dir)
            processed_files.append((file_name, text_prompt))
        elif args.method == "clip":
            text_prompt = process_query_with_clip(clip_model, clip_preprocess, prototypes, image_path, categories_dict, args.device, args.top_k)
        elif args.method == "no_gpt":
            categories = [cat["name"].replace("_"," ") for cat in dataset_info["categories"]]
            text_prompt = ", ".join(categories)
        elif args.method == "gpt_summary":
            text_prompt = gpt_summary[image_path.split("/")[-1]]
            
        print(f"Text prompt: {text_prompt}")
        
        if args.fix_no_object and args.method == "gpt_summary":
            if text_prompt == "No objects matching the given categories could be identified":
                print(f"No objects detected for {file_name}, trying to call GPT again...")
                response="Clip fix"
                text_prompt = process_query_with_clip(clip_model, clip_preprocess, prototypes, image_path, categories_dict, args.device, args.top_k)
                print(f"After clip, Text prompt: {text_prompt}")
                log_gpt_output(image_path, response, text_prompt, args.output_dir, args.gpt_log_dir)
                processed_files.append((file_name, text_prompt))
                results = process_image(
                    args,
                    dino_model, 
                    predictor,
                    image_path, 
                    text_prompt, 
                    image_id,
                    categories_dict,
                    box_threshold=args.box_threshold, 
                    text_threshold=args.text_threshold,
                    device=args.device
                )
                

                for result in results:
                    all_results.append(result)
                    
                if args.visualize:
                    output_vis_path = os.path.join(
                        args.output_dir,       
                        "visualizations", 
                        f"{os.path.splitext(file_name)[0]}_detection.jpg"
                    )
                    visualize_result(args, image_path, results, output_vis_path, categories_dict)
            else:

                if args.refer_detection:
                    image_results = [result for result in refer_detection if result["image_id"] == image_id]
                    if image_results:
                        print(f"Found {len(image_results)} existing detection results for image {image_id}")
                        all_results.extend(image_results)
                        
                    else:
                        print(f"No existing detection results found for image {image_id} in reference")
        else:
            results = process_image(
                args,
                dino_model, 
                predictor,
                image_path, 
                text_prompt, 
                image_id,
                categories_dict,
                box_threshold=args.box_threshold, 
                text_threshold=args.text_threshold,
                device=args.device
            )
            

            for result in results:
                all_results.append(result)

            if args.visualize:
                output_vis_path = os.path.join(
                    args.output_dir,       
                    "visualizations", 
                    f"{os.path.splitext(file_name)[0]}_detection.jpg"
                )
                visualize_result(args, image_path, results, output_vis_path, categories_dict)
    
    output_json_path = os.path.join(args.output_dir, "detection_results.json")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    if args.method == "gpt" and processed_files:
        create_gpt_log_summary(processed_files, args.output_dir, args.gpt_log_dir)
    
    print(f"Processing completed, results saved to {output_json_path}")

    

if __name__ == "__main__":
    main()