#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch
from tqdm import tqdm
import argparse
import numpy as np
from openai import OpenAI
import nltk

from models import load_groundingdino_model, load_sam_model, load_clip_model
from processing import process_support_set, process_query_with_clip, process_image, visualize_result
from utils import parse_response, log_gpt_output, create_gpt_log_summary
from call_gpt import call_chatgpt

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')


count=0

def main():
    parser = argparse.ArgumentParser(description="Process VizWiz dataset using GroundingDINO and CLIP")
    parser.add_argument("--query_dir", type=str, default="./dataset/query/query_images",
                        help="Directory path for query images")
    parser.add_argument("--support_dir", type=str, default="./dataset/support/support_images",
                        help="Directory path for support images")
    parser.add_argument("--support_json", type=str, default="./dataset/support_set.json",
                        help="Path to support set JSON information file")
    parser.add_argument("--image_id", type=str, default=None,
                        help="Image id")
    parser.add_argument("--json_path", type=str, default="./dataset/dev_set_images_info.json",
                        help="Path to query set JSON information file")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory")
    parser.add_argument("--config_path", type=str, 
                        default="./configs/grounding_dino_swin-t_finetune_8xb2_20e_viz.py",
                        help="MM Grounding DINO configuration file path")
    parser.add_argument("--checkpoint_path", type=str, 
                        default=None,
                        help="MM Grounding DINO checkpoint file path")
    parser.add_argument("--clip_model_path", type=str, default=None, help="CLIP model path")
    parser.add_argument("--sam_model_type", type=str, default="vit_h", help="SAM model type")
    parser.add_argument("--sam_checkpoint", type=str, default="sam_vit_h_4b8939.pth", help="SAM checkpoint path")
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
    parser.add_argument("--gpt_log_summary", type=str, default=None, help="Whether to log GPT output summary")
    parser.add_argument("--fix_no_object", type=bool, default=True, help="Whether to fix no object detection")
    parser.add_argument("--refer_detection", type=str, default=None, help="Refer to detection results")
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