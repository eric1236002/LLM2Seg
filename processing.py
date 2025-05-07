import os
import json
import numpy as np
import mmcv
import torch
import cv2
from PIL import Image, ImageOps
from torchvision import ops
import torch.nn.functional as F
from utils import preprocess_caption, mask_to_coco_polygon
import matplotlib.pyplot as plt


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
