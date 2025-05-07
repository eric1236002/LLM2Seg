# LLM2Seg
LLM2Seg: LLM-Guided Few-Shot Object Localization with Visual Transformer

A comprehensive toolkit for detecting and segmenting privacy-sensitive objects in images, including MM Grounding DINO, SAM (Segment Anything Model), and GPT-4 Vision.

## Features

- Multi-model approach combining detection, segmentation, and vision-language models
- GPT-4 Vision integration for improved object identification
- CLIP-based similarity matching with support images
- Support for various processing methods (GPT, CLIP, or combined approaches)
- Visualization of detection and segmentation results
- Detailed logging of model outputs

## Installation

### Requirements

Install the required dependencies:

```bash
conda create -n llm2seg python=3.10
conda activate llm2seg
cd LLM2Seg
pip install -r requirements.txt
```

### Required Models

Download the following models:
- SAM model: `sam_vit_h_4b8939.pth`
(https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
- MM Grounding DINO model: `groundingdino_swint_ogc_mmdet-822d7e9d.pth`
(https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth)
- CLIP model: `ViT-B-16.pt`

## Environment Variables

This project uses environment variables to manage API keys:

- `OPENAI_API_KEY`: Your OpenAI API key for GPT-4 Vision calls.

Create a `.env` file in the project root:

```bash
echo "OPENAI_API_KEY=your_api_key_here" >> .env
```

## Usage

### Basic Usage

```bash
python process_dataset_v3.py --query_dir path/to/query/images --support_dir path/to/support/images --support_json path/to/support_info.json --json_path path/to/dataset_info.json --output_dir path/to/output
```

### Advanced Options

```bash
python process_dataset_v3.py \
  --query_dir path/to/query/images \
  --support_dir path/to/support/images \
  --support_json path/to/support_info.json \
  --json_path path/to/dataset_info.json \
  --output_dir path/to/output \
  --config_path path/to/grounding_dino_config.py \
  --checkpoint_path path/to/checkpoint.pth \
  --clip_model_path path/to/clip_model.pt \
  --sam_checkpoint path/to/sam_model.pth \
  --method gpt \
  --box_threshold 0.35 \
  --text_threshold 0.25 \

```

### Usage Examples

#### GPT Mode
```bash

python process_dataset_v3.py \
  --method gpt \
  --query_dir /path/to/query/images \
  --support_dir /path/to/support/images \
  --support_json /path/to/support_info.json \
  --json_path /path/to/dataset_info.json \
  --output_dir /path/to/output \
  --sam_checkpoint /path/to/sam_model.pth \
  --config_path ./configs/grounding_dino_config.py \
  --checkpoint_path /path/to/grounding_dino_checkpoint.pth
```

#### CLIP Mode
```bash
python process_dataset_v3.py \
  --method clip \
  --query_dir /path/to/query/images \
  --support_dir /path/to/support/images \
  --support_json /path/to/support_info.json \
  --json_path /path/to/dataset_info.json \
  --output_dir /path/to/output \
  --top_k 5 \
  --clip_model_path /path/to/clip_model.pt \
  --sam_checkpoint /path/to/sam_model.pth \
  --config_path ./configs/grounding_dino_config.py \
  --checkpoint_path /path/to/grounding_dino_checkpoint.pth
```

#### No GPT Mode
```bash
python process_dataset_v3.py \
  --method no_gpt \
  --query_dir /path/to/query/images \
  --support_dir /path/to/support/images \
  --support_json /path/to/support_info.json \
  --json_path /path/to/dataset_info.json \
  --output_dir /path/to/output \
  --sam_checkpoint /path/to/sam_model.pth \
  --config_path ./configs/grounding_dino_config.py \
  --checkpoint_path /path/to/grounding_dino_checkpoint.pth
```

#### GPT Summary Mode
```bash
python process_dataset_v3.py \
  --method gpt_summary \
  --query_dir /path/to/query/images \
  --support_dir /path/to/support/images \
  --support_json /path/to/support_info.json \
  --json_path /path/to/dataset_info.json \
  --gpt_log_summary /path/to/gpt_summary.json \
  --refer_detection /path/to/detection_results.json \
  --output_dir /path/to/output \
  --sam_checkpoint /path/to/sam_model.pth \
  --config_path ./configs/grounding_dino_config.py \
  --checkpoint_path /path/to/grounding_dino_checkpoint.pth
```

#### Limiting Images (for testing)
```bash
python process_dataset_v3.py \
  --method gpt \
  --limit 20 \
  --json_path /path/to/dataset_info.json \
  --output_dir /path/to/output \
  --sam_checkpoint /path/to/sam_model.pth \
  --config_path ./configs/grounding_dino_config.py \
  --checkpoint_path /path/to/grounding_dino_checkpoint.pth
```

#### Single Image Processing
```bash
python process_dataset_v3.py \
  --image_id 324 \
  --json_path /path/to/dataset_info.json \
  --method gpt \
  --output_dir /path/to/output \
  --sam_checkpoint /path/to/sam_model.pth \
  --config_path ./configs/grounding_dino_config.py \
  --checkpoint_path /path/to/grounding_dino_checkpoint.pth
```

### Available Methods

- `gpt`: Use GPT-4 Vision to analyze each image and generate text prompts
- `clip`: Use CLIP to match query images with support set prototypes
- `no_gpt`: Use all categories without GPT assistance
- `gpt_summary`: Use pre-generated GPT analysis from a summary file

## Parameter Description

- `--query_dir`: Directory containing query images
- `--support_dir`: Directory containing support images
- `--support_json`: JSON file with support set annotations
- `--json_path`: JSON file with dataset information
- `--output_dir`: Directory to save results
- `--config_path`: MM Grounding DINO configuration file
- `--checkpoint_path`: Path to detection model checkpoint
- `--clip_model_path`: Path to CLIP model checkpoint
- `--sam_model_type`: SAM model type (default: "vit_h")
- `--sam_checkpoint`: Path to SAM model checkpoint
- `--device`: Device to use (default: "cuda" if available)
- `--box_threshold`: Detection box threshold (default: 0.35)
- `--text_threshold`: Text threshold (default: 0.25)
- `--visualize`: Whether to generate visualizations (default: True)
- `--limit`: Limit the number of images to process (for testing)
- `--method`: Processing method (default: "gpt_summary")
- `--top_k`: Top k results for CLIP matching (default: 3)
- `--scale`: Image scale factor for Real-ESRGAN processed images
- `--gpt_log_dir`: Directory to save GPT logs
- `--fix_no_object`: Use CLIP as fallback when GPT finds no objects
- `--refer_detection`: Path to reference detection results

## Output Results

- JSON file with detection results in COCO format
- Visualizations of detected objects and segmentation masks
- GPT log files containing full responses and parsed outputs
- Summary of GPT prompts for all processed images

## Technical Details

LLM2Seg integrates the following components to perform few-shot object localization:

1. **GroundingDINO** for object detection.
2. **SAM (Segment Anything Model)** for object segmentation.
3. **Prompt Generation**:
   - **CLIP**: Computes cosine similarity between query images and support set prototypes.
   - **GPT**: Uses GPT-4 Vision to analyze images and generate descriptive prompts.
4. **Processing Pipeline**:
   1. Load models.
   2. Generate text prompts.
   3. Detect objects with GroundingDINO.
   4. Segment objects with SAM.
   5. Format results to COCO.
   6. Save visualizations and logs.

## Categories

The script is configured to detect the following 16 privacy-sensitive object categories:
- local newspaper
- bank statement
- bills or receipt
- business card
- condom box
- credit or debit card
- doctors prescription
- letters with address
- medical record document
- pregnancy test
- empty pill bottle
- tattoo sleeve
- transcript
- mortgage or investment report
- condom with plastic bag
- pregnancy test box

