import torch
from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from PIL import Image
import numpy as np
import cv2
import os
import os.path as osp
from argparse import ArgumentParser
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.console import Console

console = Console()

"""Segformer face parsing module:

Labels for the segmentation
"id2label": {
    "0": "background",
    "1": "skin",
    "2": "nose",
    "3": "eye_g",
    "4": "l_eye",
    "5": "r_eye",
    "6": "l_brow",
    "7": "r_brow",
    "8": "l_ear",
    "9": "r_ear",
    "10": "mouth",
    "11": "u_lip",
    "12": "l_lip",
    "13": "hair",
    "14": "hat",
    "15": "ear_r",
    "16": "neck_l",
    "17": "neck",
    "18": "cloth"
  }

Mapping to SyncTalk format:
- Head (RED: 255,0,0): classes 1-6, 8-16 (skin, nose, eyes, brows, ears, mouth, lips, hair, hat, accessories)
- Neck (GREEN: 0,255,0): class 17 (neck)
- Torso (BLUE: 0,0,255): class 18 (cloth)
- Background (WHITE: 255,255,255): class 0 (background)

Face-only mask (for optical flow):
- Face (RED: 255,0,0): classes 1,2,4,5,6,7,10,11,12 (skin, nose, eyes, brows, mouth, lips)
"""

def create_parsing_maps(parsing_anno, img_size):
    """Create both full body segmentation and face-specific masks"""
    vis_parsing_anno = parsing_anno.astype(np.uint8)

    # Initialize with white background (255, 255, 255)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + np.array([255, 255, 255])
    vis_parsing_anno_color_face = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + np.array([255, 255, 255])

    # Full body segmentation mapping
    # Head parts (RED): skin, nose, eyes, brows, ears, mouth, lips, hair, hat, accessories
    head_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    for cls in head_classes:
        index = np.where(vis_parsing_anno == cls)
        vis_parsing_anno_color[index[0], index[1], :] = np.array([255, 0, 0])

    # Neck (GREEN)
    neck_index = np.where(vis_parsing_anno == 17)
    vis_parsing_anno_color[neck_index[0], neck_index[1], :] = np.array([0, 255, 0])

    # Torso/Cloth (BLUE)
    torso_index = np.where(vis_parsing_anno == 18)
    vis_parsing_anno_color[torso_index[0], torso_index[1], :] = np.array([0, 0, 255])

    # Resize to original image size
    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    full_mask = cv2.resize(vis_parsing_anno_color, img_size, interpolation=cv2.INTER_NEAREST)

    # Face-specific mask (for optical flow)
    # Core face parts only: skin, nose, eyes, brows, mouth, lips
    face_classes = [1, 2, 4, 5, 6, 7, 10, 11, 12]
    for cls in face_classes:
        index = np.where(vis_parsing_anno == cls)
        vis_parsing_anno_color_face[index[0], index[1], :] = np.array([255, 0, 0])

    # Add padding below face boundary (similar to original implementation)
    pad = 5
    vis_parsing_anno_color_face = vis_parsing_anno_color_face.astype(np.uint8)
    face_part = (vis_parsing_anno_color_face[..., 0] == 255) & (vis_parsing_anno_color_face[..., 1] == 0) & (vis_parsing_anno_color_face[..., 2] == 0)

    if np.any(face_part):
        face_coords = np.stack(np.nonzero(face_part), axis=-1)
        sorted_inds = np.lexsort((-face_coords[:, 0], face_coords[:, 1]))
        sorted_face_coords = face_coords[sorted_inds]
        u, uid, ucnt = np.unique(sorted_face_coords[:, 1], return_index=True, return_counts=True)
        bottom_face_coords = sorted_face_coords[uid] + np.array([pad, 0])
        rows, cols, _ = vis_parsing_anno_color_face.shape

        # Clip coordinates to image bounds
        bottom_face_coords[:, 0] = np.clip(bottom_face_coords[:, 0], 0, rows - 1)

        y_min = np.min(bottom_face_coords[:, 1])
        y_max = np.max(bottom_face_coords[:, 1])

        # Add padding in middle sections (2nd and 3rd quarters)
        y_range = y_max - y_min
        height_per_part = y_range // 4

        start_y_part1 = y_min + height_per_part
        end_y_part1 = start_y_part1 + height_per_part

        start_y_part2 = end_y_part1
        end_y_part2 = start_y_part2 + height_per_part

        for coord in bottom_face_coords:
            x, y = coord
            start_x = max(x - pad, 0)
            end_x = min(x + pad, rows)
            if start_y_part1 <= y <= end_y_part1 or start_y_part2 <= y <= end_y_part2:
                vis_parsing_anno_color_face[start_x:end_x, y] = [255, 0, 0]

        # Apply Gaussian blur for smooth transitions
        vis_parsing_anno_color_face = cv2.GaussianBlur(vis_parsing_anno_color_face, (9, 9), cv2.BORDER_DEFAULT)

    face_mask = cv2.resize(vis_parsing_anno_color_face, img_size, interpolation=cv2.INTER_NEAREST)

    return full_mask, face_mask


def evaluate(respth='./res/test_res', dspth='./data'):
    """
    Process images with Segformer model and create parsing masks

    Args:
        respth: Output directory for parsing results
        dspth: Input directory containing images
    """
    Path(respth).mkdir(parents=True, exist_ok=True)

    # Determine device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    console.print(f'[bold blue][INFO][/bold blue] Loading Segformer model on [yellow]{device}[/yellow]')

    # Load Segformer model
    image_processor = SegformerImageProcessor.from_pretrained(
        "jonathandinu/face-parsing"
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        "jonathandinu/face-parsing"
    )
    model.to(device)
    model.eval()

    # Get list of images
    image_paths = [f for f in os.listdir(dspth) if f.endswith('.jpg') or f.endswith('.png')]

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Processing with Segformer", total=len(image_paths))

        with torch.no_grad():
            for image_name in image_paths:
                image_path = osp.join(dspth, image_name)
                image = Image.open(image_path)
                ori_size = image.size  # (width, height)

                # Run inference
                inputs = image_processor(images=image, return_tensors="pt").to(device)
                outputs = model(**inputs)
                logits = outputs.logits  # shape (batch_size, num_labels, ~height/4, ~width/4)

                # Resize output to match input image dimensions
                upsampled_logits = nn.functional.interpolate(
                    logits,
                    size=(ori_size[1], ori_size[0]),  # (height, width)
                    mode="bilinear",
                    align_corners=False
                )

                # Get label masks
                parsing = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

                # Create both full and face masks
                full_mask, face_mask = create_parsing_maps(parsing, ori_size)

                # Save masks with consistent naming (matching original pipeline)
                # Extract frame number from filename
                frame_num = int(image_name.split('.')[0])
                output_name = f"{frame_num}.png"

                cv2.imwrite(osp.join(respth, output_name), full_mask)
                cv2.imwrite(osp.join(respth, output_name.replace('.png', '_face.png')), face_mask)

                progress.advance(task)

    console.print(f'[bold green][INFO][/bold green] ✓ Segformer processing completed successfully')


if __name__ == "__main__":
    parser = ArgumentParser(description="Segformer for face parsing")
    parser.add_argument('--respath', type=str, default='./result/', help='result path for label')
    parser.add_argument('--imgpath', type=str, default='./imgs/', help='path for input images')

    args = parser.parse_args()
    evaluate(respth=args.respath, dspth=args.imgpath)
