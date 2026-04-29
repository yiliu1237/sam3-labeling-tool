import os
import uuid
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw
from datetime import datetime

from .sam3_service import get_sam3_service, parse_bounding_box_file
from .storage import get_storage_service

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov'}

COLORS = [
    (255, 56,  56),
    (255, 157, 151),
    (255, 112, 31),
    (255, 178, 29),
    (207, 210, 49),
    (72,  249, 10),
    (146, 204, 23),
    (61,  219, 134),
    (26,  147, 52),
    (0,   212, 187),
    (44,  153, 168),
    (0,   194, 255),
    (52,  69,  147),
    (100, 115, 255),
    (0,   24,  236),
    (132, 56,  255),
    (82,  0,   133),
    (203, 56,  255),
    (255, 149, 200),
    (255, 55,  199),
]


def _mask_to_yolo_seg(mask: np.ndarray, class_id: int, img_w: int, img_h: int) -> Optional[str]:
    """Convert a binary mask to a YOLO segmentation line."""
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Use the largest contour
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None
    points = contour.reshape(-1, 2)
    # Normalize
    norm = points / np.array([img_w, img_h], dtype=np.float32)
    norm = np.clip(norm, 0.0, 1.0)
    coords = ' '.join(f'{x:.6f} {y:.6f}' for x, y in norm)
    return f'{class_id} {coords}'


def _draw_overlay(image: Image.Image, masks: List[np.ndarray], alpha: float = 0.45) -> Image.Image:
    """Draw semi-transparent colored masks over the image."""
    overlay = image.convert('RGBA')
    for idx, mask in enumerate(masks):
        color = COLORS[idx % len(COLORS)]
        layer = Image.new('RGBA', overlay.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        mask_bool = mask > 0
        ys, xs = np.where(mask_bool)
        if len(xs) == 0:
            continue
        for x, y in zip(xs, ys):
            draw.point((int(x), int(y)), fill=(*color, int(255 * alpha)))
        overlay = Image.alpha_composite(overlay, layer)
    return overlay.convert('RGB')


class BatchProcessor:

    def __init__(self):
        self.jobs: Dict[str, Any] = {}

    def create_job(
        self,
        input_folder: str,
        output_folder: str,
        label_folder: Optional[str],
        prompts: List[str],
        confidence_threshold: float,
        process_videos: bool,
    ) -> str:
        job_id = str(uuid.uuid4())
        extensions = IMAGE_EXTENSIONS | (VIDEO_EXTENSIONS if process_videos else set())
        files = [
            str(p) for p in Path(input_folder).iterdir()
            if p.suffix.lower() in extensions
        ]
        files.sort()

        self.jobs[job_id] = {
            'job_id': job_id,
            'status': 'pending',
            'progress': 0.0,
            'total_files': len(files),
            'processed_files': 0,
            'current_file': None,
            'error': None,
            'input_folder': input_folder,
            'output_folder': output_folder,
            'label_folder': label_folder,
            'prompts': prompts,
            'confidence_threshold': confidence_threshold,
            'files': files,
        }
        return job_id

    def process_job(self, job_id: str):
        if job_id not in self.jobs:
            raise ValueError(f'Job {job_id} not found')

        job = self.jobs[job_id]
        job['status'] = 'processing'

        output_dir = Path(job['output_folder'])
        masks_dir   = output_dir / 'masks'
        overlays_dir = output_dir / 'overlays'
        labels_dir  = output_dir / 'labels'
        for d in (masks_dir, overlays_dir, labels_dir):
            d.mkdir(parents=True, exist_ok=True)

        sam3 = get_sam3_service()
        use_boxes = bool(job['label_folder'])

        try:
            for file_idx, file_path in enumerate(job['files']):
                stem = Path(file_path).stem
                job['current_file'] = Path(file_path).name

                image = Image.open(file_path).convert('RGB')
                img_w, img_h = image.size
                image_id = f'batch_{job_id}_{file_idx}'

                masks: List[np.ndarray] = []
                class_ids: List[int] = []

                if use_boxes:
                    # --- box-prompted mode ---
                    label_path = Path(job['label_folder']) / f'{stem}.txt'
                    if not label_path.exists():
                        job['processed_files'] = file_idx + 1
                        job['progress'] = (file_idx + 1) / max(job['total_files'], 1)
                        continue

                    with open(label_path, 'rb') as f:
                        box_entries = parse_bounding_box_file(
                            f.read(), label_path.name, img_w, img_h
                        )

                    result = sam3.segment_image_with_boxes(
                        image=image,
                        box_entries=box_entries,
                        image_id=image_id,
                        confidence_threshold=job['confidence_threshold'],
                    )
                    masks = result['masks']
                    class_ids = [
                        int(e.get('label', 0)) if str(e.get('label', '0')).isdigit() else 0
                        for e in box_entries
                    ]

                else:
                    # --- text-prompted mode ---
                    for prompt_idx, prompt in enumerate(job['prompts']):
                        result = sam3.segment_image_with_text(
                            image=image,
                            prompt=prompt,
                            image_id=f'{image_id}_{prompt_idx}',
                            confidence_threshold=job['confidence_threshold'],
                        )
                        for m in result['masks']:
                            masks.append(m.cpu().numpy() if hasattr(m, 'cpu') else np.array(m))
                            class_ids.append(prompt_idx)
                        sam3.clear_image_state(f'{image_id}_{prompt_idx}')

                if not masks:
                    job['processed_files'] = file_idx + 1
                    job['progress'] = (file_idx + 1) / max(job['total_files'], 1)
                    continue

                # --- save mask PNG (combined) ---
                combined = np.zeros((img_h, img_w), dtype=np.uint8)
                for m in masks:
                    arr = m.cpu().numpy() if hasattr(m, 'cpu') else np.array(m)
                    combined = np.maximum(combined, (arr > 0).astype(np.uint8) * 255)
                Image.fromarray(combined).save(masks_dir / f'{stem}_mask.png')

                # --- save overlay PNG ---
                mask_arrays = []
                for m in masks:
                    arr = m.cpu().numpy() if hasattr(m, 'cpu') else np.array(m)
                    mask_arrays.append((arr > 0).astype(np.uint8))
                overlay = _draw_overlay(image, mask_arrays)
                overlay.save(overlays_dir / f'{stem}_overlay.png')

                # --- save YOLO seg txt ---
                yolo_lines = []
                for m, cid in zip(mask_arrays, class_ids):
                    line = _mask_to_yolo_seg(m, cid, img_w, img_h)
                    if line:
                        yolo_lines.append(line)
                if yolo_lines:
                    (labels_dir / f'{stem}.txt').write_text('\n'.join(yolo_lines) + '\n')

                sam3.clear_image_state(image_id)
                job['processed_files'] = file_idx + 1
                job['progress'] = (file_idx + 1) / max(job['total_files'], 1)

            job['status'] = 'completed'
            job['progress'] = 1.0

        except Exception as e:
            job['status'] = 'failed'
            job['error'] = str(e)
            import traceback; traceback.print_exc()

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        if job_id not in self.jobs:
            raise ValueError(f'Job {job_id} not found')
        job = self.jobs[job_id]
        return {
            'job_id': job['job_id'],
            'status': job['status'],
            'progress': job['progress'],
            'total_files': job['total_files'],
            'processed_files': job['processed_files'],
            'current_file': job['current_file'],
            'error': job['error'],
        }


_batch_processor = None


def get_batch_processor() -> BatchProcessor:
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchProcessor()
    return _batch_processor
