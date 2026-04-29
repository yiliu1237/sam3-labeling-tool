from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np
from typing import List
from pathlib import Path
import zipfile
import tempfile
import json
import uuid

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from api.models import (
    TextPromptRequest,
    RefinePromptRequest,
    VideoSegmentRequest,
    SegmentationResult,
    BoxFileSegmentationResult,
    Point,
    BBox,
    MaskEditRequest
)
from services.sam3_service import get_sam3_service, parse_bounding_box_file
from services.storage import get_storage_service

router = APIRouter(prefix="/api/segment", tags=["segmentation"])


@router.post("/upload-folder")
async def upload_folder(files: List[UploadFile] = File(...), folder_name: str = Form("uploaded_folder")):
    """Upload a local folder's files to a server temp directory. Returns the server path."""
    storage = get_storage_service()
    dest = storage.temp_path / folder_name
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        content = await f.read()
        (dest / Path(f.filename).name).write_bytes(content)
    return {"folder_path": str(dest)}


@router.get("/bbox-test", response_class=HTMLResponse, include_in_schema=False)
async def bbox_test_page():
    """Minimal browser page for testing image upload + YOLO bbox-file segmentation."""
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SAM3 BBox Test</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 24px; background: #f7f7f8; color: #111; }
    .wrap { max-width: 900px; margin: 0 auto; background: #fff; padding: 24px; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,.08); }
    h1 { margin-top: 0; }
    label { display: block; font-weight: 600; margin: 16px 0 8px; }
    input, button { font: inherit; }
    input[type="file"], input[type="number"] { width: 100%; }
    button { margin-top: 20px; padding: 10px 16px; border: 0; border-radius: 8px; background: #0b74de; color: #fff; cursor: pointer; }
    button:disabled { background: #9bbce0; cursor: wait; }
    pre { white-space: pre-wrap; word-break: break-word; background: #111; color: #eee; padding: 12px; border-radius: 8px; overflow: auto; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    img { max-width: 100%; border: 1px solid #ddd; border-radius: 8px; margin-top: 12px; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>SAM3 YOLO BBox Test</h1>
    <p>Upload an image, upload a YOLO label file like <code>5 0.464323 0.805556 0.125521 0.203704</code>, then run bbox-only segmentation.</p>

    <label for="imageFile">Image file</label>
    <input id="imageFile" type="file" accept="image/*" />

    <label for="bboxFile">BBox label file</label>
    <input id="bboxFile" type="file" accept=".txt,.csv,.json" />

    <label for="threshold">Confidence threshold</label>
    <input id="threshold" type="number" min="0" max="1" step="0.05" value="0.5" />

    <button id="runBtn">Run Test</button>

    <div class="row">
      <div>
        <label>Preview</label>
        <img id="preview" alt="preview" />
      </div>
      <div>
        <label>Visualization</label>
        <img id="viz" alt="visualization" />
      </div>
    </div>

    <label>Response</label>
    <pre id="output">Waiting.</pre>
  </div>

  <script>
    const imageFile = document.getElementById('imageFile');
    const bboxFile = document.getElementById('bboxFile');
    const threshold = document.getElementById('threshold');
    const runBtn = document.getElementById('runBtn');
    const output = document.getElementById('output');
    const preview = document.getElementById('preview');
    const viz = document.getElementById('viz');

    imageFile.addEventListener('change', () => {
      const file = imageFile.files[0];
      if (!file) return;
      preview.src = URL.createObjectURL(file);
    });

    runBtn.addEventListener('click', async () => {
      const image = imageFile.files[0];
      const bbox = bboxFile.files[0];
      if (!image || !bbox) {
        output.textContent = 'Please choose both an image file and a bbox label file.';
        return;
      }

      runBtn.disabled = true;
      output.textContent = 'Uploading image...';
      viz.removeAttribute('src');

      try {
        const uploadForm = new FormData();
        uploadForm.append('file', image);

        const uploadResp = await fetch('/api/segment/upload', {
          method: 'POST',
          body: uploadForm,
        });
        const uploadData = await uploadResp.json();
        if (!uploadResp.ok) throw new Error(JSON.stringify(uploadData));

        output.textContent = 'Running bbox segmentation...';
        const segForm = new FormData();
        segForm.append('image_id', uploadData.file_id);
        segForm.append('bbox_file', bbox);
        segForm.append('confidence_threshold', threshold.value);

        const segResp = await fetch('/api/segment/image/boxes-file', {
          method: 'POST',
          body: segForm,
        });
        const segData = await segResp.json();
        output.textContent = JSON.stringify(segData, null, 2);
        if (!segResp.ok) throw new Error(segData.detail || 'Segmentation failed');

        if (segData.visualization_path) {
          const parts = segData.visualization_path.split('/data/outputs/');
          if (parts.length === 2) {
            viz.src = '/outputs/' + parts[1] + '?t=' + Date.now();
          }
        }
      } catch (err) {
        output.textContent = String(err);
      } finally {
        runBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload an image or video file"""
    try:
        storage = get_storage_service()

        # Read file content
        content = await file.read()

        # Save file
        file_id, file_path = await storage.save_upload(content, file.filename)

        # Get file type
        file_type = "image" if file.content_type.startswith("image") else "video"

        return {
            "file_id": file_id,
            "file_path": file_path,
            "file_type": file_type,
            "filename": file.filename
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/image/text", response_model=SegmentationResult)
async def segment_image_with_text(request: TextPromptRequest):
    """Segment an image using text prompt (supports comma-separated multiple prompts)"""
    try:
        sam3_service = get_sam3_service()
        storage = get_storage_service()

        # Get image path
        image_path = storage.get_upload_path(request.image_id)
        if not image_path:
            raise HTTPException(status_code=404, detail=f"Image {request.image_id} not found")

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Split prompt by comma for multi-label support
        prompts = [p.strip() for p in request.prompt.split(',') if p.strip()]

        # Collect all masks, boxes, scores, and labels from all prompts
        all_masks = []
        all_boxes = []
        all_scores = []
        all_labels = []

        # Process each prompt separately
        for idx, prompt in enumerate(prompts):
            # Use unique image_id for each prompt to avoid state conflicts
            unique_image_id = f"{request.image_id}_prompt{idx}" if len(prompts) > 1 else request.image_id

            # Segment with SAM 3
            result = sam3_service.segment_image_with_text(
                image=image,
                prompt=prompt,
                image_id=unique_image_id,
                confidence_threshold=request.confidence_threshold
            )

            # Convert tensors to lists for JSON serialization
            for mask in result["masks"]:
                # Convert tensor to numpy if needed
                if hasattr(mask, 'cpu'):
                    mask = mask.cpu().numpy()
                elif not isinstance(mask, np.ndarray):
                    mask = np.array(mask)

                # Ensure mask is 2D (height x width)
                if mask.ndim != 2:
                    print(f"Warning: mask has unexpected dimensions: {mask.shape}")
                    # Try to reshape if possible
                    if mask.ndim == 1:
                        size = int(np.sqrt(mask.shape[0]))
                        if size * size == mask.shape[0]:
                            mask = mask.reshape(size, size)

                # Convert to int and then to list (keeping 2D structure)
                mask_int = mask.astype(int).tolist()
                all_masks.append(mask_int)

            # Add boxes, scores, and labels for this prompt
            all_boxes.extend([[float(x) for x in box] for box in result["boxes"]])
            all_scores.extend([float(score) for score in result["scores"]])
            all_labels.extend([prompt] * len(result["scores"]))

        return {
            "masks": all_masks,
            "boxes": all_boxes,
            "scores": all_scores,
            "labels": all_labels
        }

    except Exception as e:
        import traceback
        print(f"ERROR in segment_image_with_text: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


@router.post("/image/boxes-file", response_model=BoxFileSegmentationResult)
async def segment_image_with_box_file(
    image_id: str = Form(...),
    bbox_file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5),
):
    """Segment one object per provided box and clip each mask to the box area."""
    try:
        sam3_service = get_sam3_service()
        storage = get_storage_service()

        image_path = storage.get_upload_path(image_id)
        if not image_path:
            raise HTTPException(status_code=404, detail=f"Image {image_id} not found")

        image = Image.open(image_path).convert("RGB")
        bbox_content = await bbox_file.read()

        try:
            box_entries = parse_bounding_box_file(
                bbox_content,
                bbox_file.filename or "",
                image_width=image.width,
                image_height=image.height,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        result = sam3_service.segment_image_with_boxes(
            image=image,
            box_entries=box_entries,
            image_id=f"{image_id}_boxes",
            confidence_threshold=confidence_threshold,
        )

        visualization_filename = (
            f"{image_id}_bbox_segmentation_{uuid.uuid4().hex[:8]}.png"
        )
        visualization_path = storage.outputs_path / visualization_filename
        result["visualization"].save(visualization_path)

        return {
            "masks": [mask.astype(int).tolist() for mask in result["masks"]],
            "boxes": [[float(value) for value in box] for box in result["boxes"]],
            "predicted_boxes": [
                [float(value) for value in box] for box in result["predicted_boxes"]
            ],
            "scores": [float(score) for score in result["scores"]],
            "labels": result["labels"],
            "visualization_path": str(visualization_path),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"BBox file segmentation failed: {str(e)}",
        )


@router.post("/image/seg-file", response_model=SegmentationResult)
async def segment_image_with_seg_file(
    image_id: str = Form(...),
    seg_file: UploadFile = File(...),
):
    """
    Rasterize a YOLO segmentation file (polygon points) into binary masks.
    No SAM3 inference — the masks come directly from the polygon annotations.
    """
    try:
        storage = get_storage_service()
        image_path = storage.get_upload_path(image_id)
        if not image_path:
            raise HTTPException(status_code=404, detail=f"Image {image_id} not found")

        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size

        content = (await seg_file.read()).decode("utf-8")

        masks, boxes, scores, labels = [], [], [], []

        for line_num, line in enumerate(content.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:  # class_id + at least 3 points (6 coords)
                raise HTTPException(
                    status_code=400,
                    detail=f"Line {line_num}: YOLO seg needs class_id + at least 3 x,y pairs"
                )
            class_id = parts[0]
            coords = list(map(float, parts[1:]))
            if len(coords) % 2 != 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Line {line_num}: odd number of coordinates"
                )

            # Denormalize polygon points
            points = [
                (int(coords[i] * img_w), int(coords[i + 1] * img_h))
                for i in range(0, len(coords), 2)
            ]

            # Rasterize polygon to binary mask
            mask_img = Image.new("L", (img_w, img_h), 0)
            ImageDraw.Draw(mask_img).polygon(points, fill=255)
            mask_arr = np.array(mask_img) // 255

            # Compute bounding box from polygon
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

            masks.append(mask_arr.tolist())
            boxes.append([float(x1), float(y1), float(x2), float(y2)])
            scores.append(1.0)
            labels.append(class_id)

        return {"masks": masks, "boxes": boxes, "scores": scores, "labels": labels}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Seg file processing failed: {str(e)}")


@router.post("/image/refine", response_model=SegmentationResult)
async def refine_segmentation(request: RefinePromptRequest):
    """Refine segmentation with points or boxes"""
    try:
        sam3_service = get_sam3_service()

        result = None

        # Refine with points
        if request.points:
            points = [(p.x, p.y) for p in request.points]
            labels = [p.label for p in request.points]

            result = sam3_service.refine_with_points(
                image_id=request.image_id,
                points=points,
                labels=labels,
                mask_id=request.mask_id
            )

        # Refine with boxes
        elif request.boxes:
            box = request.boxes[0]  # Use first box
            result = sam3_service.refine_with_box(
                image_id=request.image_id,
                box=(box.x1, box.y1, box.x2, box.y2)
            )

        else:
            raise HTTPException(status_code=400, detail="Must provide points or boxes")

        # Convert tensors to lists for JSON serialization
        # Convert masks to integers (0 or 1) - handle both numpy arrays and tensors
        masks_converted = []
        for mask in result["masks"]:
            # Convert tensor to numpy if needed
            if hasattr(mask, 'cpu'):
                mask = mask.cpu().numpy()
            elif not isinstance(mask, np.ndarray):
                mask = np.array(mask)

            # Ensure mask is 2D (height x width)
            if mask.ndim != 2:
                print(f"Warning: mask has unexpected dimensions: {mask.shape}")

            # Convert to int and then to list (keeping 2D structure)
            mask_int = mask.astype(int).tolist()
            masks_converted.append(mask_int)

        return {
            "masks": masks_converted,
            "boxes": [[float(x) for x in box] for box in result["boxes"]],
            "scores": [float(score) for score in result["scores"]]
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Refinement failed: {str(e)}")


@router.post("/video/text")
async def segment_video_with_text(request: VideoSegmentRequest):
    """Segment a video using text prompt"""
    try:
        import torch
        sam3_service = get_sam3_service()
        storage = get_storage_service()

        # Get video path
        video_path = storage.get_upload_path(request.video_id)
        if not video_path:
            raise HTTPException(status_code=404, detail=f"Video {request.video_id} not found")

        # Segment with SAM 3
        result = sam3_service.segment_video_with_text(
            video_path=video_path,
            prompt=request.prompt,
            video_id=request.video_id,
            frame_index=request.frame_index,
            confidence_threshold=request.confidence_threshold
        )

        # Parse video segmentation outputs
        outputs = result.get('outputs', {})
        session_id = result.get('session_id')
        prompt = result.get('prompt', '')

        # Extract masks and convert to same format as image segmentation
        # Video output structure: out_binary_masks, out_boxes_xywh, out_probs, out_obj_ids
        out_masks = outputs.get('out_binary_masks')  # Shape: (num_objects, H, W)
        out_boxes = outputs.get('out_boxes_xywh')    # Shape: (num_objects, 4) in xywh format
        out_probs = outputs.get('out_probs')         # Shape: (num_objects,)
        out_obj_ids = outputs.get('out_obj_ids')     # Shape: (num_objects,)

        print(f"DEBUG: Video masks shape: {out_masks.shape if out_masks is not None else None}")
        print(f"DEBUG: Video boxes shape: {out_boxes.shape if out_boxes is not None else None}")
        print(f"DEBUG: Video probs shape: {out_probs.shape if out_probs is not None else None}")

        # Convert to image-like format for frontend compatibility
        masks_list = []
        boxes_list = []
        scores_list = []

        if out_masks is not None:
            # Convert masks from numpy array to list of 2D arrays
            for i in range(out_masks.shape[0]):
                mask = out_masks[i]  # (H, W)
                masks_list.append(mask.astype(int).tolist())

        if out_boxes is not None:
            # Convert boxes from xywh to xyxy format
            for i in range(out_boxes.shape[0]):
                x, y, w, h = out_boxes[i]
                # Convert xywh to xyxy (x1, y1, x2, y2)
                x1, y1 = x, y
                x2, y2 = x + w, y + h
                boxes_list.append([float(x1), float(y1), float(x2), float(y2)])

        if out_probs is not None:
            scores_list = [float(score) for score in out_probs]

        # Return in same format as image segmentation for frontend compatibility
        return {
            "masks": masks_list,
            "boxes": boxes_list,
            "scores": scores_list,
            "session_id": session_id,
            "prompt": prompt,
            "num_instances": len(masks_list)
        }

    except Exception as e:
        import traceback
        print(f"ERROR in segment_video_with_text: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Video segmentation failed: {str(e)}")


@router.get("/video/info/{video_id}")
async def get_video_info(video_id: str):
    """Get video metadata (total frames, fps, duration, dimensions)"""
    try:
        import cv2
        storage = get_storage_service()

        # Get video path
        video_path = storage.get_upload_path(video_id)
        if not video_path:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

        # Open video with OpenCV
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        cap.release()

        return {
            "total_frames": total_frames,
            "fps": fps,
            "duration": duration,
            "width": width,
            "height": height
        }

    except Exception as e:
        import traceback
        print(f"ERROR in get_video_info: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Video info retrieval failed: {str(e)}")


@router.get("/video/frame/{video_id}")
async def get_video_frame(video_id: str, frame_index: int = 0):
    """Extract and return a specific frame from a video as an image"""
    try:
        import cv2
        storage = get_storage_service()

        # Get video path
        video_path = storage.get_upload_path(video_id)
        if not video_path:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

        # Open video with OpenCV
        cap = cv2.VideoCapture(video_path)

        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # Read frame
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise HTTPException(status_code=400, detail=f"Could not read frame {frame_index}")

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        frame_image = Image.fromarray(frame_rgb)

        # Convert to bytes
        img_byte_arr = io.BytesIO()
        frame_image.save(img_byte_arr, format='JPEG', quality=85)
        img_byte_arr.seek(0)

        # Return with caching headers to prevent flickering
        return StreamingResponse(
            img_byte_arr,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
                "ETag": f"{video_id}-{frame_index}"  # Unique identifier for this frame
            }
        )

    except Exception as e:
        import traceback
        print(f"ERROR in get_video_frame: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Frame extraction failed: {str(e)}")


@router.delete("/clear/{file_id}")
async def clear_file_state(file_id: str, file_type: str = "image"):
    """Clear cached state for a file"""
    try:
        sam3_service = get_sam3_service()

        if file_type == "image":
            sam3_service.clear_image_state(file_id)
        else:
            sam3_service.clear_video_session(file_id)

        return {"message": f"State cleared for {file_id}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")


@router.post("/save_masks")
async def save_masks(request: dict):
    """
    Save instance segmentation masks in multiple formats

    Expected request format:
    {
        "image_id": "xxx",
        "output_path": "/path/to/save/folder",
        "masks": [[mask1], [mask2], ...],  # 2D arrays
        "scores": [0.9, 0.8, ...],
        "boxes": [[x1,y1,x2,y2], ...]
    }

    Saves:
    1. overlay_visualization.png - Colored overlay on original image
    2. instances.png - Instance ID map (0=bg, 1=inst0, 2=inst1, ...)
    3. combined_mask.png - All instances merged into binary mask
    4. masks/mask_XX.png - Individual binary masks
    """
    try:
        storage = get_storage_service()

        # Extract request data
        image_id = request.get("image_id")
        output_path = request.get("output_path")
        masks = request.get("masks", [])
        scores = request.get("scores", [])
        boxes = request.get("boxes", [])

        print(f"🔍 Backend received save request:")
        print(f"   image_id: {image_id}")
        print(f"   output_path: {output_path}")
        print(f"   output_path type: {type(output_path)}")
        print(f"   masks count: {len(masks)}")

        if not image_id or not output_path or not masks:
            raise HTTPException(status_code=400, detail="Missing required fields")

        # Get original image
        image_path = storage.get_upload_path(image_id)
        if not image_path:
            raise HTTPException(status_code=404, detail=f"Image {image_id} not found")

        original_image = Image.open(image_path).convert('RGB')
        width, height = original_image.size

        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(exist_ok=True)

        # Convert masks to numpy arrays
        masks_np = []
        for mask in masks:
            mask_array = np.array(mask, dtype=np.uint8)
            # Resize mask to match original image size if needed
            if mask_array.shape != (height, width):
                mask_pil = Image.fromarray(mask_array * 255)
                mask_pil = mask_pil.resize((width, height), Image.NEAREST)
                mask_array = (np.array(mask_pil) > 128).astype(np.uint8)
            masks_np.append(mask_array)

        # 1. Create overlay visualization with boundaries (colored masks on original image)
        overlay = original_image.copy().convert('RGBA')
        overlay_layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))

        for idx, mask in enumerate(masks_np):
            # Generate unique color for this instance (same as frontend)
            hue = (idx * 360 / max(len(masks_np), 1)) % 360
            saturation = 70 + (idx % 3) * 10
            lightness = 50 + (idx % 2) * 10

            # Convert HSL to RGB
            from colorsys import hls_to_rgb
            r, g, b = hls_to_rgb(hue/360, lightness/100, saturation/100)
            fill_color = (int(r*255), int(g*255), int(b*255), 76)  # 30% opacity for fill
            border_color = (int(r*255), int(g*255), int(b*255), 255)  # Full opacity for border

            # Create colored mask (transparent fill)
            colored_mask = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            colored_mask_array = np.array(colored_mask)
            colored_mask_array[mask > 0] = fill_color

            # Detect edges (4-connectivity)
            mask_h, mask_w = mask.shape
            edges = np.zeros_like(mask, dtype=bool)
            for y in range(mask_h):
                for x in range(mask_w):
                    if mask[y, x] > 0:
                        # Check 4-connected neighbors
                        is_edge = False
                        if x == 0 or x == mask_w - 1 or y == 0 or y == mask_h - 1:
                            is_edge = True
                        elif (mask[y-1, x] == 0 or mask[y+1, x] == 0 or
                              mask[y, x-1] == 0 or mask[y, x+1] == 0):
                            is_edge = True
                        if is_edge:
                            edges[y, x] = True

            # Draw bright colored borders
            colored_mask_array[edges] = border_color
            colored_mask = Image.fromarray(colored_mask_array)
            overlay_layer = Image.alpha_composite(overlay_layer, colored_mask)

        overlay = Image.alpha_composite(overlay, overlay_layer)
        overlay.convert('RGB').save(output_dir / "overlay_visualization.png")

        # 2. Create instances map (pixel value = instance ID)
        instances_map = np.zeros((height, width), dtype=np.uint8)
        for idx, mask in enumerate(masks_np):
            instances_map[mask > 0] = idx + 1  # 0=background, 1=inst0, 2=inst1, ...
        Image.fromarray(instances_map).save(output_dir / "instances.png")

        # 3. Create combined binary mask (all instances merged)
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        for mask in masks_np:
            combined_mask = np.maximum(combined_mask, mask)
        Image.fromarray(combined_mask * 255).save(output_dir / "combined_mask.png")

        # 4. Save individual binary masks
        for idx, mask in enumerate(masks_np):
            mask_filename = f"mask_{idx:02d}.png"
            Image.fromarray(mask * 255).save(masks_dir / mask_filename)

        return {
            "message": "Masks saved successfully",
            "output_path": str(output_dir),
            "files_created": {
                "overlay_visualization": str(output_dir / "overlay_visualization.png"),
                "instances_map": str(output_dir / "instances.png"),
                "combined_mask": str(output_dir / "combined_mask.png"),
                "individual_masks": len(masks_np)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save failed: {str(e)}")


@router.post("/download_masks")
async def download_masks(request: dict):
    """
    Download instance segmentation masks as a ZIP file

    Returns a ZIP containing:
    - overlay_visualization.png - Colored masks with boundaries
    - overlay_with_labels.png - Colored masks with ID labels at center
    - combined_mask.png - All instances merged into binary mask
    - masks/mask_XX.png - Individual binary masks
    - metadata.json - Prompt, scores, boxes, and other metadata
    """
    try:
        storage = get_storage_service()

        # Extract request data
        image_id = request.get("image_id")
        masks = request.get("masks", [])
        scores = request.get("scores", [])
        boxes = request.get("boxes", [])
        prompt = request.get("prompt", "")  # Get prompt if available

        if not image_id or not masks:
            raise HTTPException(status_code=400, detail="Missing required fields")

        # Get original image - remove any _promptN suffix that was added for multi-label
        original_image_id = image_id.split('_prompt')[0]
        image_path = storage.get_upload_path(original_image_id)
        if not image_path:
            raise HTTPException(status_code=404, detail=f"Image {original_image_id} not found")

        original_image = Image.open(image_path).convert('RGB')
        width, height = original_image.size

        # Create ZIP file in memory
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Convert masks to numpy arrays
            masks_np = []
            for mask in masks:
                mask_array = np.array(mask, dtype=np.uint8)
                if mask_array.shape != (height, width):
                    mask_pil = Image.fromarray(mask_array * 255)
                    mask_pil = mask_pil.resize((width, height), Image.NEAREST)
                    mask_array = (np.array(mask_pil) > 128).astype(np.uint8)
                masks_np.append(mask_array)

            # 1. Create overlay visualization with boundaries
            overlay = original_image.copy().convert('RGBA')
            overlay_layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))

            for idx, mask in enumerate(masks_np):
                hue = (idx * 360 / max(len(masks_np), 1)) % 360
                saturation = 70 + (idx % 3) * 10
                lightness = 50 + (idx % 2) * 10

                from colorsys import hls_to_rgb
                r, g, b = hls_to_rgb(hue/360, lightness/100, saturation/100)
                fill_color = (int(r*255), int(g*255), int(b*255), 76)  # 30% opacity for fill
                border_color = (int(r*255), int(g*255), int(b*255), 255)  # Full opacity for border

                # Create colored mask (transparent fill)
                colored_mask = Image.new('RGBA', (width, height), (0, 0, 0, 0))
                colored_mask_array = np.array(colored_mask)
                colored_mask_array[mask > 0] = fill_color

                # Detect edges (4-connectivity)
                mask_h, mask_w = mask.shape
                edges = np.zeros_like(mask, dtype=bool)
                for y in range(mask_h):
                    for x in range(mask_w):
                        if mask[y, x] > 0:
                            # Check 4-connected neighbors
                            is_edge = False
                            if x == 0 or x == mask_w - 1 or y == 0 or y == mask_h - 1:
                                is_edge = True
                            elif (mask[y-1, x] == 0 or mask[y+1, x] == 0 or
                                  mask[y, x-1] == 0 or mask[y, x+1] == 0):
                                is_edge = True
                            if is_edge:
                                edges[y, x] = True

                # Draw bright colored borders
                colored_mask_array[edges] = border_color
                colored_mask = Image.fromarray(colored_mask_array)
                overlay_layer = Image.alpha_composite(overlay_layer, colored_mask)

            overlay = Image.alpha_composite(overlay, overlay_layer)

            # Save overlay visualization to ZIP
            overlay_bytes = io.BytesIO()
            overlay.convert('RGB').save(overlay_bytes, format='PNG')
            zip_file.writestr('overlay_visualization.png', overlay_bytes.getvalue())

            # 2. Create overlay with labels (ID at mask center)
            overlay_labeled = overlay.copy()
            draw = ImageDraw.Draw(overlay_labeled)

            # Try to use a nice font with larger size, fallback to default if not available
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 60)
            except:
                try:
                    font = ImageFont.truetype("arial.ttf", 60)
                except:
                    font = ImageFont.load_default()

            # Calculate mask centers and draw labels
            for idx, mask in enumerate(masks_np):
                # Find mask center
                ys, xs = np.where(mask > 0)
                if len(ys) > 0:
                    center_y = int(np.mean(ys))
                    center_x = int(np.mean(xs))

                    # Get darker color for this instance (lower lightness)
                    hue = (idx * 360 / max(len(masks_np), 1)) % 360
                    saturation = 80 + (idx % 3) * 5  # Higher saturation
                    lightness = 30 + (idx % 2) * 5   # Much darker (was 50)
                    from colorsys import hls_to_rgb
                    r, g, b = hls_to_rgb(hue/360, lightness/100, saturation/100)
                    text_color = (int(r*255), int(g*255), int(b*255))

                    # Draw text directly without background box
                    text = str(idx)
                    draw.text((center_x, center_y), text, fill=text_color, font=font, anchor="mm")

            # Save labeled overlay to ZIP
            overlay_labeled_bytes = io.BytesIO()
            overlay_labeled.save(overlay_labeled_bytes, format='PNG')
            zip_file.writestr('overlay_with_labels.png', overlay_labeled_bytes.getvalue())

            # 3. Create combined binary mask
            combined_mask = np.zeros((height, width), dtype=np.uint8)
            for mask in masks_np:
                combined_mask = np.maximum(combined_mask, mask)

            combined_bytes = io.BytesIO()
            Image.fromarray(combined_mask * 255).save(combined_bytes, format='PNG')
            zip_file.writestr('combined_mask.png', combined_bytes.getvalue())

            # 4. Save individual binary masks
            for idx, mask in enumerate(masks_np):
                mask_filename = f"masks/mask_{idx:02d}.png"
                mask_bytes = io.BytesIO()
                Image.fromarray(mask * 255).save(mask_bytes, format='PNG')
                zip_file.writestr(mask_filename, mask_bytes.getvalue())

            # 5. Create metadata JSON
            # Split prompts if comma-separated
            prompts_list = [p.strip() for p in prompt.split(',') if p.strip()] if prompt else []

            metadata = {
                "image_id": image_id,
                "prompts": prompts_list if prompts_list else ["N/A"],
                "num_instances": len(masks_np),
                "image_size": {
                    "width": width,
                    "height": height
                },
                "instances": []
            }

            # Add per-instance metadata
            # Note: If we have labels from segmentation result, use them
            # Otherwise just use the full prompt string
            labels = request.get("labels", [])

            for idx in range(len(masks_np)):
                instance_data = {
                    "id": idx,
                    "label": labels[idx] if idx < len(labels) else (prompts_list[0] if prompts_list else "N/A"),
                    "score": float(scores[idx]) if idx < len(scores) else None,
                    "box": [float(x) for x in boxes[idx]] if idx < len(boxes) else None,
                    "area": int(np.sum(masks_np[idx] > 0))
                }
                metadata["instances"].append(instance_data)

            # Save metadata JSON to ZIP
            metadata_json = json.dumps(metadata, indent=2)
            zip_file.writestr('metadata.json', metadata_json)

        # Reset buffer position
        zip_buffer.seek(0)

        # Return ZIP file as download
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=segmentation_masks.zip"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@router.post("/image/edit-mask", response_model=SegmentationResult)
async def edit_mask(request: MaskEditRequest):
    """
    Edit a mask using brush or eraser strokes

    Operations:
    - 'add': Add pixels to an existing mask (brush)
    - 'remove': Remove pixels from an existing mask (eraser)
    - 'create': Create a new mask from strokes (brush on 'new')
    """
    try:
        from scipy.ndimage import binary_dilation
        from scipy.ndimage import distance_transform_edt
        import cv2

        sam3_service = get_sam3_service()
        storage = get_storage_service()

        # Get image path and dimensions
        image_path = storage.get_upload_path(request.image_id)
        if not image_path:
            raise HTTPException(status_code=404, detail=f"Image {request.image_id} not found")

        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        # Get current segmentation state from SAM3 service
        # This would need to be stored in the service - for now we'll create a simple approach
        # In production, you'd want to maintain state in the service

        # For this implementation, we'll work with the masks directly
        # The frontend should send the current masks along with the edit request
        # For now, we'll create a helper function to rasterize strokes

        def rasterize_strokes(strokes, brush_size, img_width, img_height):
            """Convert stroke paths into a binary mask"""
            mask = np.zeros((img_height, img_width), dtype=np.uint8)

            for stroke in strokes:
                if len(stroke) < 2:
                    continue

                # Convert stroke points to numpy array
                points = np.array(stroke, dtype=np.int32)

                # Draw lines between consecutive points with thickness
                for i in range(len(points) - 1):
                    pt1 = tuple(points[i])
                    pt2 = tuple(points[i + 1])
                    cv2.line(mask, pt1, pt2, 1, thickness=brush_size)

                # Also draw circles at each point for smoother strokes
                for point in points:
                    cv2.circle(mask, tuple(point), brush_size // 2, 1, -1)

            return mask

        # Create stroke mask from the provided strokes
        stroke_mask = rasterize_strokes(request.strokes, request.brush_size, width, height)

        # Since we need to maintain mask state, we'll return a simple response
        # In a full implementation, this would integrate with SAM3 service state management

        # For now, return the stroke mask as a demonstration
        # The frontend will need to handle merging this with existing masks
        return {
            "masks": [stroke_mask.astype(int).tolist()],
            "boxes": [[0.0, 0.0, float(width), float(height)]],
            "scores": [1.0],
            "labels": [f"edited_mask_{request.mask_id}"]
        }

    except Exception as e:
        import traceback
        print(f"ERROR in edit_mask: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Mask editing failed: {str(e)}")
