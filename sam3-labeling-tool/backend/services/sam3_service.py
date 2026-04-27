import torch
import numpy as np
from PIL import Image, ImageColor, ImageDraw
from typing import List, Dict, Optional, Tuple, Any
import sys
import os
import json
from pathlib import Path

# Add parent directory to path to import sam3
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
from sam3.model.sam3_image_processor import Sam3Processor


BOX_OVERLAY_COLORS = [
    "#ff4d4f",
    "#52c41a",
    "#1677ff",
    "#fa8c16",
    "#13c2c2",
    "#eb2f96",
]


def _parse_box_entry(entry: Any, index: int) -> Dict[str, Any]:
    if isinstance(entry, dict):
        label = str(entry.get("label") or entry.get("name") or f"box_{index}")

        if all(key in entry for key in ("x1", "y1", "x2", "y2")):
            box = [entry["x1"], entry["y1"], entry["x2"], entry["y2"]]
        elif all(key in entry for key in ("x", "y", "w", "h")):
            x, y, w, h = entry["x"], entry["y"], entry["w"], entry["h"]
            box = [x, y, x + w, y + h]
        elif "bbox" in entry and isinstance(entry["bbox"], (list, tuple)) and len(entry["bbox"]) >= 4:
            box = list(entry["bbox"][:4])
        else:
            raise ValueError(
                "Each JSON bbox entry must contain either x1/y1/x2/y2, x/y/w/h, or bbox."
            )
    elif isinstance(entry, (list, tuple)) and len(entry) >= 4:
        box = list(entry[:4])
        label = f"box_{index}"
    else:
        raise ValueError("Each bbox entry must be an object or a 4-value list.")

    try:
        box = [float(value) for value in box]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Box {index} contains non-numeric coordinates.") from exc

    return {"box": box, "label": label}


def parse_bounding_box_file(
    file_content: bytes,
    filename: str,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> List[Dict[str, Any]]:
    suffix = Path(filename or "").suffix.lower()
    content = file_content.decode("utf-8-sig").strip()
    if not content:
        raise ValueError("Bounding-box file is empty.")

    if suffix == ".json":
        payload = json.loads(content)
        if isinstance(payload, dict):
            payload = payload.get("boxes") or payload.get("bboxes") or payload.get("annotations")
        if not isinstance(payload, list):
            raise ValueError("JSON bbox file must contain a list or a top-level 'boxes' list.")
        boxes = [_parse_box_entry(entry, index) for index, entry in enumerate(payload)]
    elif suffix in {".txt", ".csv"}:
        boxes = []
        for line_number, line in enumerate(content.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = [part for part in stripped.replace(",", " ").split() if part]
            if len(parts) < 4:
                raise ValueError(
                    f"Line {line_number} must contain at least 4 values: x1 y1 x2 y2 [label]."
                )

            # Support YOLO detection labels:
            # class_id center_x center_y width height
            # where cx/cy/w/h are normalized to [0, 1].
            if len(parts) == 5:
                try:
                    class_id = parts[0]
                    cx, cy, w, h = [float(value) for value in parts[1:5]]
                except ValueError as exc:
                    raise ValueError(
                        f"Line {line_number} contains non-numeric YOLO bbox values."
                    ) from exc

                if (
                    image_width is not None
                    and image_height is not None
                    and all(0.0 <= value <= 1.0 for value in (cx, cy, w, h))
                ):
                    x1 = (cx - (w / 2.0)) * image_width
                    y1 = (cy - (h / 2.0)) * image_height
                    x2 = (cx + (w / 2.0)) * image_width
                    y2 = (cy + (h / 2.0)) * image_height
                    box = [x1, y1, x2, y2]
                    label = str(class_id)
                    boxes.append({"box": box, "label": label})
                    continue

            try:
                box = [float(value) for value in parts[:4]]
            except ValueError as exc:
                raise ValueError(
                    f"Line {line_number} contains non-numeric bbox coordinates."
                ) from exc
            label = " ".join(parts[4:]) if len(parts) > 4 else f"box_{len(boxes)}"
            boxes.append({"box": box, "label": label})
    else:
        raise ValueError("Unsupported bbox file type. Use .json, .txt, or .csv.")

    if not boxes:
        raise ValueError("No bounding boxes were found in the file.")

    return boxes


def _clamp_xyxy_box(box: List[float], image_width: int, image_height: int) -> List[float]:
    x1, y1, x2, y2 = box
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = min(max(x1, 0.0), float(image_width))
    x2 = min(max(x2, 0.0), float(image_width))
    y1 = min(max(y1, 0.0), float(image_height))
    y2 = min(max(y2, 0.0), float(image_height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bounding box after clamping: {box}")
    return [x1, y1, x2, y2]


def _xyxy_to_normalized_cxcywh(box: List[float], image_width: int, image_height: int) -> List[float]:
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    return [
        ((x1 + x2) / 2.0) / image_width,
        ((y1 + y2) / 2.0) / image_height,
        width / image_width,
        height / image_height,
    ]


def _compute_box_iou(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0.0 else 0.0


def _clip_mask_to_box(mask: np.ndarray, box: List[float]) -> np.ndarray:
    clipped_mask = np.zeros_like(mask, dtype=np.uint8)
    x1, y1, x2, y2 = box
    left = int(np.floor(x1))
    top = int(np.floor(y1))
    right = int(np.ceil(x2))
    bottom = int(np.ceil(y2))
    clipped_mask[top:bottom, left:right] = mask[top:bottom, left:right].astype(np.uint8)
    return clipped_mask


class SAM3Service:
    """Service class for SAM 3 model inference"""

    def __init__(self, device: str = None):
        """Initialize SAM 3 models"""
        if device is None:
            # Prefer CUDA, fallback to MPS on macOS, then CPU
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"Initializing SAM 3 on device: {self.device}")

        try:
            # Try to find local checkpoint first, then fall back to environment variable or HuggingFace
            default_checkpoint_path = os.path.join(
                os.path.dirname(__file__), '../../../checkpoints/sam3/sam3.pt'
            )

            if os.path.exists(default_checkpoint_path):
                checkpoint_path = default_checkpoint_path
                load_from_hf = False
                print(f"Using local checkpoint: {checkpoint_path}")
            else:
                # Fall back to environment variable or HuggingFace
                checkpoint_path = os.environ.get("SAM3_CHECKPOINT_PATH", None)
                load_from_hf = checkpoint_path is None
                if checkpoint_path:
                    print(f"Using checkpoint from environment: {checkpoint_path}")
                else:
                    print("No local checkpoint found, will download from HuggingFace")

            # Initialize image model with explicit device
            self.image_model = build_sam3_image_model(
                device=self.device,
                checkpoint_path=checkpoint_path,
                load_from_HF=load_from_hf
            )
            self.image_model = self.image_model.to(self.device)
            self.image_processor = Sam3Processor(self.image_model)

            # Initialize video predictor (only takes checkpoint_path, uses CUDA by default)
            # If load_from_hf is True, don't pass checkpoint_path and let it download from HF
            video_checkpoint = checkpoint_path if not load_from_hf else None
            self.video_predictor = build_sam3_video_predictor(
                checkpoint_path=video_checkpoint
            )

            # Store active sessions
            self.image_states = {}  # image_id -> inference_state
            self.video_sessions = {}  # video_id -> session_id

            print(f"SAM 3 models loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading SAM3 models: {e}")
            raise

    def segment_image_with_text(
        self,
        image: Image.Image,
        prompt: str,
        image_id: str,
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Segment an image using text prompt

        Args:
            image: PIL Image
            prompt: Text prompt for segmentation
            image_id: Unique identifier for the image
            confidence_threshold: Minimum confidence score

        Returns:
            Dictionary with masks, boxes, and scores
        """
        # Set the image
        inference_state = self.image_processor.set_image(image)

        # Store the state for refinement
        self.image_states[image_id] = inference_state

        # Run text prompt segmentation
        output = self.image_processor.set_text_prompt(
            state=inference_state,
            prompt=prompt
        )

        # Filter by confidence threshold
        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]

        # Filter low confidence predictions
        filtered_indices = [i for i, score in enumerate(scores) if score >= confidence_threshold]

        # Squeeze masks to remove extra dimensions (N, 1, H, W) -> list of (H, W)
        # masks is a single tensor with shape (N, 1, H, W), we need to convert to list of 2D arrays
        if len(filtered_indices) > 0:
            filtered_masks_tensor = masks[filtered_indices]  # Shape: (num_filtered, 1, H, W)
            filtered_masks = [filtered_masks_tensor[i].squeeze() for i in range(len(filtered_indices))]  # List of (H, W)
        else:
            filtered_masks = []

        filtered_boxes = [boxes[i] for i in filtered_indices]
        filtered_scores = [scores[i] for i in filtered_indices]

        return {
            "masks": filtered_masks,
            "boxes": filtered_boxes,
            "scores": filtered_scores,
            "prompt": prompt
        }

    def refine_with_points(
        self,
        image_id: str,
        points: List[Tuple[float, float]],
        labels: List[int],
        mask_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Refine segmentation with point prompts

        NOTE: SAM3 Image Processor doesn't support point prompts directly.
        We convert points to small bounding boxes as a workaround.

        Args:
            image_id: Image identifier
            points: List of (x, y) coordinates
            labels: List of labels (1 for positive, 0 for negative)
            mask_id: Optional mask ID to refine

        Returns:
            Updated segmentation results
        """
        if image_id not in self.image_states:
            raise ValueError(f"Image {image_id} not found. Please segment with text first.")

        state = self.image_states[image_id]

        # SAM3 Image Processor only supports geometric (box) prompts, not point prompts
        # Convert each point to a small box around it (±5% of image size)
        img_h = state["original_height"]
        img_w = state["original_width"]

        box_size_w = 0.05  # 5% of width
        box_size_h = 0.05  # 5% of height

        for point, label in zip(points, labels):
            # Normalize point coordinates to [0, 1]
            norm_x = point[0] / img_w
            norm_y = point[1] / img_h

            # Create box: [center_x, center_y, width, height]
            box = [norm_x, norm_y, box_size_w, box_size_h]

            # Add geometric prompt (box)
            state = self.image_processor.add_geometric_prompt(
                box=box,
                label=bool(label),  # True for positive, False for negative
                state=state
            )

        # Extract results from state
        masks_tensor = state["masks"]
        if masks_tensor.shape[0] > 0:
            masks = [masks_tensor[i].squeeze() for i in range(masks_tensor.shape[0])]
        else:
            masks = []

        return {
            "masks": masks,
            "boxes": state["boxes"].cpu().numpy().tolist(),
            "scores": state["scores"].cpu().numpy().tolist()
        }

    def refine_with_box(
        self,
        image_id: str,
        box: Tuple[float, float, float, float]
    ) -> Dict[str, Any]:
        """
        Refine segmentation with box prompt

        Args:
            image_id: Image identifier
            box: Bounding box (x1, y1, x2, y2)

        Returns:
            Updated segmentation results
        """
        if image_id not in self.image_states:
            raise ValueError(f"Image {image_id} not found. Please segment with text first.")

        state = self.image_states[image_id]

        # Add box prompt
        output = self.image_processor.add_box_prompt(
            state=state,
            box=np.array(box)
        )

        # Squeeze masks to remove extra dimensions (N, 1, H, W) -> list of (H, W)
        masks_tensor = output["masks"]
        if masks_tensor.shape[0] > 0:
            masks = [masks_tensor[i].squeeze() for i in range(masks_tensor.shape[0])]
        else:
            masks = []

        return {
            "masks": masks,
            "boxes": output["boxes"],
            "scores": output["scores"]
        }

    def segment_image_with_boxes(
        self,
        image: Image.Image,
        box_entries: List[Dict[str, Any]],
        image_id: str,
        confidence_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Segment one object per input box, and clip each mask so it only exists
        inside the provided bounding box.
        """
        image = image.convert("RGB")
        image_width, image_height = image.size
        inference_state = self.image_processor.set_image(image)
        self.image_states[image_id] = inference_state

        previous_threshold = self.image_processor.confidence_threshold
        self.image_processor.confidence_threshold = confidence_threshold

        try:
            masks = []
            input_boxes = []
            predicted_boxes = []
            scores = []
            labels = []

            visualization = image.convert("RGBA")
            visualization_array = np.array(visualization)

            for index, entry in enumerate(box_entries):
                self.image_processor.reset_all_prompts(inference_state)

                clamped_box = _clamp_xyxy_box(entry["box"], image_width, image_height)
                normalized_box = _xyxy_to_normalized_cxcywh(
                    clamped_box, image_width, image_height
                )
                output = self.image_processor.add_geometric_prompt(
                    box=normalized_box,
                    label=True,
                    state=inference_state,
                )

                current_masks = output["masks"]
                current_boxes = output["boxes"]
                current_scores = output["scores"]

                if current_masks.shape[0] == 0:
                    empty_mask = np.zeros((image_height, image_width), dtype=np.uint8)
                    masks.append(empty_mask)
                    input_boxes.append(clamped_box)
                    predicted_boxes.append(clamped_box)
                    scores.append(0.0)
                    labels.append(entry["label"])
                    continue

                best_idx = max(
                    range(current_masks.shape[0]),
                    key=lambda idx: (
                        _compute_box_iou(
                            current_boxes[idx].detach().cpu().tolist(),
                            clamped_box,
                        ),
                        float(current_scores[idx].detach().cpu().item()),
                    ),
                )

                best_mask = current_masks[best_idx].squeeze().detach().cpu().numpy()
                clipped_mask = _clip_mask_to_box(best_mask, clamped_box)
                clipped_mask = (clipped_mask > 0).astype(np.uint8)

                best_box = current_boxes[best_idx].detach().cpu().tolist()
                best_score = float(current_scores[best_idx].detach().cpu().item())

                masks.append(clipped_mask)
                input_boxes.append([float(value) for value in clamped_box])
                predicted_boxes.append([float(value) for value in best_box])
                scores.append(best_score)
                labels.append(entry["label"])

                color = ImageColor.getrgb(BOX_OVERLAY_COLORS[index % len(BOX_OVERLAY_COLORS)])
                fill_mask = clipped_mask.astype(bool)
                visualization_array[fill_mask, :3] = (
                    0.65 * visualization_array[fill_mask, :3] + 0.35 * np.array(color)
                ).astype(np.uint8)
                visualization_array[fill_mask, 3] = 255

            visualization = Image.fromarray(visualization_array, mode="RGBA")
            draw = ImageDraw.Draw(visualization)
            for index, (box, label) in enumerate(zip(input_boxes, labels)):
                color = BOX_OVERLAY_COLORS[index % len(BOX_OVERLAY_COLORS)]
                draw.rectangle(box, outline=color, width=3)
                draw.text((box[0] + 4, max(0, box[1] - 14)), label, fill=color)

            return {
                "masks": masks,
                "boxes": input_boxes,
                "predicted_boxes": predicted_boxes,
                "scores": scores,
                "labels": labels,
                "visualization": visualization.convert("RGB"),
            }
        finally:
            self.image_processor.confidence_threshold = previous_threshold

    def segment_video_with_text(
        self,
        video_path: str,
        prompt: str,
        video_id: str,
        frame_index: int = 0,
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Segment a video using text prompt

        Args:
            video_path: Path to video file or JPEG folder
            prompt: Text prompt for segmentation
            video_id: Unique identifier for the video
            frame_index: Frame to start segmentation
            confidence_threshold: Minimum confidence score

        Returns:
            Dictionary with segmentation results
        """
        # Start a session
        response = self.video_predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=video_path,
            )
        )

        session_id = response["session_id"]
        self.video_sessions[video_id] = session_id

        # Add text prompt
        response = self.video_predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=frame_index,
                text=prompt,
            )
        )

        output = response["outputs"]

        return {
            "session_id": session_id,
            "outputs": output,
            "prompt": prompt
        }

    def clear_image_state(self, image_id: str):
        """Clear stored image state to free memory"""
        if image_id in self.image_states:
            del self.image_states[image_id]

    def clear_video_session(self, video_id: str):
        """Clear video session"""
        if video_id in self.video_sessions:
            session_id = self.video_sessions[video_id]
            # End the session
            self.video_predictor.handle_request(
                request=dict(
                    type="end_session",
                    session_id=session_id
                )
            )
            del self.video_sessions[video_id]


# Global instance
_sam3_service = None


def get_sam3_service() -> SAM3Service:
    """Get or create SAM3Service singleton"""
    global _sam3_service
    if _sam3_service is None:
        _sam3_service = SAM3Service()
    return _sam3_service
