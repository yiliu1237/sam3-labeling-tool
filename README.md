# SAM 3 Labeling Tool

A web-based labeling tool powered by SAM 3 (Segment Anything Model 3) for automated image and video segmentation with text prompts and interactive refinement.

![SAM 3 Labeling Tool](https://img.shields.io/badge/SAM-3-blue)
![React](https://img.shields.io/badge/React-18-61dafb)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688)

## Features

### Dual Mode Operation

- **Single Mode**: Upload and segment individual images/videos with real-time interaction
- **Batch Mode**: Process entire folders of images/videos automatically

### Powered by SAM 3

- Text-based prompting (e.g., "plant", "person", "car")
- Open-vocabulary segmentation (270K+ concepts)
- High-quality instance segmentation
- Video object tracking

### Interactive Mask Refinement

- **Instance-Aware Editing**: Select and modify individual mask instances
- **Brush Tool**: Paint to add pixels to selected masks with adjustable brush size
- **Eraser Tool**: Remove unwanted pixels from masks with precision control
- **Undo/Redo System**: Revert changes with full history tracking (50 states)
- **Mask Management**: Delete, create, and organize multiple mask instances

### Batch Processing

- Process folders with hundreds/thousands of images
- Multiple text prompts per batch
- Progress tracking with live updates
- Export in COCO JSON, Mask PNG, or both

### Beautiful UI

- Clean, modern interface with dark/light themes
- Responsive design
- Smooth animations
- Toast notifications
- Professional color scheme

## Architecture

```
sam3-labeling-tool/
├── backend/                 # FastAPI backend
│   ├── api/
│   │   ├── routes/         # API endpoints
│   │   └── models.py       # Pydantic models
│   ├── services/
│   │   ├── sam3_service.py # SAM 3 integration
│   │   ├── batch_processor.py
│   │   └── storage.py
│   └── requirements.txt
│
├── frontend/                # React frontend
│   ├── src/
│   │   ├── components/     # UI components
│   │   ├── pages/          # Main pages
│   │   ├── api/            # API client
│   │   └── store/          # State management
│   └── package.json
│
└── data/                    # Data storage
    ├── uploads/
    ├── outputs/
    └── temp/
```

## Installation

### Prerequisites

- **Python 3.12+**
- **Node.js 18+**
- **CUDA-compatible GPU** (recommended)
- **SAM 3 model checkpoints** (HuggingFace authentication required)

### 1. Clone the Repository

```bash
cd /path/to/sam3
cd sam3-labeling-tool
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install SAM 3 (from parent directory)
pip install -e ../../
```

### 3. Frontend Setup

```bash
cd ../frontend

# Install dependencies
npm install
```

### 4. Model Setup (Choose ONE option)

SAM 3 requires model checkpoints. Choose one of these options:

#### Option A: HuggingFace 

```bash
# Install huggingface-cli if not already installed
pip install huggingface_hub

# Login to HuggingFace
huggingface-cli login
```

Then request access to the SAM 3 model at: https://huggingface.co/facebook/sam3

#### Option B: Local Checkpoint

If you have a local checkpoint file, set the environment variable:

```bash
export SAM3_CHECKPOINT_PATH=/path/to/sam3.pt
```

Or add it to your shell startup file (~/.bashrc or ~/.zshrc)

## Running the Application

### Start Backend Server

```bash
cd backend
source venv/bin/activate  # Activate virtual environment

# Run with uvicorn
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at: `http://localhost:8000`

API docs available at: `http://localhost:8000/docs`

### Start Frontend Development Server

```bash
cd frontend

# Run Vite dev server
npm run dev
```

Frontend will be available at: `http://localhost:3000`

## Usage

### Single Mode

1. Navigate to **Single Mode** tab
2. **Upload** an image or video file
3. Enter a **text prompt** (e.g., "crack", "person", "car")
4. Click **Segment** to run SAM 3 inference
5. **Review** detected mask instances in the mask list panel
6. **Refine** individual masks using interactive tools:
   - **Select** a mask instance from the list
   - **Brush Tool**: Paint to add missing pixels to the selected mask
   - **Eraser Tool**: Remove incorrectly segmented pixels from the mask
   - **Adjust** brush size (5-100px) for fine or coarse editing
   - **Undo/Redo**: Use keyboard shortcuts (Ctrl+Z / Ctrl+Shift+Z) to revert changes
   - **Delete**: Remove entire mask instances if needed
7. **Create** new mask instances manually using the brush tool
8. **Download** results as a ZIP file containing masks and metadata

### Batch Mode

1. Navigate to **Batch Mode** tab
2. Specify **Input Folder** (containing images/videos)
3. Specify **Output Folder** (for results)
4. Add **Text Prompts** (one or more)
5. Configure:
   - Export format (COCO, Mask PNG, or both)
   - Confidence threshold
   - Process videos option
6. Click **Start Batch Processing**
7. Monitor progress in real-time
8. **Download** results when complete

## Mask Refinement Workflow

The tool provides instance-aware mask editing capabilities for precise segmentation refinement:

### Initial Segmentation

Text-based prompts generate multiple mask instances automatically. SAM 3 detects all objects matching the prompt across the image, with each detection stored as a separate instance.

### Instance Selection

The mask list panel displays all detected instances with their confidence scores. Users select a specific mask to edit by clicking its entry in the list. The selected mask is highlighted on the canvas with visual emphasis.

### Brush Tool Operations

The brush tool adds pixels to the selected mask instance. Users paint directly on the canvas to extend mask boundaries or fill missed regions. Brush size is adjustable from 5 to 100 pixels for different levels of precision. The tool operates only on the selected mask, leaving other instances unmodified.

### Eraser Tool Operations

The eraser tool removes pixels from the selected mask instance. Users paint over incorrectly segmented regions to subtract them from the mask. If all pixels are removed, the mask instance is automatically deleted from the list. The eraser requires an active mask selection and cannot modify other instances.

### History Management

All mask modifications are tracked in a history buffer with a capacity of 50 states. Users can undo operations with Ctrl+Z or Cmd+Z, and redo with Ctrl+Shift+Z or Cmd+Shift+Y. The history system preserves the complete segmentation state including all mask instances.

### Manual Mask Creation

Users can create new mask instances from scratch by selecting "Create New Mask" and using the brush tool. The new mask is added to the instance list and can be refined with the same tools.

### Export Format

Downloaded results include binary mask images for each instance, bounding box coordinates, confidence scores, and metadata in JSON format. All files are packaged in a ZIP archive for convenient download.

## API Endpoints

### Segmentation

- `POST /api/segment/upload` - Upload image or video file
- `POST /api/segment/image/text` - Segment image using text prompt
- `POST /api/segment/image/boxes-file` - Segment image from a bounding-box file and clip masks to each box
- `POST /api/segment/image/edit-mask` - Edit mask using brush/eraser strokes
- `POST /api/segment/video/text` - Segment video using text prompt
- `GET /api/segment/video/frame/{file_id}` - Retrieve video frame
- `DELETE /api/segment/clear/{file_id}` - Clear cached inference state

### Batch Processing

- `POST /api/batch/process` - Create batch job
- `GET /api/batch/status/{job_id}` - Get job status

### Export

- `POST /api/export/annotations` - Export annotations
- `GET /api/export/download/{job_id}` - Download batch results

## Configuration

### Backend Settings

Edit `backend/api/main.py` to configure:

- CORS origins
- Upload limits
- Storage paths

## Bounding Box File Input

`POST /api/segment/image/boxes-file` accepts an uploaded bbox file plus an already-uploaded `image_id`.
The backend will:

- Draw the provided boxes on the image
- Run one SAM 3 box-prompted segmentation per box
- Clip each output mask so pixels outside that box are forced to `0`
- Save a visualization image under `sam3-labeling-tool/data/outputs/`

Accepted bbox file formats:

```json
[
  {"x1": 120, "y1": 80, "x2": 360, "y2": 420, "label": "object_a"},
  {"x": 420, "y": 100, "w": 140, "h": 220, "label": "object_b"}
]
```

Or plain text / CSV:

```text
120 80 360 420 object_a
420,100,560,320,object_b
```

YOLO label files are also supported directly in the standard format:

```text
5 0.464323 0.805556 0.125521 0.203704
```

This is interpreted as:

```text
class_id center_x center_y width height
```

where `center_x`, `center_y`, `width`, and `height` are normalized to the uploaded image size.

### Frontend Settings

Create `frontend/.env` file:

```env
VITE_API_URL=http://localhost:8000
```

## Tech Stack

### Backend

- **FastAPI** - Modern, fast web framework
- **SAM 3** - Meta's foundation model for segmentation
- **PyTorch** - Deep learning framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation

### Frontend

- **React 18** - UI library
- **Vite** - Build tool
- **TailwindCSS** - Utility-first CSS
- **Konva.js** - Canvas manipulation
- **Zustand** - State management
- **Framer Motion** - Animations
- **Axios** - HTTP client
- **Lucide React** - Icons

## Project Structure Details

### Backend Services

- **sam3_service.py**: Wraps SAM 3 models for inference
- **batch_processor.py**: Handles batch job creation and processing
- **storage.py**: Manages file uploads and outputs

### Frontend Components

- **Header**: Navigation and theme toggle
- **ImageUploader**: Drag and drop file upload with preview
- **SegmentationCanvas**: Interactive canvas with Konva.js for mask visualization and editing
- **ToolPanel**: Tool selection, brush size controls, and undo/redo interface
- **MaskList**: Instance list with selection, deletion, and creation controls
- **VideoPlayer**: Video frame navigation and mask overlay
- **ToastContainer**: Notification system for user feedback

### Frontend Pages

- **SingleMode**: Single image/video segmentation interface
- **BatchMode**: Batch processing interface

## Performance Tips

1. **GPU Acceleration**: Ensure CUDA is properly configured for faster inference
2. **Batch Size**: For large batches, consider processing in chunks
3. **Confidence Threshold**: Adjust to filter low-quality predictions
4. **Memory Management**: Clear file states after processing to free memory

## Troubleshooting

### Backend Issues

**Model download fails:**
```bash
# Ensure HuggingFace authentication
huggingface-cli login
```

**CUDA out of memory:**
- Reduce image resolution
- Clear cached states
- Process smaller batches

### Frontend Issues

**API connection fails:**
- Check backend is running on port 8000
- Verify CORS settings in `backend/api/main.py`

**Canvas rendering issues:**
- Update browser to latest version
- Check WebGL support

## Development

### Backend Development

```bash
cd backend

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black .
```

### Frontend Development

```bash
cd frontend

# Build for production
npm run build

# Preview production build
npm run preview
```

## Docker Deployment (Optional)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    environment:
      - CUDA_VISIBLE_DEVICES=0

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend
```

Run with:
```bash
docker-compose up
```


## License

This project uses SAM 3, which is licensed under the SAM License. See the main SAM 3 repository for details.

## Acknowledgments

- **Meta AI** for SAM 3 model
- **FastAPI** team for the excellent framework
- **React** and **Vite** communities

## Contact & Support

For issues and questions:
- Open an issue on GitHub
- Check SAM 3 documentation: https://ai.meta.com/sam3

---

**Built with using SAM 3, React, and FastAPI**
