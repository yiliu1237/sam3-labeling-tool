import React, { useState, useEffect } from 'react';
import { Sparkles, Loader2 } from 'lucide-react';
import ImageUploader from '../components/ImageUploader';
import SegmentationCanvas from '../components/SegmentationCanvas';
import VideoPlayer from '../components/VideoPlayer';
import ToolPanel from '../components/ToolPanel';
import MaskList from '../components/MaskList';
import useStore from '../store/useStore';
import { API_BASE_URL } from '../config';
import {
  uploadFile,
  segmentImageWithText,
  segmentImageWithBoxFile,
  segmentImageWithSegFile,
  segmentVideoWithText,
  // refineWithPoints,  // Disabled - Point tool removed
  // refineWithBox,     // Disabled - Box tool removed
  downloadMasksAsZip,
  editMask,
} from '../api/client';

const SingleMode = () => {
  const [imagePreview, setImagePreview] = useState(null);
  const [segmentationMode, setSegmentationMode] = useState('text');
  const [bboxFile, setBboxFile] = useState(null);
  const [bboxPreviewBoxes, setBboxPreviewBoxes] = useState([]);
  const [segFile, setSegFile] = useState(null);
  const [segPreviewPolygons, setSegPreviewPolygons] = useState([]);

  const {
    currentFile,
    currentFileId,
    currentFileType,
    setCurrentFile,
    clearCurrentFile,
    textPrompt,
    setTextPrompt,
    segmentationResult,
    setSegmentationResult,
    isLoading,
    setIsLoading,
    confidenceThreshold,
    // refinementPoints,      // Disabled - Point tool removed
    // addRefinementPoint,    // Disabled - Point tool removed
    // clearRefinementPoints, // Disabled - Point tool removed
    addToast,
    pushToHistory,
    undo,
    redo,
    canUndo,
    canRedo,
    clearHistory,
  } = useStore();

  const loadImageSize = (file) => new Promise((resolve, reject) => {
    const objectUrl = URL.createObjectURL(file);
    const img = new window.Image();

    img.onload = () => {
      resolve({ width: img.width, height: img.height });
      URL.revokeObjectURL(objectUrl);
    };

    img.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      reject(new Error('Failed to read image size'));
    };

    img.src = objectUrl;
  });

  const parseYoloPreviewBoxes = (content, imageWidth, imageHeight) => {
    const boxes = [];

    content.split(/\r?\n/).forEach((line, index) => {
      const stripped = line.trim();
      if (!stripped || stripped.startsWith('#')) {
        return;
      }

      const parts = stripped.split(/\s+/);
      if (parts.length !== 5) {
        throw new Error(`Line ${index + 1} must contain exactly 5 YOLO values`);
      }

      const [classId, cxRaw, cyRaw, wRaw, hRaw] = parts;
      const cx = Number(cxRaw);
      const cy = Number(cyRaw);
      const width = Number(wRaw);
      const height = Number(hRaw);

      if ([cx, cy, width, height].some((value) => Number.isNaN(value))) {
        throw new Error(`Line ${index + 1} contains non-numeric YOLO values`);
      }

      if ([cx, cy, width, height].some((value) => value < 0 || value > 1)) {
        throw new Error(`Line ${index + 1} must use normalized YOLO values in [0, 1]`);
      }

      boxes.push({
        label: classId,
        x1: (cx - (width / 2)) * imageWidth,
        y1: (cy - (height / 2)) * imageHeight,
        x2: (cx + (width / 2)) * imageWidth,
        y2: (cy + (height / 2)) * imageHeight,
      });
    });

    return boxes;
  };

  const handleBboxFileChange = async (file) => {
    setBboxFile(file);

    if (!file) {
      setBboxPreviewBoxes([]);
      return;
    }

    if (!currentFile) {
      addToast('Please upload an image first', 'error');
      setBboxPreviewBoxes([]);
      return;
    }

    try {
      const [content, imageSize] = await Promise.all([
        file.text(),
        loadImageSize(currentFile),
      ]);
      const parsedBoxes = parseYoloPreviewBoxes(content, imageSize.width, imageSize.height);
      setBboxPreviewBoxes(parsedBoxes);
      addToast(`Loaded ${parsedBoxes.length} YOLO boxes`, 'success');
    } catch (error) {
      console.error('BBox preview parse error:', error);
      setBboxPreviewBoxes([]);
      addToast(error.message || 'Failed to read YOLO bbox file', 'error');
    }
  };

  const handleSegFileChange = async (file) => {
    setSegFile(file);
    if (!file) { setSegPreviewPolygons([]); return; }
    if (!currentFile) {
      addToast('Please upload an image first', 'error');
      setSegPreviewPolygons([]);
      return;
    }
    try {
      const [content, imageSize] = await Promise.all([
        file.text(),
        loadImageSize(currentFile),
      ]);
      const polygons = [];
      content.split(/\r?\n/).forEach((line, i) => {
        const stripped = line.trim();
        if (!stripped || stripped.startsWith('#')) return;
        const parts = stripped.split(/\s+/);
        if (parts.length < 7) throw new Error(`Line ${i + 1}: need class_id + at least 3 points`);
        const coords = parts.slice(1).map(Number);
        if (coords.length % 2 !== 0) throw new Error(`Line ${i + 1}: odd number of coordinates`);
        const points = [];
        for (let j = 0; j < coords.length; j += 2) {
          points.push([coords[j] * imageSize.width, coords[j + 1] * imageSize.height]);
        }
        polygons.push({ label: parts[0], points });
      });
      setSegPreviewPolygons(polygons);
      addToast(`Loaded ${polygons.length} YOLO seg polygons`, 'success');
    } catch (error) {
      setSegPreviewPolygons([]);
      addToast(error.message || 'Failed to parse YOLO seg file', 'error');
    }
  };

  // Handle file selection
  const handleFileSelect = async (file) => {
    try {
      setIsLoading(true);
      setBboxPreviewBoxes([]);
      setBboxFile(null);

      // Upload file first
      const result = await uploadFile(file);
      setCurrentFile(file, result.file_id, result.file_type);

      // Create preview based on file type
      if (result.file_type === 'video') {
        // For videos, fetch the first frame as preview
        const frameUrl = `${API_BASE_URL}/api/segment/video/frame/${result.file_id}?frame_index=0`;
        setImagePreview(frameUrl);
        addToast('Video uploaded successfully', 'success');
      } else {
        // For images, create data URL preview
        const reader = new FileReader();
        reader.onload = (e) => setImagePreview(e.target.result);
        reader.readAsDataURL(file);
        addToast('Image uploaded successfully', 'success');
      }

    } catch (error) {
      console.error('Upload error:', error);
      addToast('Failed to upload file', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  // Handle text segmentation
  const handleSegment = async () => {
    if (!currentFileId) {
      addToast('Please upload a file first', 'error');
      return;
    }

    try {
      setIsLoading(true);

      let result;
      if (currentFileType === 'video') {
        if (!textPrompt.trim()) {
          addToast('Please enter a prompt for video segmentation', 'error');
          return;
        }
        // Use video segmentation API
        result = await segmentVideoWithText(
          currentFileId,
          textPrompt,
          0, // frame_index
          confidenceThreshold
        );
        addToast(`Found ${result.masks?.length || 0} instances in video`, 'success');
      } else {
        if (segmentationMode === 'bbox-file') {
          if (!bboxFile) {
            addToast('Please choose a YOLO bbox label file', 'error');
            return;
          }
          result = await segmentImageWithBoxFile(
            currentFileId,
            bboxFile,
            confidenceThreshold
          );
          addToast(`Processed ${result.masks?.length || 0} bounding boxes`, 'success');
        } else if (segmentationMode === 'seg-file') {
          if (!segFile) {
            addToast('Please choose a YOLO seg label file', 'error');
            return;
          }
          result = await segmentImageWithSegFile(currentFileId, segFile);
          addToast(`Loaded ${result.masks?.length || 0} segmentation masks`, 'success');
        } else {
          if (!textPrompt.trim()) {
            addToast('Please enter a text prompt', 'error');
            return;
          }
          // Use image segmentation API
          result = await segmentImageWithText(
            currentFileId,
            textPrompt,
            confidenceThreshold
          );
          addToast(`Found ${result.masks?.length || 0} instances`, 'success');
        }
      }

      setSegmentationResult(result);
      // Initialize history with first segmentation result
      clearHistory();
      pushToHistory(result);
    } catch (error) {
      console.error('Segmentation error:', error);
      addToast('Segmentation failed', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  // NOTE: Point and Box refinement handlers are disabled.
  // SAM3's architecture doesn't support true instance-aware refinement:
  // - add_geometric_prompt() reruns entire detection (unpredictable)
  // - predict_inst() creates new masks (not refinement)
  // Use Brush/Eraser tools for precise instance-aware mask editing.

  // // Handle point refinement
  // const handlePointClick = async (point) => {
  //   addRefinementPoint(point);

  //   try {
  //     setIsLoading(true);

  //     const points = [...refinementPoints, point];
  //     const result = await refineWithPoints(currentFileId, points);

  //     setSegmentationResult(result);
  //     addToast('Segmentation refined', 'success');
  //   } catch (error) {
  //     console.error('Refinement error:', error);
  //     addToast('Refinement failed', 'error');
  //   } finally {
  //     setIsLoading(false);
  //   }
  // };

  // // Handle box refinement
  // const handleBoxDraw = async (box) => {
  //   try {
  //     setIsLoading(true);

  //     const result = await refineWithBox(currentFileId, box);

  //     setSegmentationResult(result);
  //     addToast('Segmentation refined with box', 'success');
  //   } catch (error) {
  //     console.error('Box refinement error:', error);
  //     addToast('Box refinement failed', 'error');
  //   } finally {
  //     setIsLoading(false);
  //   }
  // };

  // Handle brush/eraser strokes
  const handleBrushStroke = async (strokeData) => {
    console.log('Brush stroke:', strokeData);

    try {
      setIsLoading(true);

      // Call backend API to edit the mask
      const result = await editMask(
        currentFileId,
        strokeData.maskId,
        strokeData.operation,
        [strokeData.points], // Wrap in array as backend expects list of strokes
        strokeData.brushSize
      );

      console.log('Mask edit result:', result);

      // Merge the edited mask back into the segmentation result
      if (result && result.masks && result.masks.length > 0) {
        const strokeMask = result.masks[0]; // The stroke mask from backend

        // Clone current segmentation result
        const updatedResult = { ...segmentationResult };

        if (strokeData.maskId === 'new') {
          // Creating a new mask - add it to the list
          updatedResult.masks = [...updatedResult.masks, strokeMask];
          updatedResult.boxes = [...updatedResult.boxes, result.boxes[0]];
          updatedResult.scores = [...updatedResult.scores, result.scores[0]];
          if (updatedResult.labels) {
            updatedResult.labels = [...updatedResult.labels, 'manual_mask'];
          }

          addToast('New mask created successfully', 'success');
        } else {
          // Editing existing mask
          const maskIndex = strokeData.maskId;

          if (strokeData.operation === 'add') {
            // Merge stroke with existing mask (OR operation)
            const existingMask = updatedResult.masks[maskIndex];
            const mergedMask = existingMask.map((row, y) =>
              row.map((pixel, x) =>
                pixel || (strokeMask[y]?.[x] || 0)
              )
            );
            updatedResult.masks[maskIndex] = mergedMask;
            addToast('Pixels added to mask', 'success');

          } else if (strokeData.operation === 'remove') {
            // Remove stroke from existing mask (AND NOT operation)
            const existingMask = updatedResult.masks[maskIndex];
            const mergedMask = existingMask.map((row, y) =>
              row.map((pixel, x) =>
                pixel && !(strokeMask[y]?.[x] || 0) ? 1 : 0
              )
            );

            // Check if mask is completely empty (no pixels left)
            const hasPixels = mergedMask.some(row => row.some(pixel => pixel > 0));

            if (!hasPixels) {
              // Mask is completely erased - remove it from the list
              updatedResult.masks.splice(maskIndex, 1);
              updatedResult.boxes.splice(maskIndex, 1);
              updatedResult.scores.splice(maskIndex, 1);
              if (updatedResult.labels) {
                updatedResult.labels.splice(maskIndex, 1);
              }
              addToast('Mask completely erased and removed', 'success');

              // Clear the selected mask since it no longer exists
              const { setSelectedMaskId } = useStore.getState();
              setSelectedMaskId(null);
            } else {
              updatedResult.masks[maskIndex] = mergedMask;
              addToast('Pixels removed from mask', 'success');
            }
          }
        }

        // Update the segmentation result state and push to history
        pushToHistory(updatedResult);
        console.log('Updated segmentation result:', updatedResult);
      }

    } catch (error) {
      console.error('Brush stroke error:', error);
      addToast(`Failed to apply brush stroke: ${error.message}`, 'error');
    } finally {
      setIsLoading(false);
    }
  };

  // Handle save masks - download as ZIP
  const handleSave = async () => {
    if (!currentFileId || !segmentationResult) {
      addToast('No segmentation results to save', 'error');
      return;
    }

    try {
      setIsLoading(true);
      addToast('Preparing download...', 'info');

      const blob = await downloadMasksAsZip(
        currentFileId,
        segmentationResult.masks,
        segmentationResult.scores,
        segmentationResult.boxes,
        textPrompt,
        segmentationResult.labels || []
      );

      // Create download link
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'segmentation_masks.zip';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      addToast('Download started!', 'success');
    } catch (error) {
      console.error('Download error:', error);
      addToast(`Download failed: ${error.message}`, 'error');
    } finally {
      setIsLoading(false);
    }
  };


  // Handle mask deletion
  const handleDeleteMask = (maskIndex) => {
    if (!segmentationResult || !segmentationResult.masks) {
      addToast('No masks to delete', 'error');
      return;
    }

    console.log('🗑️ Deleting mask at index:', maskIndex);

    // Clone the segmentation result
    const updatedResult = { ...segmentationResult };

    // Remove the mask and its associated data
    updatedResult.masks = [...updatedResult.masks];
    updatedResult.masks.splice(maskIndex, 1);

    updatedResult.boxes = [...updatedResult.boxes];
    updatedResult.boxes.splice(maskIndex, 1);

    updatedResult.scores = [...updatedResult.scores];
    updatedResult.scores.splice(maskIndex, 1);

    if (updatedResult.labels) {
      updatedResult.labels = [...updatedResult.labels];
      updatedResult.labels.splice(maskIndex, 1);
    }

    // Push to history and update state
    pushToHistory(updatedResult);

    console.log('Mask deleted. Remaining masks:', updatedResult.masks.length);
  };

  // Handle reset
  const handleReset = () => {
    clearCurrentFile();
    setImagePreview(null);
    setTextPrompt('');
    setBboxFile(null);
    setBboxPreviewBoxes([]);
    setSegFile(null);
    setSegPreviewPolygons([]);
    setSegmentationMode('text');
    // clearRefinementPoints();  // Disabled - Point tool removed
    clearHistory();
  };

  // Keyboard shortcuts for undo/redo
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Check if we're in an input field
      const isInputFocused = ['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName);
      if (isInputFocused) return;

      // Ctrl+Z or Cmd+Z for undo
      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
        e.preventDefault();
        if (canUndo()) {
          undo();
          addToast('Undo', 'info');
        }
      }

      // Ctrl+Shift+Z or Cmd+Shift+Z for redo
      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && e.shiftKey) {
        e.preventDefault();
        if (canRedo()) {
          redo();
          addToast('Redo', 'info');
        }
      }

      // Alternative: Ctrl+Y or Cmd+Y for redo
      if ((e.ctrlKey || e.metaKey) && e.key === 'y') {
        e.preventDefault();
        if (canRedo()) {
          redo();
          addToast('Redo', 'info');
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [undo, redo, canUndo, canRedo, addToast]);

  const isSegmentDisabled = (() => {
    if (isLoading || !currentFileId) {
      return true;
    }
    if (currentFileType === 'video') {
      return !textPrompt.trim();
    }
    if (segmentationMode === 'bbox-file') return !bboxFile;
    if (segmentationMode === 'seg-file') return !segFile;
    return !textPrompt.trim();
  })();

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
          Single Image/Video Segmentation
        </h2>
        <p className="text-gray-600 dark:text-gray-400">
          Upload an image or video, then segment with text or a YOLO bbox label file
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main canvas area */}
        <div className="lg:col-span-2 space-y-4">
          {!imagePreview ? (
            <ImageUploader onFileSelect={handleFileSelect} />
          ) : (
            <>
              {/* Prompt input */}
              <div className="card p-4">
                {currentFileType === 'image' && (
                  <div className="mb-4">
                    <div className="inline-flex rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
                      <button
                        type="button"
                        onClick={() => setSegmentationMode('text')}
                        className={`px-4 py-2 text-sm font-medium transition-colors ${
                          segmentationMode === 'text'
                            ? 'bg-primary-600 text-white'
                            : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300'
                        }`}
                      >
                        Text Prompt
                      </button>
                      <button
                        type="button"
                        onClick={() => setSegmentationMode('bbox-file')}
                        className={`px-4 py-2 text-sm font-medium transition-colors ${
                          segmentationMode === 'bbox-file'
                            ? 'bg-primary-600 text-white'
                            : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300'
                        }`}
                      >
                        YOLO BBox File
                      </button>
                      <button
                        type="button"
                        onClick={() => setSegmentationMode('seg-file')}
                        className={`px-4 py-2 text-sm font-medium transition-colors ${
                          segmentationMode === 'seg-file'
                            ? 'bg-primary-600 text-white'
                            : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300'
                        }`}
                      >
                        YOLO Seg File
                      </button>
                    </div>
                  </div>
                )}

                {currentFileType === 'image' && segmentationMode === 'seg-file' ? (
                  <div className="space-y-3">
                    <input
                      type="file"
                      accept=".txt"
                      onChange={(e) => handleSegFileChange(e.target.files?.[0] || null)}
                      className="input"
                      disabled={isLoading}
                    />
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      YOLO seg format: <code>class_id x1 y1 x2 y2 x3 y3 ...</code> (normalized polygon points)
                    </p>
                    {segFile && (
                      <p className="text-sm text-gray-700 dark:text-gray-300">
                        Selected: <span className="font-medium">{segFile.name}</span>
                      </p>
                    )}
                  </div>
                ) : currentFileType === 'image' && segmentationMode === 'bbox-file' ? (
                  <div className="space-y-3">
                    <input
                      type="file"
                      accept=".txt,.csv,.json"
                      onChange={(e) => handleBboxFileChange(e.target.files?.[0] || null)}
                      className="input"
                      disabled={isLoading}
                    />
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      Accepts YOLO label files like <code>5 0.464323 0.805556 0.125521 0.203704</code>
                    </p>
                    {bboxFile && (
                      <div className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                        <p>
                          Selected bbox file: <span className="font-medium">{bboxFile.name}</span>
                        </p>
                        <p>
                          Preview boxes: <span className="font-medium">{bboxPreviewBoxes.length}</span>
                        </p>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="flex space-x-2">
                    <input
                      type="text"
                      value={textPrompt}
                      onChange={(e) => setTextPrompt(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && handleSegment()}
                      placeholder="Enter text prompt (e.g., 'leaf, crack' for multiple)..."
                      className="input flex-1"
                      disabled={isLoading}
                    />
                  </div>
                )}

                <div className="mt-4">
                  <button
                    onClick={handleSegment}
                    disabled={isSegmentDisabled}
                    className="btn-primary flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isLoading ? (
                      <>
                        <Loader2 className="animate-spin" size={18} />
                        <span>Processing...</span>
                      </>
                    ) : (
                      <>
                        <Sparkles size={18} />
                        <span>
                          {segmentationMode === 'bbox-file' ? 'Segment From BBox File'
                            : segmentationMode === 'seg-file' ? 'Load Seg Masks'
                            : 'Segment'}
                        </span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Canvas or Video Player */}
              {currentFileType === 'video' ? (
                <VideoPlayer
                  videoId={currentFileId}
                  masks={segmentationResult?.masks}
                  // onPointClick={handlePointClick}  // Disabled - Point tool removed
                  // onBoxDraw={handleBoxDraw}         // Disabled - Box tool removed
                />
              ) : (
                <SegmentationCanvas
                  imageUrl={imagePreview}
                  masks={segmentationResult?.masks}
                  previewBoxes={segmentationMode === 'bbox-file' ? bboxPreviewBoxes : []}
                  previewPolygons={segmentationMode === 'seg-file' ? segPreviewPolygons : []}
                  // onPointClick={handlePointClick}  // Disabled - Point tool removed
                  // onBoxDraw={handleBoxDraw}         // Disabled - Box tool removed
                  onBrushStroke={handleBrushStroke}
                />
              )}

              {/* Mask List (show when we have segmentation results) */}
              {segmentationResult && segmentationResult.masks && segmentationResult.masks.length > 0 && (
                <MaskList
                  masks={segmentationResult.masks}
                  scores={segmentationResult.scores || []}
                  onDeleteMask={handleDeleteMask}
                />
              )}

              {/* Reset button */}
              <button
                onClick={handleReset}
                className="btn-secondary w-full"
              >
                Upload New File
              </button>
            </>
          )}
        </div>

        {/* Tool panel */}
        <div>
          <ToolPanel
            // onClearPoints={clearRefinementPoints}  // Disabled - Point tool removed
            onSave={handleSave}
          />
        </div>
      </div>
    </div>
  );
};

export default SingleMode;
