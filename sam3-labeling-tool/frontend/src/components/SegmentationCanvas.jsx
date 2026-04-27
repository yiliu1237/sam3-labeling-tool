import React, { useRef, useEffect, useState } from 'react';
import { Stage, Layer, Image as KonvaImage, Circle, Rect, Shape, Line, Text as KonvaText } from 'react-konva';
import useStore from '../store/useStore';

const SegmentationCanvas = ({
  imageUrl,
  masks,
  previewBoxes = [],
  onPointClick,
  onBoxDraw,
  onBrushStroke
}) => {
  console.log('🔵 SegmentationCanvas render - masks:', masks ? masks.length : 'null');

  const stageRef = useRef(null);
  const [image, setImage] = useState(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [drawingBox, setDrawingBox] = useState(null);
  const [boxStart, setBoxStart] = useState(null);

  // Brush/Eraser state
  const [isDrawing, setIsDrawing] = useState(false);
  const [currentStroke, setCurrentStroke] = useState([]);
  const [cursorPos, setCursorPos] = useState({ x: 0, y: 0 });
  const [showCursor, setShowCursor] = useState(false);

  const {
    activeTool,
    refinementPoints,
    selectedMaskId,
    setSelectedMaskId,
    brushSize,
    addToast,
  } = useStore();

  // Generate unique colors for each instance
  const generateInstanceColor = (index, total) => {
    // Use HSL color space for better color distribution
    const hue = (index * 360 / Math.max(total, 1)) % 360;
    const saturation = 70 + (index % 3) * 10;
    const lightness = 50 + (index % 2) * 10;

    // Convert HSL to RGB
    const h = hue / 360;
    const s = saturation / 100;
    const l = lightness / 100;

    const hue2rgb = (p, q, t) => {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1/6) return p + (q - p) * 6 * t;
      if (t < 1/2) return q;
      if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
      return p;
    };

    let r, g, b;
    if (s === 0) {
      r = g = b = l;
    } else {
      const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
      const p = 2 * l - q;
      r = hue2rgb(p, q, h + 1/3);
      g = hue2rgb(p, q, h);
      b = hue2rgb(p, q, h - 1/3);
    }

    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
  };

  // Calculate mask statistics (pixel count and center)
  const calculateMaskStats = (mask) => {
    let sumX = 0, sumY = 0, count = 0;

    for (let y = 0; y < mask.length; y++) {
      for (let x = 0; x < mask[y].length; x++) {
        if (mask[y][x]) {
          sumX += x;
          sumY += y;
          count++;
        }
      }
    }

    return {
      pixelCount: count,
      centerX: count > 0 ? sumX / count : 0,
      centerY: count > 0 ? sumY / count : 0,
    };
  };

  // Load image
  useEffect(() => {
    if (!imageUrl) return;

    const img = new window.Image();
    img.crossOrigin = 'anonymous';

    img.onload = () => {
      setImage(img);

      // Calculate dimensions to fit container (only on first load or size change)
      const maxWidth = 800;
      const maxHeight = 600;
      const ratio = Math.min(maxWidth / img.width, maxHeight / img.height);

      setDimensions({
        width: img.width * ratio,
        height: img.height * ratio,
      });
    };

    // Set src after onload handler is attached
    img.src = imageUrl;

    // Cleanup function to prevent memory leaks
    return () => {
      img.onload = null;
    };
  }, [imageUrl]);

  // Handle mask click for selection
  const handleMaskClick = (maskId) => {
    if (activeTool !== 'cursor') return;
    setSelectedMaskId(maskId);
  };

  // Handle canvas click for points
  const handleStageClick = (e) => {
    if (activeTool !== 'point') return;

    const stage = e.target.getStage();
    const point = stage.getPointerPosition();

    if (point && onPointClick) {
      // Convert to image coordinates
      const scaleX = image.width / dimensions.width;
      const scaleY = image.height / dimensions.height;

      onPointClick({
        x: point.x * scaleX,
        y: point.y * scaleY,
        label: 1, // Positive point by default
      });
    }
  };

  // Handle box drawing
  const handleMouseDown = (e) => {
    const stage = e.target.getStage();
    const point = stage.getPointerPosition();

    if (activeTool === 'box') {
      setBoxStart(point);
      setDrawingBox({ x: point.x, y: point.y, width: 0, height: 0 });
    } else if (activeTool === 'brush' || activeTool === 'eraser') {
      // Validate mask selection
      if (selectedMaskId === null) {
        addToast('Please select a mask first', 'warning');
        return;
      }
      if (activeTool === 'eraser' && selectedMaskId === 'new') {
        addToast('Cannot erase from a new mask', 'warning');
        return;
      }

      setIsDrawing(true);
      setCurrentStroke([point.x, point.y]);
    }
  };

  const handleMouseMove = (e) => {
    const stage = e.target.getStage();
    const point = stage.getPointerPosition();

    // Update cursor position for brush/eraser cursor
    if (activeTool === 'brush' || activeTool === 'eraser') {
      setCursorPos(point);
    }

    if (activeTool === 'box' && boxStart) {
      setDrawingBox({
        x: Math.min(boxStart.x, point.x),
        y: Math.min(boxStart.y, point.y),
        width: Math.abs(point.x - boxStart.x),
        height: Math.abs(point.y - boxStart.y),
      });
    } else if ((activeTool === 'brush' || activeTool === 'eraser') && isDrawing) {
      // Add point to current stroke
      setCurrentStroke((prev) => [...prev, point.x, point.y]);
    }
  };

  const handleMouseUp = (e) => {
    if (activeTool === 'box' && drawingBox) {
      // Convert to image coordinates
      const scaleX = image.width / dimensions.width;
      const scaleY = image.height / dimensions.height;

      if (onBoxDraw) {
        onBoxDraw({
          x1: drawingBox.x * scaleX,
          y1: drawingBox.y * scaleY,
          x2: (drawingBox.x + drawingBox.width) * scaleX,
          y2: (drawingBox.y + drawingBox.height) * scaleY,
        });
      }

      setDrawingBox(null);
      setBoxStart(null);
    } else if ((activeTool === 'brush' || activeTool === 'eraser') && isDrawing) {
      // Finish stroke
      if (currentStroke.length > 0 && onBrushStroke) {
        // Convert to image coordinates
        const scaleX = image.width / dimensions.width;
        const scaleY = image.height / dimensions.height;

        const scaledStroke = [];
        for (let i = 0; i < currentStroke.length; i += 2) {
          scaledStroke.push([
            currentStroke[i] * scaleX,
            currentStroke[i + 1] * scaleY
          ]);
        }

        onBrushStroke({
          maskId: selectedMaskId,
          operation: activeTool === 'brush' ? 'add' : 'remove',
          points: scaledStroke,
          brushSize: brushSize,
        });
      }

      setIsDrawing(false);
      setCurrentStroke([]);
    }
  };

  const handleMouseEnter = () => {
    if (activeTool === 'brush' || activeTool === 'eraser') {
      setShowCursor(true);
    }
  };

  const handleMouseLeave = () => {
    setShowCursor(false);
    // Cancel any ongoing drawing
    if (isDrawing) {
      setIsDrawing(false);
      setCurrentStroke([]);
    }
  };

  // Render mask overlay with single transparent color
  const renderMasks = () => {
    if (!masks || !image) return null;

    console.log('🎨 Total masks to render:', masks.length);
    console.log('📐 Image dimensions:', image.width, 'x', image.height);
    console.log('📐 Display dimensions:', dimensions.width, 'x', dimensions.height);
    console.log('📐 Mask dimensions:', masks[0]?.[0]?.length || 0, 'x', masks[0]?.length || 0);

    const maskElements = [];
    const scaleX = dimensions.width / (masks[0]?.[0]?.length || 1);
    const scaleY = dimensions.height / (masks[0]?.length || 1);

    console.log('🔍 Scale factors - scaleX:', scaleX.toFixed(3), 'scaleY:', scaleY.toFixed(3));

    masks.forEach((mask, idx) => {
      const isSelected = selectedMaskId === idx;

      // Generate unique color for this instance
      const baseColor = generateInstanceColor(idx, masks.length);
      // Higher opacity for selected mask (0.7), lower for unselected (0.3)
      const opacity = isSelected ? 0.7 : 0.3;
      const fillColor = `rgba(${baseColor[0]}, ${baseColor[1]}, ${baseColor[2]}, ${opacity})`;

      // Bright border color (increase brightness by 20%)
      const brightColor = baseColor.map(c => Math.min(255, Math.round(c * 1.2)));
      const strokeColor = `rgb(${brightColor[0]}, ${brightColor[1]}, ${brightColor[2]})`;

      // Add transparent fill (clickable)
      maskElements.push(
        <Shape
          key={`mask-${idx}`}
          sceneFunc={(context, shape) => {
            context.beginPath();

            // Draw filled regions
            for (let y = 0; y < mask.length; y++) {
              for (let x = 0; x < mask[y].length; x++) {
                if (mask[y][x]) {
                  context.rect(x * scaleX, y * scaleY, scaleX, scaleY);
                }
              }
            }

            context.fillStrokeShape(shape);
          }}
          fill={fillColor}
          onClick={() => handleMaskClick(idx)}
          onTap={() => handleMaskClick(idx)}
          listening={activeTool === 'cursor'}
        />
      );

      // Add bright border - draw using Shape to stroke the edges
      maskElements.push(
        <Shape
          key={`border-${idx}`}
          sceneFunc={(context, shape) => {
            context.beginPath();

            // Draw only the edge pixels
            for (let y = 0; y < mask.length; y++) {
              for (let x = 0; x < mask[y].length; x++) {
                if (mask[y][x]) {
                  // Check if this pixel is on the boundary
                  const isEdge =
                    x === 0 || x === mask[y].length - 1 || y === 0 || y === mask.length - 1 ||
                    !mask[y-1]?.[x] || !mask[y+1]?.[x] || !mask[y][x-1] || !mask[y][x+1];

                  if (isEdge) {
                    context.rect(x * scaleX, y * scaleY, scaleX, scaleY);
                  }
                }
              }
            }

            context.fillStrokeShape(shape);
          }}
          fill={strokeColor}
          listening={false}
        />
      );

      // Calculate mask statistics (for debugging if needed)
      const stats = calculateMaskStats(mask);
      const { pixelCount } = stats;

      console.log(`  Mask ${idx}: ${pixelCount} pixels (${(pixelCount * scaleX * scaleY).toFixed(1)} display pixels)`);
    });

    return maskElements;
  };

  // Render refinement points
  const renderPoints = () => {
    if (!image) return null;

    const scaleX = dimensions.width / image.width;
    const scaleY = dimensions.height / image.height;

    return refinementPoints.map((point, idx) => (
      <Circle
        key={idx}
        x={point.x * scaleX}
        y={point.y * scaleY}
        radius={5}
        fill={point.label === 1 ? '#22c55e' : '#ef4444'}
        stroke="white"
        strokeWidth={2}
      />
    ));
  };

  const renderPreviewBoxes = () => {
    if (!image || !previewBoxes || previewBoxes.length === 0) return null;

    const scaleX = dimensions.width / image.width;
    const scaleY = dimensions.height / image.height;
    const boxColors = ['#ff4d4f', '#52c41a', '#1677ff', '#fa8c16', '#13c2c2', '#eb2f96'];

    return previewBoxes.flatMap((box, idx) => {
      const color = boxColors[idx % boxColors.length];
      const x = box.x1 * scaleX;
      const y = box.y1 * scaleY;
      const width = (box.x2 - box.x1) * scaleX;
      const height = (box.y2 - box.y1) * scaleY;
      const label = box.label ?? `${idx}`;

      return [
        <Rect
          key={`preview-box-${idx}`}
          x={x}
          y={y}
          width={width}
          height={height}
          stroke={color}
          strokeWidth={2}
          dash={[8, 4]}
          listening={false}
        />,
        <KonvaText
          key={`preview-label-${idx}`}
          x={x + 4}
          y={Math.max(2, y - 18)}
          text={String(label)}
          fontSize={14}
          fontStyle="bold"
          fill={color}
          listening={false}
        />
      ];
    });
  };

  return (
    <div className="card p-4">
      <Stage
        ref={stageRef}
        width={dimensions.width}
        height={dimensions.height}
        onClick={handleStageClick}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        className="rounded-lg overflow-hidden"
        style={{
          cursor: (activeTool === 'brush' || activeTool === 'eraser') ? 'none' : 'default'
        }}
      >
        <Layer>
          {image && (
            <KonvaImage
              image={image}
              width={dimensions.width}
              height={dimensions.height}
            />
          )}
          {renderPreviewBoxes()}
          {renderMasks()}
          {renderPoints()}
          {drawingBox && (
            <Rect
              x={drawingBox.x}
              y={drawingBox.y}
              width={drawingBox.width}
              height={drawingBox.height}
              stroke="#3b82f6"
              strokeWidth={2}
              dash={[5, 5]}
            />
          )}

          {/* Current brush/eraser stroke being drawn */}
          {currentStroke.length > 0 && (
            <Line
              points={currentStroke}
              stroke={activeTool === 'brush' ? '#22c55e' : '#ef4444'}
              strokeWidth={brushSize / (image?.width / dimensions.width || 1)}
              tension={0.5}
              lineCap="round"
              lineJoin="round"
              globalCompositeOperation={activeTool === 'brush' ? 'source-over' : 'destination-out'}
              opacity={0.6}
            />
          )}
        </Layer>

        {/* Brush cursor layer (always on top) */}
        {showCursor && (activeTool === 'brush' || activeTool === 'eraser') && (
          <Layer listening={false}>
            <Circle
              x={cursorPos.x}
              y={cursorPos.y}
              radius={brushSize / 2}
              stroke={activeTool === 'brush' ? '#22c55e' : '#ef4444'}
              strokeWidth={2}
              dash={[5, 5]}
              listening={false}
            />
            {/* Inner crosshair */}
            <Circle
              x={cursorPos.x}
              y={cursorPos.y}
              radius={2}
              fill={activeTool === 'brush' ? '#22c55e' : '#ef4444'}
              listening={false}
            />
          </Layer>
        )}
      </Stage>
    </div>
  );
};

export default SegmentationCanvas;
