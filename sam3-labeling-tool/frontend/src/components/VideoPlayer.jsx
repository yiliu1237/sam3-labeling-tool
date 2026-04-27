import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, SkipBack, SkipForward } from 'lucide-react';
import SegmentationCanvas from './SegmentationCanvas';
import { getVideoInfo } from '../api/client';
import { API_BASE_URL } from '../config';

const VideoPlayer = ({ videoId, masks, onPointClick, onBoxDraw, onFrameChange }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [totalFrames, setTotalFrames] = useState(100); // Default, will be updated
  const [fps, setFps] = useState(30); // Default fps
  const [videoInfo, setVideoInfo] = useState(null);
  const [frameUrl, setFrameUrl] = useState(null);
  const [loadedFrames, setLoadedFrames] = useState(new Set());
  const [isBuffering, setIsBuffering] = useState(false);
  const intervalRef = useRef(null);
  const preloadBuffer = 10; // Preload next 10 frames
  const minBufferSize = 5; // Minimum frames ahead to have loaded before playing

  // Preload frames ahead of current frame
  const preloadFrames = (startFrame) => {
    if (!videoId) return;

    const framesToPreload = [];
    for (let i = 0; i < preloadBuffer; i++) {
      const frameIndex = startFrame + i;
      if (frameIndex >= totalFrames) break;
      if (!loadedFrames.has(frameIndex)) {
        framesToPreload.push(frameIndex);
      }
    }

    // Preload frames in parallel
    framesToPreload.forEach((frameIndex) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      const url = `${API_BASE_URL}/api/segment/video/frame/${videoId}?frame_index=${frameIndex}`;
      img.onload = () => {
        setLoadedFrames((prev) => new Set([...prev, frameIndex]));
      };
      img.src = url;
    });
  };

  // Fetch video info on mount
  useEffect(() => {
    const fetchVideoInfo = async () => {
      if (videoId) {
        try {
          const info = await getVideoInfo(videoId);
          setTotalFrames(info.total_frames);
          setFps(info.fps);
          setVideoInfo(info);
          console.log('Video info:', info);

          // Preload first few frames
          setTimeout(() => {
            preloadFrames(1); // Preload frames 1-10 (frame 0 loads automatically)
          }, 100);
        } catch (error) {
          console.error('Failed to fetch video info:', error);
        }
      }
    };

    fetchVideoInfo();
  }, [videoId]);

  // Load current frame and preload next frames
  useEffect(() => {
    if (!videoId) return;

    // Use frame index directly in URL - browser will cache based on full URL
    const url = `${API_BASE_URL}/api/segment/video/frame/${videoId}?frame_index=${currentFrame}`;

    // Update immediately - let the browser handle caching
    setFrameUrl(url);

    // Mark current frame as loaded
    setLoadedFrames((prev) => new Set([...prev, currentFrame]));

    // Preload upcoming frames when playing
    if (isPlaying) {
      preloadFrames(currentFrame + 1);
    }

    // Notify parent component about frame change
    if (onFrameChange) {
      onFrameChange(currentFrame);
    }
  }, [videoId, currentFrame, isPlaying]);

  // Check if we have enough buffered frames ahead
  useEffect(() => {
    if (!isPlaying) {
      setIsBuffering(false);
      return;
    }

    // Count how many frames ahead are loaded
    let bufferedCount = 0;
    for (let i = 1; i <= minBufferSize; i++) {
      if (loadedFrames.has(currentFrame + i) || currentFrame + i >= totalFrames) {
        bufferedCount++;
      } else {
        break;
      }
    }

    // If we don't have enough buffered, pause playback temporarily
    if (bufferedCount < minBufferSize && currentFrame < totalFrames - minBufferSize) {
      setIsBuffering(true);
    } else {
      setIsBuffering(false);
    }
  }, [currentFrame, loadedFrames, isPlaying, totalFrames]);

  // Handle play/pause
  useEffect(() => {
    if (isPlaying && !isBuffering) {
      // Calculate interval based on actual FPS
      const interval = fps > 0 ? 1000 / fps : 100;

      intervalRef.current = setInterval(() => {
        setCurrentFrame((prev) => {
          if (prev >= totalFrames - 1) {
            setIsPlaying(false);
            return 0; // Loop back to start
          }
          return prev + 1;
        });
      }, interval);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isPlaying, isBuffering, totalFrames, fps]);

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const handlePreviousFrame = () => {
    setIsPlaying(false);
    setCurrentFrame((prev) => Math.max(0, prev - 1));
  };

  const handleNextFrame = () => {
    setIsPlaying(false);
    setCurrentFrame((prev) => Math.min(totalFrames - 1, prev + 1));
  };

  const handleSliderChange = (e) => {
    setIsPlaying(false);
    setCurrentFrame(parseInt(e.target.value));
  };

  return (
    <div className="space-y-4">
      {/* Canvas with current frame */}
      <div className="relative">
        {frameUrl ? (
          <>
            <SegmentationCanvas
              imageUrl={frameUrl}
              masks={masks}
              onPointClick={onPointClick}
              onBoxDraw={onBoxDraw}
            />
            {isBuffering && (
              <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-40 pointer-events-none">
                <div className="bg-white dark:bg-gray-800 px-4 py-2 rounded-lg shadow-lg">
                  <div className="flex items-center space-x-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-2 border-primary-500 border-t-transparent"></div>
                    <span className="text-sm font-medium text-gray-900 dark:text-white">Buffering...</span>
                  </div>
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="card p-4 flex items-center justify-center h-96 bg-gray-100 dark:bg-gray-700">
            <div className="text-center">
              <Play className="mx-auto mb-2 text-gray-400" size={48} />
              <p className="text-gray-600 dark:text-gray-400">Loading video...</p>
            </div>
          </div>
        )}
      </div>

      {/* Video controls */}
      <div className="card p-4 space-y-3">
        {/* Play/Pause and Frame navigation */}
        <div className="flex items-center justify-center space-x-4">
          <button
            onClick={handlePreviousFrame}
            className="p-2 rounded-lg bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
            title="Previous frame"
          >
            <SkipBack size={20} />
          </button>

          <button
            onClick={handlePlayPause}
            className="p-3 rounded-lg bg-primary-500 hover:bg-primary-600 text-white transition-colors"
            title={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? <Pause size={24} /> : <Play size={24} />}
          </button>

          <button
            onClick={handleNextFrame}
            className="p-2 rounded-lg bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
            title="Next frame"
          >
            <SkipForward size={20} />
          </button>
        </div>

        {/* Frame slider */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm text-gray-600 dark:text-gray-400">
            <span>Frame: {currentFrame + 1} / {totalFrames}</span>
            {videoInfo && (
              <span className="text-xs">
                {fps.toFixed(1)} fps • {videoInfo.duration.toFixed(1)}s • {videoInfo.width}x{videoInfo.height}
              </span>
            )}
          </div>
          <input
            type="range"
            min="0"
            max={totalFrames - 1}
            value={currentFrame}
            onChange={handleSliderChange}
            className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-primary-500"
          />
          <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-500">
            <span>{(currentFrame / Math.max(fps, 1)).toFixed(2)}s</span>
            <span>{videoInfo ? videoInfo.duration.toFixed(2) : '0.00'}s</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VideoPlayer;
