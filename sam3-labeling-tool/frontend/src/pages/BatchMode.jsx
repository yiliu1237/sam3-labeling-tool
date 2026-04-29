import React, { useState, useEffect } from 'react';
import { FolderOpen, Play, Loader2, CheckCircle, XCircle } from 'lucide-react';
import { motion } from 'framer-motion';
import useStore from '../store/useStore';
import { createBatchJob, getBatchJobStatus } from '../api/client';

const BatchMode = () => {
  const [inputFolder, setInputFolder] = useState('');
  const [outputFolder, setOutputFolder] = useState('');
  const [labelFolder, setLabelFolder] = useState('');
  const [prompts, setPrompts] = useState(['']);
  const [currentJobId, setCurrentJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);

  const { confidenceThreshold, isLoading, setIsLoading, addToast } = useStore();

  const useLabelFolder = labelFolder.trim() !== '';

  useEffect(() => {
    if (!currentJobId) return;
    const interval = setInterval(async () => {
      try {
        const status = await getBatchJobStatus(currentJobId);
        setJobStatus(status);
        if (status.status === 'completed' || status.status === 'failed') {
          clearInterval(interval);
          setIsLoading(false);
          if (status.status === 'completed') {
            addToast('Batch processing completed!', 'success');
          } else {
            addToast(`Batch processing failed: ${status.error}`, 'error');
          }
        }
      } catch {
        clearInterval(interval);
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [currentJobId, setIsLoading, addToast]);

  const handlePromptChange = (index, value) => {
    const next = [...prompts];
    next[index] = value;
    setPrompts(next);
  };

  const handleStartBatch = async () => {
    if (!inputFolder || !outputFolder) {
      addToast('Please provide input and output folders', 'error');
      return;
    }
    if (!useLabelFolder && prompts.filter((p) => p.trim()).length === 0) {
      addToast('Please provide a label folder or at least one text prompt', 'error');
      return;
    }
    try {
      setIsLoading(true);
      const result = await createBatchJob(
        inputFolder,
        outputFolder,
        prompts.filter((p) => p.trim()),
        {
          labelFolder: labelFolder.trim() || null,
          confidenceThreshold,
        }
      );
      setCurrentJobId(result.job_id);
      addToast('Batch processing started', 'info');
    } catch {
      addToast('Failed to start batch processing', 'error');
      setIsLoading(false);
    }
  };

  const statusIcon = () => {
    if (!jobStatus) return null;
    if (jobStatus.status === 'completed') return <CheckCircle className="text-green-500" size={24} />;
    if (jobStatus.status === 'failed')    return <XCircle    className="text-red-500"   size={24} />;
    if (jobStatus.status === 'processing') return <Loader2 className="text-blue-500 animate-spin" size={24} />;
    return null;
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">Batch Processing</h2>
        <p className="text-gray-600 dark:text-gray-400">
          Process a folder of images. Outputs: mask PNG, overlay PNG, and YOLO seg txt per image.
        </p>
      </div>

      <div className="space-y-6">
        <div className="card p-6 space-y-5">
          {/* Folders */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-900 dark:text-white mb-1">
                Input Folder <span className="text-red-500">*</span>
              </label>
              <div className="flex space-x-2">
                <input type="text" value={inputFolder} onChange={(e) => setInputFolder(e.target.value)}
                  placeholder="/path/to/images" className="input flex-1" />
                <button className="btn-secondary"><FolderOpen size={18} /></button>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-900 dark:text-white mb-1">
                Output Folder <span className="text-red-500">*</span>
              </label>
              <div className="flex space-x-2">
                <input type="text" value={outputFolder} onChange={(e) => setOutputFolder(e.target.value)}
                  placeholder="/path/to/output" className="input flex-1" />
                <button className="btn-secondary"><FolderOpen size={18} /></button>
              </div>
            </div>
          </div>

          {/* Label folder */}
          <div>
            <label className="block text-sm font-medium text-gray-900 dark:text-white mb-1">
              Label Folder <span className="text-gray-400 font-normal">(optional — YOLO bbox txt files)</span>
            </label>
            <div className="flex space-x-2">
              <input type="text" value={labelFolder} onChange={(e) => setLabelFolder(e.target.value)}
                placeholder="/path/to/labels  (leave empty to use text prompts instead)"
                className="input flex-1" />
              <button className="btn-secondary"><FolderOpen size={18} /></button>
            </div>
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
              {useLabelFolder
                ? '✓ Box-prompted mode — each image will use its matching .txt label file'
                : 'Text-prompted mode — SAM3 will search for objects matching your prompts below'}
            </p>
          </div>

          {/* Text prompts — only shown when no label folder */}
          {!useLabelFolder && (
            <div>
              <label className="block text-sm font-medium text-gray-900 dark:text-white mb-1">
                Text Prompts
              </label>
              <div className="space-y-2">
                {prompts.map((p, i) => (
                  <div key={i} className="flex space-x-2">
                    <input type="text" value={p} onChange={(e) => handlePromptChange(i, e.target.value)}
                      placeholder={`e.g. "pipe", "crack", "valve"`} className="input flex-1" />
                    {prompts.length > 1 && (
                      <button onClick={() => setPrompts(prompts.filter((_, j) => j !== i))}
                        className="btn-secondary px-3">×</button>
                    )}
                  </div>
                ))}
                <button onClick={() => setPrompts([...prompts, ''])} className="btn-secondary text-sm">
                  + Add Prompt
                </button>
              </div>
            </div>
          )}

          {/* Confidence threshold */}
          <div className="w-40">
            <label className="block text-sm font-medium text-gray-900 dark:text-white mb-1">
              Confidence Threshold
            </label>
            <input type="number" value={confidenceThreshold}
              onChange={(e) => useStore.setState({ confidenceThreshold: parseFloat(e.target.value) })}
              min="0" max="1" step="0.05" className="input" />
          </div>

          {/* Output description */}
          <div className="rounded-lg bg-gray-50 dark:bg-gray-800 p-3 text-sm text-gray-600 dark:text-gray-400">
            <p className="font-medium mb-1">Each image will produce:</p>
            <ul className="list-disc list-inside space-y-0.5">
              <li><code>masks/{'{stem}'}_mask.png</code> — binary mask</li>
              <li><code>overlays/{'{stem}'}_overlay.png</code> — image with colored mask overlay</li>
              <li><code>labels/{'{stem}'}.txt</code> — YOLO segmentation format</li>
            </ul>
          </div>

          <button onClick={handleStartBatch} disabled={isLoading}
            className="btn-primary w-full flex items-center justify-center space-x-2 disabled:opacity-50">
            {isLoading ? (
              <><Loader2 className="animate-spin" size={18} /><span>Processing...</span></>
            ) : (
              <><Play size={18} /><span>Start Batch Processing</span></>
            )}
          </button>
        </div>

        {/* Progress */}
        {jobStatus && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="card p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Processing Status</h3>
              {statusIcon()}
            </div>
            <div className="mb-4">
              <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400 mb-2">
                <span>{jobStatus.processed_files} / {jobStatus.total_files} files</span>
                <span>{(jobStatus.progress * 100).toFixed(0)}%</span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${jobStatus.progress * 100}%` }} />
              </div>
            </div>
            {jobStatus.current_file && (
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Processing: <span className="font-medium">{jobStatus.current_file}</span>
              </p>
            )}
            {jobStatus.error && (
              <p className="text-sm text-red-600 mt-2">Error: {jobStatus.error}</p>
            )}
            {jobStatus.status === 'completed' && (
              <p className="text-sm text-green-600 dark:text-green-400 mt-3">
                ✓ Results saved to: <code>{outputFolder}</code>
              </p>
            )}
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default BatchMode;
