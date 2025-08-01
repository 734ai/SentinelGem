/**
 * SentinelGem File Upload Component
 * Author: Muzan Sano
 * Multi-modal file analysis interface
 */

import React, { useState, useCallback } from 'react';
import { Upload, X, CheckCircle, AlertCircle, FileText, Image, Music, Link2 } from 'lucide-react';

export const FileUploadAnalyzer = () => {
  const [files, setFiles] = useState([]);
  const [analyses, setAnalyses] = useState({});
  const [uploading, setUploading] = useState(false);

  const onDrop = useCallback((acceptedFiles) => {
    const newFiles = acceptedFiles.map(file => ({
      id: Math.random().toString(36).substring(7),
      file,
      status: 'pending',
      analysis: null
    }));
    setFiles(prev => [...prev, ...newFiles]);
  }, []);

  const analyzeFile = async (fileItem) => {
    setFiles(prev => prev.map(f => 
      f.id === fileItem.id ? { ...f, status: 'analyzing' } : f
    ));

    const formData = new FormData();
    formData.append('file', fileItem.file);
    formData.append('analysis_types', JSON.stringify(['ocr', 'phishing', 'malware']));

    try {
      const response = await fetch('/api/v1/analyze', {
        method: 'POST',
        body: formData
      });
      
      const result = await response.json();
      
      setFiles(prev => prev.map(f => 
        f.id === fileItem.id ? { 
          ...f, 
          status: 'completed', 
          analysis: result 
        } : f
      ));
      
      setAnalyses(prev => ({ ...prev, [fileItem.id]: result }));
    } catch (error) {
      setFiles(prev => prev.map(f => 
        f.id === fileItem.id ? { 
          ...f, 
          status: 'error', 
          error: error.message 
        } : f
      ));
    }
  };

  const removeFile = (fileId) => {
    setFiles(prev => prev.filter(f => f.id !== fileId));
    setAnalyses(prev => {
      const { [fileId]: removed, ...rest } = prev;
      return rest;
    });
  };

  const getFileIcon = (fileType) => {
    if (fileType.startsWith('image/')) return <Image className="h-6 w-6" />;
    if (fileType.startsWith('audio/')) return <Music className="h-6 w-6" />;
    if (fileType.includes('pdf') || fileType.includes('document')) return <FileText className="h-6 w-6" />;
    return <FileText className="h-6 w-6" />;
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return 'text-green-600 bg-green-100';
      case 'analyzing': return 'text-blue-600 bg-blue-100';
      case 'error': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-lg">
        {/* Header */}
        <div className="p-6 border-b border-gray-200">
          <h2 className="text-2xl font-bold text-gray-900">File Analysis</h2>
          <p className="text-gray-600 mt-2">
            Upload files for comprehensive threat analysis including OCR, phishing detection, and malware scanning.
          </p>
        </div>

        {/* Upload Area */}
        <div className="p-6">
          <div
            className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors cursor-pointer"
            onDrop={(e) => {
              e.preventDefault();
              const droppedFiles = Array.from(e.dataTransfer.files);
              onDrop(droppedFiles);
            }}
            onDragOver={(e) => e.preventDefault()}
            onClick={() => document.getElementById('fileInput').click()}
          >
            <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              Drop files here or click to upload
            </h3>
            <p className="text-gray-500">
              Supports images, audio files, documents, and more
            </p>
            <input
              id="fileInput"
              type="file"
              multiple
              className="hidden"
              onChange={(e) => onDrop(Array.from(e.target.files))}
              accept="image/*,audio/*,.pdf,.doc,.docx,.txt"
            />
          </div>
        </div>

        {/* File List */}
        {files.length > 0 && (
          <div className="p-6 border-t border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Uploaded Files ({files.length})
            </h3>
            
            <div className="space-y-4">
              {files.map((fileItem) => (
                <FileAnalysisCard
                  key={fileItem.id}
                  fileItem={fileItem}
                  onAnalyze={analyzeFile}
                  onRemove={removeFile}
                  getFileIcon={getFileIcon}
                  getStatusColor={getStatusColor}
                />
              ))}
            </div>
          </div>
        )}

        {/* Bulk Actions */}
        {files.length > 0 && (
          <div className="p-6 border-t border-gray-200 bg-gray-50">
            <div className="flex items-center justify-between">
              <div className="text-sm text-gray-600">
                {files.filter(f => f.status === 'completed').length} of {files.length} files analyzed
              </div>
              <div className="space-x-3">
                <button
                  onClick={() => files.forEach(f => f.status === 'pending' && analyzeFile(f))}
                  className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
                  disabled={!files.some(f => f.status === 'pending')}
                >
                  Analyze All
                </button>
                <button
                  onClick={() => {
                    setFiles([]);
                    setAnalyses({});
                  }}
                  className="px-4 py-2 bg-gray-300 text-gray-700 rounded-md hover:bg-gray-400"
                >
                  Clear All
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// Individual File Analysis Card Component
const FileAnalysisCard = ({ fileItem, onAnalyze, onRemove, getFileIcon, getStatusColor }) => {
  const { file, status, analysis, error } = fileItem;

  return (
    <div className="border border-gray-200 rounded-lg p-4">
      <div className="flex items-start justify-between">
        <div className="flex items-start space-x-3">
          <div className="text-gray-500">
            {getFileIcon(file.type)}
          </div>
          
          <div className="flex-1 min-w-0">
            <h4 className="text-sm font-medium text-gray-900 truncate">
              {file.name}
            </h4>
            <div className="flex items-center space-x-4 mt-1">
              <span className="text-xs text-gray-500">
                {(file.size / 1024 / 1024).toFixed(2)} MB
              </span>
              <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(status)}`}>
                {status.charAt(0).toUpperCase() + status.slice(1)}
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          {status === 'pending' && (
            <button
              onClick={() => onAnalyze(fileItem)}
              className="px-3 py-1 text-xs bg-blue-100 text-blue-700 rounded-md hover:bg-blue-200"
            >
              Analyze
            </button>
          )}
          <button
            onClick={() => onRemove(fileItem.id)}
            className="text-gray-400 hover:text-red-500"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Analysis Results */}
      {analysis && (
        <div className="mt-4 p-4 bg-gray-50 rounded-md">
          <h5 className="text-sm font-semibold text-gray-900 mb-3">Analysis Results</h5>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center">
              <div className={`text-2xl font-bold ${
                analysis.overall_risk_score > 0.7 ? 'text-red-600' :
                analysis.overall_risk_score > 0.4 ? 'text-yellow-600' : 'text-green-600'
              }`}>
                {Math.round(analysis.overall_risk_score * 100)}%
              </div>
              <div className="text-xs text-gray-600">Risk Score</div>
            </div>
            
            <div className="text-center">
              <div className="text-sm font-medium text-gray-900">
                {analysis.threat_category || 'Benign'}
              </div>
              <div className="text-xs text-gray-600">Threat Type</div>
            </div>
            
            <div className="text-center">
              <div className="text-sm font-medium text-gray-900">
                {Math.round(analysis.confidence_score * 100)}%
              </div>
              <div className="text-xs text-gray-600">Confidence</div>
            </div>
          </div>

          {/* Detailed Analysis */}
          {analysis.analysis_details && (
            <div className="mt-4 space-y-3">
              {Object.entries(analysis.analysis_details).map(([type, details]) => (
                <AnalysisDetail key={type} type={type} details={details} />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-md">
          <div className="flex items-center">
            <AlertCircle className="h-4 w-4 text-red-500 mr-2" />
            <span className="text-sm text-red-700">Analysis failed: {error}</span>
          </div>
        </div>
      )}

      {/* Loading State */}
      {status === 'analyzing' && (
        <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-md">
          <div className="flex items-center">
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2"></div>
            <span className="text-sm text-blue-700">Analyzing file...</span>
          </div>
        </div>
      )}
    </div>
  );
};

// Analysis Detail Component
const AnalysisDetail = ({ type, details }) => {
  const getAnalysisIcon = (type) => {
    switch (type) {
      case 'ocr': return <FileText className="h-4 w-4" />;
      case 'phishing': return <AlertCircle className="h-4 w-4" />;
      case 'malware': return <AlertCircle className="h-4 w-4" />;
      default: return <CheckCircle className="h-4 w-4" />;
    }
  };

  const formatAnalysisType = (type) => {
    return type.split('_').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
  };

  return (
    <div className="border border-gray-200 rounded-md p-3">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-2">
          {getAnalysisIcon(type)}
          <span className="text-sm font-medium text-gray-900">
            {formatAnalysisType(type)}
          </span>
        </div>
        
        {details.detected !== undefined && (
          <span className={`px-2 py-1 text-xs font-medium rounded-full ${
            details.detected 
              ? 'bg-red-100 text-red-700' 
              : 'bg-green-100 text-green-700'
          }`}>
            {details.detected ? 'Detected' : 'Clean'}
          </span>
        )}
      </div>

      <div className="space-y-2">
        {details.confidence && (
          <div className="flex justify-between text-sm">
            <span className="text-gray-600">Confidence:</span>
            <span className="font-medium">{Math.round(details.confidence * 100)}%</span>
          </div>
        )}

        {details.extracted_text && (
          <div>
            <span className="text-sm text-gray-600">Extracted Text:</span>
            <p className="text-xs text-gray-800 mt-1 p-2 bg-white rounded border">
              {details.extracted_text.substring(0, 200)}
              {details.extracted_text.length > 200 && '...'}
            </p>
          </div>
        )}

        {details.indicators && details.indicators.length > 0 && (
          <div>
            <span className="text-sm text-gray-600">Indicators:</span>
            <div className="flex flex-wrap gap-1 mt-1">
              {details.indicators.map((indicator, index) => (
                <span
                  key={index}
                  className="px-2 py-1 text-xs bg-yellow-100 text-yellow-800 rounded"
                >
                  {indicator.replace(/_/g, ' ')}
                </span>
              ))}
            </div>
          </div>
        )}

        {details.risk_factors && details.risk_factors.length > 0 && (
          <div>
            <span className="text-sm text-gray-600">Risk Factors:</span>
            <ul className="text-xs text-gray-800 mt-1 space-y-1">
              {details.risk_factors.slice(0, 3).map((factor, index) => (
                <li key={index} className="flex items-center space-x-2">
                  <span className={`w-2 h-2 rounded-full ${
                    factor.severity === 'critical' ? 'bg-red-500' :
                    factor.severity === 'high' ? 'bg-orange-500' :
                    factor.severity === 'medium' ? 'bg-yellow-500' : 'bg-blue-500'
                  }`}></span>
                  <span>{factor.description || factor.type}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};

export default FileUploadAnalyzer;
