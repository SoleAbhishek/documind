import React, { useState, useRef } from 'react';
import {
  Upload,
  FileText,
  Trash2,
  CheckCircle2,
  Clock,
  AlertCircle,
  Loader2,
  Filter,
  FileCode
} from 'lucide-react';

export default function DocumentSidebar({
  documents,
  isUploading,
  onUploadFile,
  onDeleteDocument,
  selectedDocId,
  onSelectDocFilter
}) {
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onUploadFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      onUploadFile(e.target.files[0]);
      e.target.value = '';
    }
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
  };

  return (
    <aside className="w-80 border-r border-slate-800 bg-slate-900/40 flex flex-col h-[calc(100vh-4rem)]">
      {/* Header */}
      <div className="p-4 border-b border-slate-800 flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-white">Knowledge Base</h2>
          <p className="text-xs text-slate-400">
            {documents.length} document{documents.length !== 1 ? 's' : ''} indexed
          </p>
        </div>
        {selectedDocId && (
          <button
            onClick={() => onSelectDocFilter(null)}
            className="text-[11px] text-indigo-400 hover:text-indigo-300 flex items-center space-x-1"
          >
            <Filter className="w-3 h-3" />
            <span>Clear Filter</span>
          </button>
        )}
      </div>

      {/* Upload Dropzone */}
      <div className="p-4">
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-4 text-center cursor-pointer transition-all ${
            dragActive
              ? 'border-indigo-500 bg-indigo-500/10'
              : 'border-slate-700/80 hover:border-slate-600 bg-slate-900/60'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.md,.txt,.csv"
            onChange={handleFileChange}
            className="hidden"
          />
          {isUploading ? (
            <div className="flex flex-col items-center justify-center space-y-2 py-2">
              <Loader2 className="w-6 h-6 text-indigo-400 animate-spin" />
              <p className="text-xs text-indigo-300 font-medium">Uploading to MinIO...</p>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center space-y-1.5 py-1">
              <div className="w-8 h-8 rounded-lg bg-indigo-500/10 flex items-center justify-center text-indigo-400">
                <Upload className="w-4 h-4" />
              </div>
              <p className="text-xs font-semibold text-slate-200">
                Click to upload or drag & drop
              </p>
              <p className="text-[11px] text-slate-400">
                PDF (LlamaParse OCR), Markdown, or Text
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Document List */}
      <div className="flex-1 overflow-y-auto px-4 pb-4 space-y-2.5">
        {documents.length === 0 ? (
          <div className="text-center py-8 text-slate-500 text-xs">
            No documents uploaded yet.
            <br />
            Upload a policy, SLA, or manual above.
          </div>
        ) : (
          documents.map((doc) => {
            const isSelected = selectedDocId === doc.id;
            return (
              <div
                key={doc.id}
                onClick={() => onSelectDocFilter(isSelected ? null : doc.id)}
                className={`group relative p-3 rounded-lg border transition-all cursor-pointer ${
                  isSelected
                    ? 'bg-indigo-950/40 border-indigo-500/60 shadow-md shadow-indigo-950/50'
                    : 'bg-slate-900/60 border-slate-800/80 hover:border-slate-700 hover:bg-slate-900'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-2.5 min-w-0">
                    <div className="mt-0.5 text-slate-400 group-hover:text-indigo-400 transition-colors">
                      {doc.filename.endsWith('.md') ? (
                        <FileCode className="w-4 h-4" />
                      ) : (
                        <FileText className="w-4 h-4" />
                      )}
                    </div>
                    <div className="min-w-0 flex-1">
                      <p className="text-xs font-medium text-slate-200 truncate group-hover:text-white">
                        {doc.filename}
                      </p>
                      <div className="flex items-center space-x-2 mt-1">
                        <span className="text-[10px] text-slate-400">
                          {formatFileSize(doc.file_size_bytes)}
                        </span>
                        {doc.num_chunks > 0 && (
                          <span className="text-[10px] px-1.5 py-0.2 rounded bg-slate-800 text-slate-300 font-mono">
                            {doc.num_chunks} chunks
                          </span>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Delete Button */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      if (confirm(`Delete document "${doc.filename}"? This will delete all Qdrant vectors and storage blobs.`)) {
                        onDeleteDocument(doc.id);
                      }
                    }}
                    className="opacity-0 group-hover:opacity-100 p-1 text-slate-500 hover:text-rose-400 hover:bg-rose-500/10 rounded transition-all"
                    title="Delete Document"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>

                {/* Status Indicator */}
                <div className="mt-2.5 flex items-center justify-between text-[11px]">
                  {doc.status === 'INDEXED' && (
                    <span className="inline-flex items-center text-emerald-400 space-x-1 font-medium">
                      <CheckCircle2 className="w-3 h-3" />
                      <span>Indexed</span>
                    </span>
                  )}
                  {doc.status === 'PARSING' && (
                    <span className="inline-flex items-center text-cyan-400 space-x-1 font-medium">
                      <Loader2 className="w-3 h-3 animate-spin" />
                      <span>OCR & Parsing...</span>
                    </span>
                  )}
                  {doc.status === 'PENDING' && (
                    <span className="inline-flex items-center text-amber-400 space-x-1 font-medium">
                      <Clock className="w-3 h-3" />
                      <span>Queued</span>
                    </span>
                  )}
                  {doc.status === 'FAILED' && (
                    <span className="inline-flex items-center text-rose-400 space-x-1 font-medium" title={doc.error_message}>
                      <AlertCircle className="w-3 h-3" />
                      <span>Failed</span>
                    </span>
                  )}

                  {isSelected && (
                    <span className="text-[10px] font-semibold text-indigo-400 uppercase tracking-wider">
                      Filtering
                    </span>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>
    </aside>
  );
}
