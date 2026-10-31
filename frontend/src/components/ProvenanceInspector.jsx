import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import {
  FileText,
  Compass,
  Zap,
  Cpu,
  Layers,
  ChevronRight,
  ExternalLink,
  Info
} from 'lucide-react';

export default function ProvenanceInspector({
  citations,
  selectedSourceId,
  onSelectCitation,
  documentsMap
}) {
  return (
    <aside className="w-96 border-l border-slate-800 bg-slate-900/60 flex flex-col h-[calc(100vh-4rem)]">
      {/* Header */}
      <div className="p-4 border-b border-slate-800 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Compass className="w-4 h-4 text-indigo-400" />
          <h2 className="text-sm font-semibold text-white">Provenance Inspector</h2>
        </div>
        <span className="text-[11px] font-mono text-slate-400 bg-slate-800/80 px-2 py-0.5 rounded border border-slate-700">
          {citations ? citations.length : 0} Sources
        </span>
      </div>

      {/* RAG Explanation Tooltip Banner */}
      <div className="p-3 bg-indigo-950/20 border-b border-indigo-500/10 text-slate-400 text-[11px] flex items-start space-x-2">
        <Info className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />
        <p>
          Candidates are retrieved via <strong>Qdrant RRF</strong> (Dense + BM25) and rescored by the <strong>Cross-Encoder</strong> transformer model.
        </p>
      </div>

      {/* Citations List */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3.5">
        {!citations || citations.length === 0 ? (
          <div className="text-center py-16 text-slate-500 text-xs flex flex-col items-center justify-center space-y-2">
            <Layers className="w-8 h-8 text-slate-600" />
            <p className="max-w-[200px]">
              Ask a question to inspect retrieved source chunks and attention scores.
            </p>
          </div>
        ) : (
          citations.map((c, idx) => {
            const isSelected = selectedSourceId === c.source_id;
            const rerankScore = typeof c.rerank_score === 'number' ? c.rerank_score : null;
            const isHighConfidence = rerankScore !== null && rerankScore > 0;
            const doc = documentsMap ? documentsMap[c.document_id] : null;
            const docName = doc ? doc.filename : c.document_id || 'Document';

            return (
              <div
                key={idx}
                id={`citation-${c.source_id}`}
                onClick={() => onSelectCitation(c.source_id)}
                className={`p-3.5 rounded-xl border transition-all cursor-pointer ${
                  isSelected
                    ? 'bg-indigo-950/50 border-indigo-500 shadow-lg shadow-indigo-950/50'
                    : 'bg-slate-900 border-slate-800 hover:border-slate-700'
                }`}
              >
                {/* Source Badge & Document Name */}
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2 min-w-0">
                    <span className="text-[11px] font-bold px-2 py-0.5 rounded bg-indigo-600/30 text-indigo-300 border border-indigo-500/40 font-mono">
                      {c.source_id}
                    </span>
                    <span className="text-xs font-semibold text-slate-200 truncate" title={docName}>
                      {docName}
                    </span>
                  </div>
                  <span className="text-[10px] text-slate-400 shrink-0 font-medium">
                    Page {c.page_number || 1}
                  </span>
                </div>

                {/* Breadcrumb Hierarchy */}
                {c.header_path && (
                  <div className="text-[11px] text-indigo-300/90 font-medium mb-2 flex items-center space-x-1 overflow-hidden text-ellipsis whitespace-nowrap">
                    <span className="text-slate-500 text-[10px]">Section:</span>
                    <span className="truncate">{c.header_path}</span>
                  </div>
                )}

                {/* Score Breakdown Pills */}
                <div className="grid grid-cols-2 gap-2 my-2.5">
                  {/* RRF Score */}
                  <div className="bg-slate-950/80 rounded-lg p-2 border border-slate-800">
                    <div className="flex items-center justify-between text-[10px] text-slate-400 mb-0.5">
                      <span className="flex items-center space-x-1">
                        <Zap className="w-3 h-3 text-cyan-400" />
                        <span>RRF Score</span>
                      </span>
                    </div>
                    <span className="text-xs font-mono font-bold text-cyan-300">
                      {c.rrf_score !== undefined && c.rrf_score !== null
                        ? Number(c.rrf_score).toFixed(4)
                        : 'N/A'}
                    </span>
                  </div>

                  {/* Cross-Encoder Score */}
                  <div className="bg-slate-950/80 rounded-lg p-2 border border-slate-800">
                    <div className="flex items-center justify-between text-[10px] text-slate-400 mb-0.5">
                      <span className="flex items-center space-x-1">
                        <Cpu className="w-3 h-3 text-amber-400" />
                        <span>Rerank Score</span>
                      </span>
                    </div>
                    <span
                      className={`text-xs font-mono font-bold ${
                        isHighConfidence ? 'text-emerald-400' : 'text-slate-400'
                      }`}
                    >
                      {rerankScore !== null ? rerankScore.toFixed(4) : 'N/A'}
                    </span>
                  </div>
                </div>

                {/* Content Snippet */}
                <div className="bg-slate-950/50 rounded-lg p-2.5 border border-slate-800/80 text-[11px] text-slate-300 leading-relaxed font-sans max-h-56 overflow-y-auto">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    rehypePlugins={[rehypeRaw]}
                    components={{
                      table: ({ children }) => (
                        <div className="overflow-x-auto my-1.5 rounded border border-slate-700/80 bg-slate-900/60">
                          <table className="min-w-full divide-y divide-slate-700/80 text-[10px]">
                            {children}
                          </table>
                        </div>
                      ),
                      th: ({ children }) => (
                        <th className="px-2 py-1 text-left font-semibold text-indigo-300 bg-slate-800/80 uppercase tracking-wider text-[9px]">
                          {children}
                        </th>
                      ),
                      td: ({ children }) => (
                        <td className="px-2 py-1 text-slate-300 border-t border-slate-800 whitespace-pre-wrap">
                          {children}
                        </td>
                      ),
                      p: ({ children }) => <p className="mb-1">{children}</p>
                    }}
                  >
                    {c.snippet}
                  </ReactMarkdown>
                </div>

                {/* Qdrant Point UUID */}
                {c.point_id && (
                  <div className="mt-2 text-[9px] font-mono text-slate-500 truncate" title={`Qdrant Point: ${c.point_id}`}>
                    Point ID: {c.point_id}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </aside>
  );
}
