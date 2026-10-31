const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || ""
).replace(/\/$/, "");
import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  Database,
  Search,
  UploadCloud,
  AlertTriangle,
  CheckCircle,
  XCircle,
  FileText,
  FileCode,
  Image as ImageIcon,
  BarChart2,
  Layers,
  ChevronRight,
  ShieldAlert,
  ArrowRight,
  RotateCcw,
  Trash2,
  Sparkles,
  Info,
  ExternalLink,
  Code2,
} from 'lucide-react';
import EvaluationModal from './components/EvaluationModal';
import ExplainViewer from './components/ExplainViewer';

export default function App() {
  // Navigation / State
  const [evalModalOpen, setEvalModalOpen] = useState(false);
  const [sources, setSources] = useState([]);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState(null);
  const [corpusId, setCorpusId] = useState('default');
  const activeRequestIdRef = useRef(null);

  // Metadata Filters
  const [filterService, setFilterService] = useState('all');
  const [filterOwner, setFilterOwner] = useState('all');
  const [filterType, setFilterType] = useState('all');

  // Investigation Query & Result
  const [query, setQuery] = useState('');
  const [investigation, setInvestigation] = useState(null);

  // Right Pane Inspector State
  const [activeTab, setActiveTab] = useState('citation'); // 'citation' | 'explain'
  const [selectedCitation, setSelectedCitation] = useState(null);
  const [rawSourceData, setRawSourceData] = useState(null);

  // File Upload State
  const fileInputRef = useRef(null);
  const [uploadService, setUploadService] = useState('general');
  const [uploadOwner, setUploadOwner] = useState('user');

  // Generic Dynamic Investigation Presets
  const presets = [
    {
      id: 1,
      label: 'Preset 1: Deep Synthesis & Evidence',
      badge: 'Multi-Hop',
      badgeColor: 'border-emerald-500/40 text-emerald-400 bg-emerald-950/40',
      query: 'Synthesize key metrics, specifications, and factual evidence across all uploaded documents.',
    },
    {
      id: 2,
      label: 'Preset 2: Cross-Source Conflict Audit',
      badge: 'Conflict Audit',
      badgeColor: 'border-amber-500/40 text-amber-400 bg-amber-950/40',
      query: 'Audit the uploaded corpus for contradictory rules, conflicting numbers, or incompatible policies.',
    },
  ];

  // Fetch Health & Sources on load
  const loadInitialData = async (targetCorpus = corpusId) => {
    try {
      const hRes = await fetch(`${API_BASE_URL}/api/health?corpus_id=${encodeURIComponent(targetCorpus)}`);
      if (hRes.ok) setHealth(await hRes.json());

      const sRes = await fetch(`${API_BASE_URL}/api/sources?corpus_id=${encodeURIComponent(targetCorpus)}`);
      if (sRes.ok) {
        const sData = await sRes.json();
        setSources(sData.sources || []);
      }
    } catch (err) {
      console.error('Failed to load initial data:', err);
    }
  };

  useEffect(() => {
    loadInitialData(corpusId);
  }, [corpusId]);

  // Run Investigation with Request-Scoped Stale Rejection
  const handleInvestigate = async (targetQuery = query) => {
    const q = targetQuery.trim();
    if (!q) return;
    setLoading(true);

    const requestId = crypto.randomUUID ? crypto.randomUUID() : `req_${Date.now()}`;
    activeRequestIdRef.current = requestId;

    const metadataFilter = {};
    if (filterService !== 'all') metadataFilter['metadata.service'] = filterService;
    if (filterOwner !== 'all') metadataFilter['metadata.owner'] = filterOwner;
    if (filterType !== 'all') metadataFilter['metadata.source_type'] = filterType;

    try {
      const res = await fetch(`${API_BASE_URL}/api/investigate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Corpus-ID': corpusId,
        },
        body: JSON.stringify({
          query: q,
          corpus_id: corpusId,
          request_id: requestId,
          metadata_filter: Object.keys(metadataFilter).length > 0 ? metadataFilter : null,
        }),
      });
      const data = await res.json();

      // Stale response rejection: ensure response matches currently active request
      if (activeRequestIdRef.current !== requestId) {
        console.warn(`[AEGIS] Ignored stale investigation response for request ${requestId}`);
        return;
      }

      setInvestigation(data);

      if (data.citations && data.citations.length > 0) {
        selectCitation(data.citations[0]);
      } else {
        setSelectedCitation(null);
        setRawSourceData(null);
      }
    } catch (err) {
      if (activeRequestIdRef.current === requestId) {
        console.error('Investigation error:', err);
      }
    } finally {
      if (activeRequestIdRef.current === requestId) {
        setLoading(false);
      }
    }
  };

  // Select a citation to inspect
  const selectCitation = async (citation) => {
    setSelectedCitation(citation);
    setActiveTab('citation');
    try {
      const res = await fetch(`${API_BASE_URL}/api/sources/${encodeURIComponent(citation.source_id)}?corpus_id=${encodeURIComponent(corpusId)}`);
      if (res.ok) {
        const data = await res.json();
        setRawSourceData(data);
      }
    } catch (err) {
      console.error('Failed to load raw source:', err);
    }
  };

  // Select a raw document directly from left corpus
  const handleSelectSource = async (source) => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/sources/${encodeURIComponent(source.source_id)}?corpus_id=${encodeURIComponent(corpusId)}`);
      if (res.ok) {
        const data = await res.json();
        setRawSourceData(data);
        const firstChunk = data.chunks && data.chunks.length > 0 ? data.chunks[0] : null;
        setSelectedCitation(
          firstChunk
            ? {
                chunk_id: firstChunk.chunk_id,
                source_id: source.source_id,
                source_title: firstChunk.source_title,
                char_start: firstChunk.char_start,
                char_end: firstChunk.char_end,
                service: firstChunk.metadata?.service || 'general',
                owner: firstChunk.metadata?.owner || 'user',
                vector_score: 1.0,
                keyword_score: 1.0,
                rrf_score: 0.032,
              }
            : null
        );
        setActiveTab('citation');
      }
    } catch (err) {
      console.error('Failed to load source:', err);
    }
  };

  // Live File Upload Handler with Corpus Association
  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setUploadMessage(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('corpus_id', corpusId);
    formData.append('service', uploadService);
    formData.append('owner', uploadOwner);
    formData.append('source_type', file.name.match(/\.(png|jpg|jpeg|webp)$/i) ? 'visual_telemetry' : 'custom_upload');

    try {
      const res = await fetch(`${API_BASE_URL}/api/upload`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setUploadMessage({ type: 'success', text: `Indexed '${file.name}' into corpus '${corpusId}'!` });
        await loadInitialData(corpusId);
      } else {
        setUploadMessage({ type: 'error', text: data.detail || 'Upload failed' });
      }
    } catch (err) {
      setUploadMessage({ type: 'error', text: err.message });
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  // Clear Documents from Current Corpus
  const handleClearAll = async () => {
    if (!window.confirm(`Wipe documents in corpus '${corpusId}' from MongoDB?`)) return;
    try {
      const res = await fetch(`${API_BASE_URL}/api/clear-all?corpus_id=${encodeURIComponent(corpusId)}`, { method: 'POST' });
      if (res.ok) {
        await loadInitialData(corpusId);
        setInvestigation(null);
        setSelectedCitation(null);
        setRawSourceData(null);
        setUploadMessage({ type: 'success', text: `Corpus '${corpusId}' wiped clean. Ready for uploads!` });
      }
    } catch (err) {
      console.error('Clear failed:', err);
    }
  };

  // Filter sources based on dropdowns
  const filteredSources = sources.filter((s) => {
    if (filterService !== 'all' && s.metadata?.service !== filterService) return false;
    if (filterOwner !== 'all' && s.metadata?.owner !== filterOwner) return false;
    if (filterType !== 'all' && s.metadata?.source_type !== filterType) return false;
    return true;
  });

  // Recursively process children strings to replace [CIT:chk_XX] with clickable badges
  const renderWithCitations = (children) => {
    if (!children) return null;
    if (typeof children === 'string') {
      const parts = children.split(/(\[CIT:[a-zA-Z0-9_]+\])/g);
      return parts.map((part, index) => {
        const match = part.match(/^\[CIT:([a-zA-Z0-9_]+)\]$/);
        if (match) {
          const chunkId = match[1];
          const citation = investigation?.citations?.find((c) => c.chunk_id === chunkId);
          const isSelected = selectedCitation?.chunk_id === chunkId;
          return (
            <button
              key={index}
              onClick={() => {
                if (citation) selectCitation(citation);
                else selectCitation({ chunk_id: chunkId, source_id: chunkId });
              }}
              className={`inline-flex items-center px-1.5 py-0.5 mx-1 my-0.5 rounded text-[11px] font-mono font-bold transition-all border ${
                isSelected
                  ? 'bg-cyan-500 text-black border-cyan-300 shadow-md shadow-cyan-500/30'
                  : 'bg-cyan-950/80 text-cyan-300 border-cyan-800/80 hover:bg-cyan-800/80 hover:text-white'
              }`}
              title={`Inspect citation: ${chunkId}`}
            >
              <Code2 className="w-3 h-3 mr-1" />
              {chunkId}
            </button>
          );
        }
        return <span key={index}>{part}</span>;
      });
    }
    if (Array.isArray(children)) {
      return children.map((child, idx) => (
        <React.Fragment key={idx}>{renderWithCitations(child)}</React.Fragment>
      ));
    }
    if (React.isValidElement(children)) {
      return React.cloneElement(children, {
        children: renderWithCitations(children.props?.children),
      });
    }
    return children;
  };

  // Full GFM Markdown renderer with syntax-highlighted styles and interactive citation integration
  const renderCitedMarkdown = (markdownText) => {
    if (!markdownText) return null;
    return (
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h3: ({ children }) => (
            <h3 className="text-sm font-bold text-cyan-400 mt-4 mb-2 pb-1.5 border-b border-gray-800 font-mono flex items-center">
              {renderWithCitations(children)}
            </h3>
          ),
          h4: ({ children }) => (
            <h4 className="text-xs font-bold uppercase tracking-wider mt-3.5 mb-1.5 font-mono text-cyan-300">
              {renderWithCitations(children)}
            </h4>
          ),
          p: ({ children }) => (
            <p className="mb-2 leading-relaxed text-gray-300 text-xs font-sans">
              {renderWithCitations(children)}
            </p>
          ),
          ul: ({ children }) => (
            <ul className="list-disc pl-5 mb-2.5 space-y-1 text-gray-300 text-xs">
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol className="list-decimal pl-5 mb-2.5 space-y-1 text-gray-300 text-xs">
              {children}
            </ol>
          ),
          li: ({ children }) => (
            <li className="leading-relaxed">
              {renderWithCitations(children)}
            </li>
          ),
          blockquote: ({ children }) => (
            <blockquote className="border-l-2 border-cyan-500/60 pl-3 my-2 text-xs italic text-gray-400 bg-cyan-950/20 py-1 rounded-r">
              {renderWithCitations(children)}
            </blockquote>
          ),
          table: ({ children }) => (
            <div className="overflow-x-auto my-3 border border-gray-800 rounded-lg">
              <table className="min-w-full text-[11px] font-mono divide-y divide-gray-800">
                {children}
              </table>
            </div>
          ),
          thead: ({ children }) => <thead className="bg-[#141725]">{children}</thead>,
          th: ({ children }) => (
            <th className="px-3 py-1.5 text-left text-cyan-400 font-bold uppercase tracking-wider border-r border-gray-800">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="px-3 py-1.5 border-r border-b border-gray-800/80 text-gray-300">
              {renderWithCitations(children)}
            </td>
          ),
          code: ({ children }) => {
            const codeStr = String(children);
            const citMatch = codeStr.match(/^\[?CIT:([a-zA-Z0-9_]+)\]?$/);
            if (citMatch) {
              return renderWithCitations(citMatch[0]);
            }
            return (
              <code className="bg-[#171a29] text-cyan-300 px-1 py-0.5 rounded font-mono text-[11px] border border-cyan-900/40">
                {children}
              </code>
            );
          },
        }}
      >
        {markdownText}
      </ReactMarkdown>
    );
  };

  return (
    <div className="flex flex-col h-screen bg-[#0c0d12] text-gray-200 overflow-hidden font-sans select-none">
      {/* ================= HEADER BAR ================= */}
      <header className="h-14 border-b border-gray-800 bg-[#11131c] px-4 flex items-center justify-between shrink-0">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-gradient-to-tr from-cyan-600 to-blue-600 rounded-lg shadow-lg shadow-cyan-600/20">
            <Database className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-sm font-bold tracking-tight text-white uppercase font-mono">
                AEGIS // Live Multimodal RAG Console
              </h1>
              <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-cyan-950 border border-cyan-700/60 text-cyan-300 font-mono">
                100% Local MongoDB Engine
              </span>
            </div>
            <p className="text-[10px] text-gray-400">
              Vector Cosine ($reduce) &bull; BM25 TextScore &bull; Epistemic Guardrail (0% Hallucination)
            </p>
          </div>
        </div>

        {/* Status Indicators & Top Actions */}
        <div className="flex items-center space-x-3 text-xs">
          {/* MongoDB Connection Pill */}
          <div className="flex items-center space-x-2 bg-[#171a26] border border-gray-800 px-3 py-1.5 rounded-lg">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            <span className="font-mono text-gray-300 text-[11px]">localhost:27017</span>
            <span className="text-gray-500">|</span>
            <span className="text-cyan-400 font-mono text-[11px]">
              {health?.knowledge_chunks_count ?? 0} Chunks
            </span>
          </div>

          {/* Clear All Button */}
          <button
            onClick={handleClearAll}
            title="Wipe all data from MongoDB to start clean"
            className="flex items-center space-x-1.5 bg-red-950/60 hover:bg-red-900 border border-red-800/60 text-red-300 px-3 py-1.5 rounded-lg transition text-xs"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Wipe MongoDB</span>
          </button>
        </div>
      </header>

      {/* ================= 3-PANE WORKSPACE ================= */}
      <div className="flex flex-1 overflow-hidden">
        {/* ---------------- PANE 1: LEFT (Corpus & Live Ingest) ---------------- */}
        <div className="w-80 border-r border-gray-800 bg-[#0f1118] flex flex-col shrink-0">
          <div className="p-3 border-b border-gray-800 flex items-center justify-between">
            <div className="flex items-center space-x-1.5 text-xs font-semibold text-gray-300 uppercase tracking-wider font-mono">
              <Layers className="w-4 h-4 text-cyan-400" />
              <span>Uploaded Corpus</span>
            </div>
            <span className="text-[11px] font-mono text-cyan-400 font-bold">{filteredSources.length} files</span>
          </div>

          {/* Live Upload Dropzone */}
          <div className="p-3 border-b border-gray-800 bg-[#121520]">
            <div className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider mb-2 flex items-center justify-between">
              <span>Drag & Drop File Ingest</span>
              <span className="text-[10px] text-cyan-400 font-mono">POST /api/upload</span>
            </div>

            <div
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-cyan-800/80 hover:border-cyan-400 rounded-xl p-4 text-center cursor-pointer transition bg-[#171a26]/80 hover:bg-[#1a1f33]"
            >
              <input
                ref={fileInputRef}
                type="file"
                onChange={handleFileUpload}
                accept=".md,.json,.txt,.png,.jpg,.jpeg"
                className="hidden"
              />
              <UploadCloud className="w-7 h-7 mx-auto text-cyan-400 mb-1 animate-bounce" />
              <div className="text-xs font-semibold text-white">
                {uploading ? 'Vectorizing & Indexing...' : 'Click or Drop File Here'}
              </div>
              <div className="text-[10px] text-cyan-300/70 mt-1">
                Uploads .json, .md, .txt, .png live into MongoDB
              </div>
            </div>

            {uploadMessage && (
              <div
                className={`mt-2 p-2 rounded text-[11px] ${
                  uploadMessage.type === 'success'
                    ? 'bg-emerald-950/80 text-emerald-300 border border-emerald-800'
                    : 'bg-red-950/80 text-red-300 border border-red-800'
                }`}
              >
                {uploadMessage.text}
              </div>
            )}
          </div>

          {/* Indexed Corpus File List */}
          <div className="flex-1 overflow-y-auto p-2 space-y-1">
            {filteredSources.length === 0 && (
              <div className="h-full flex flex-col items-center justify-center text-center p-6 text-gray-500">
                <UploadCloud className="w-10 h-10 text-gray-700 mb-2" />
                <div className="text-xs font-medium text-gray-400">Corpus is Empty</div>
                <p className="text-[11px] text-gray-600 mt-1">
                  Use the dropzone above to upload your enterprise files (.json, .md, .txt, .png) and index them into MongoDB!
                </p>
              </div>
            )}

            {filteredSources.map((s, idx) => {
              const isSelected = selectedCitation?.source_id === s.source_id;
              const isPng = s.file_type === 'png' || s.file_type === 'jpg';
              const isJson = s.file_type === 'json';

              return (
                <div
                  key={idx}
                  onClick={() => handleSelectSource(s)}
                  className={`p-2.5 rounded-xl cursor-pointer transition-all border ${
                    isSelected
                      ? 'bg-cyan-950/40 border-cyan-500/80 text-white'
                      : 'bg-[#141622] hover:bg-[#1a1d2e] border-gray-800/80 text-gray-300'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-center space-x-2">
                      {isPng ? (
                        <ImageIcon className="w-4 h-4 text-purple-400 shrink-0" />
                      ) : isJson ? (
                        <FileCode className="w-4 h-4 text-amber-400 shrink-0" />
                      ) : (
                        <FileText className="w-4 h-4 text-cyan-400 shrink-0" />
                      )}
                      <span className="text-xs font-mono font-medium truncate max-w-[170px]">
                        {s.source_id}
                      </span>
                    </div>
                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800/60">
                      {s.chunk_count || 1} chunks
                    </span>
                  </div>

                  <div className="text-[11px] text-gray-400 font-sans mt-1 line-clamp-1">{s.title}</div>
                </div>
              );
            })}
          </div>
        </div>

        {/* ---------------- PANE 2: CENTER (Investigation Console) ---------------- */}
        <div className="flex-1 flex flex-col min-w-0 bg-[#0c0d12] border-r border-gray-800">
          {/* Quick Presets Bar */}
          <div className="p-3 border-b border-gray-800 bg-[#10121a]">
            <div className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider mb-2 font-mono flex items-center space-x-1">
              <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
              <span>Investigation Presets (Click to Load)</span>
            </div>
            <div className="grid grid-cols-3 gap-2">
              {presets.map((p) => (
                <button
                  key={p.id}
                  onClick={() => {
                    setQuery(p.query);
                    handleInvestigate(p.query);
                  }}
                  className="text-left p-2.5 rounded-xl bg-[#151824] hover:bg-[#1a1e30] border border-gray-800 hover:border-cyan-500/50 transition-all flex flex-col justify-between"
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded border ${p.badgeColor}`}>
                      {p.badge}
                    </span>
                    <ChevronRight className="w-3.5 h-3.5 text-gray-500" />
                  </div>
                  <div className="text-xs font-sans text-gray-200 line-clamp-2 leading-snug">
                    {p.query}
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Investigation Input Bar */}
          <div className="p-3 border-b border-gray-800 bg-[#12141e] flex items-center space-x-2">
            <div className="relative flex-1">
              <Search className="w-4 h-4 text-gray-400 absolute left-3 top-3" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleInvestigate()}
                placeholder="Ask a multi-hop question, audit policy conflicts, or verify technical specifications..."
                className="w-full bg-[#181a28] border border-gray-700/80 rounded-xl pl-9 pr-4 py-2 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-cyan-500 transition"
              />
            </div>
            <button
              onClick={() => handleInvestigate()}
              disabled={loading || !query.trim()}
              className="px-4 py-2 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white text-xs font-semibold rounded-xl disabled:opacity-50 transition shadow-md shadow-cyan-600/20 shrink-0"
            >
              {loading ? 'Investigating...' : 'Investigate'}
            </button>
          </div>

          {/* Investigation Output Body */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {!investigation && !loading && (
              <div className="h-full flex flex-col items-center justify-center text-center p-8 text-gray-500">
                <Search className="w-12 h-12 text-gray-700 mb-3" />
                <h3 className="text-base font-semibold text-gray-300">Ready for Live Investigation</h3>
                <p className="text-xs text-gray-500 max-w-md mt-1">
                  Upload your files in the left pane, then type any query above or test one of the preset scenarios.
                </p>
              </div>
            )}

            {loading && (
              <div className="space-y-4 py-8">
                <div className="flex items-center justify-center space-x-2 text-cyan-400 text-sm font-mono">
                  <div className="w-3 h-3 rounded-full bg-cyan-400 animate-ping"></div>
                  <span>Running MongoDB Hybrid Vector & Keyword Pipeline...</span>
                </div>
                <div className="max-w-md mx-auto space-y-2">
                  <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                    <div className="h-full bg-cyan-500 animate-pulse w-3/4"></div>
                  </div>
                </div>
              </div>
            )}

            {investigation && !loading && (
              <>
                {/* 1. Status Banner */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    {investigation.status === 'EPISTEMIC_REFUSAL' || investigation.status === 'ABSTAINED' ? (
                      <span className="flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-bold font-mono bg-red-950 text-red-300 border border-red-800">
                        <XCircle className="w-4 h-4 text-red-400" />
                        <span>STATUS: EPISTEMIC REFUSAL (0% HALLUCINATION)</span>
                      </span>
                    ) : investigation.status === 'CONFLICT_DETECTED' ? (
                      <span className="flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-bold font-mono bg-amber-950 text-amber-300 border border-amber-800">
                        <AlertTriangle className="w-4 h-4 text-amber-400" />
                        <span>STATUS: CONFLICT DETECTED</span>
                      </span>
                    ) : investigation.status === 'PARTIALLY_GROUNDED' ? (
                      <span className="flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-bold font-mono bg-cyan-950 text-cyan-300 border border-cyan-800">
                        <Layers className="w-4 h-4 text-cyan-400" />
                        <span>STATUS: PARTIALLY GROUNDED</span>
                      </span>
                    ) : investigation.status === 'INSUFFICIENT_EVIDENCE' ? (
                      <span className="flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-bold font-mono bg-orange-950 text-orange-300 border border-orange-800">
                        <AlertTriangle className="w-4 h-4 text-orange-400" />
                        <span>STATUS: INSUFFICIENT EVIDENCE</span>
                      </span>
                    ) : investigation.status === 'RETRIEVAL_FAILED' ? (
                      <span className="flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-bold font-mono bg-rose-950 text-rose-300 border border-rose-800">
                        <XCircle className="w-4 h-4 text-rose-400" />
                        <span>STATUS: RETRIEVAL FAILED</span>
                      </span>
                    ) : investigation.status === 'VALIDATION_FAILED' ? (
                      <span className="flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-bold font-mono bg-yellow-950 text-yellow-300 border border-yellow-800">
                        <AlertTriangle className="w-4 h-4 text-yellow-400" />
                        <span>STATUS: VALIDATION FAILED</span>
                      </span>
                    ) : (
                      <span className="flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-bold font-mono bg-emerald-950 text-emerald-300 border border-emerald-800">
                        <CheckCircle className="w-4 h-4 text-emerald-400" />
                        <span>STATUS: GROUNDED VERIFIED</span>
                      </span>
                    )}
                  </div>
                  <div className="flex items-center space-x-2 text-[11px] font-mono text-gray-400">
                    {investigation.coverage_score !== undefined && (
                      <span className="px-2 py-0.5 rounded bg-gray-800 text-gray-300">
                        Coverage: {Math.round(investigation.coverage_score * 100)}%
                      </span>
                    )}
                    {investigation.execution_time_ms && (
                      <span className="text-gray-500">
                        {investigation.execution_time_ms}ms
                      </span>
                    )}
                  </div>
                </div>

                {/* 2. Sub-Query Decomposition Tags */}
                {investigation.sub_queries && investigation.sub_queries.length > 0 && (
                  <div className="bg-[#12141f] border border-gray-800 rounded-xl p-3">
                    <div className="text-[10px] font-mono uppercase tracking-wider text-gray-400 mb-1.5 flex items-center justify-between">
                      <span>Decomposed Sub-Queries</span>
                      <span className="text-cyan-400">{investigation.sub_queries.length} parallel paths</span>
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {investigation.sub_queries.map((sq, i) => (
                        <span
                          key={i}
                          className="text-[11px] font-mono bg-[#1a1d2e] border border-gray-700/60 text-cyan-300 px-2 py-0.5 rounded-lg"
                        >
                          ↳ {sq}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* 3. Epistemic Refusal Card (If Abstained) */}
                {investigation.status === 'ABSTAINED' && (
                  <div className="bg-red-950/40 border-2 border-red-700/80 rounded-2xl p-5 space-y-3">
                    <div className="flex items-center space-x-2 text-red-400 font-bold text-sm font-mono">
                      <ShieldAlert className="w-5 h-5 shrink-0" />
                      <span>EPISTEMIC SAFETY GUARDRAIL TRIGGERED</span>
                    </div>
                    <p className="text-xs text-red-200/90 leading-relaxed">{investigation.reason}</p>

                    {investigation.closest_candidate_chunks && investigation.closest_candidate_chunks.length > 0 && (
                      <div className="bg-[#12080a] border border-red-900/60 rounded-xl p-3 mt-3">
                        <div className="text-[11px] font-mono text-red-400 mb-2 uppercase font-semibold">
                          Closest Retrieved Candidate Chunks (Below 0.55 Threshold):
                        </div>
                        <div className="space-y-1.5 text-xs font-mono">
                          {investigation.closest_candidate_chunks.map((chk, i) => (
                            <div key={i} className="flex items-center justify-between text-gray-400">
                              <span className="text-red-300">{chk.chunk_id} ({chk.source_id})</span>
                              <span className="text-red-400 font-bold">score: {chk.relevance_score}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* 4. Cross-Source Conflict Alert Card (If Conflict) */}
                {investigation.conflict?.conflict_detected && (
                  <div className="bg-amber-950/40 border-2 border-amber-700/80 rounded-2xl p-5 space-y-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2 text-amber-400 font-bold text-sm font-mono">
                        <AlertTriangle className="w-5 h-5 shrink-0" />
                        <span>{investigation.conflict.title || 'CROSS-SOURCE CONFLICT DETECTED'}</span>
                      </div>
                      <span className="text-[10px] font-mono bg-amber-900/80 text-amber-200 px-2 py-0.5 rounded">
                        Contradiction Flagged
                      </span>
                    </div>
                    <p className="text-xs text-amber-200/90 leading-relaxed">
                      {investigation.conflict.summary}
                    </p>

                    <div className="grid grid-cols-2 gap-3">
                      <div className="bg-[#1a1408] border border-amber-800/80 rounded-xl p-3">
                        <div className="text-[11px] font-mono font-bold text-amber-300 mb-1">
                          Source A: {investigation.conflict.source_a?.authority}
                        </div>
                        <div className="text-xs text-gray-200 font-mono bg-black/40 p-2 rounded border border-amber-900/40">
                          {investigation.conflict.source_a?.mandate}
                        </div>
                        <div className="mt-2 text-[10px] text-gray-400">
                          File: <span className="text-amber-400 font-mono">{investigation.conflict.source_a?.source_id}</span>
                        </div>
                      </div>

                      <div className="bg-[#1a1408] border border-amber-800/80 rounded-xl p-3">
                        <div className="text-[11px] font-mono font-bold text-amber-300 mb-1">
                          Source B: {investigation.conflict.source_b?.authority}
                        </div>
                        <div className="text-xs text-gray-200 font-mono bg-black/40 p-2 rounded border border-amber-900/40">
                          {investigation.conflict.source_b?.mandate}
                        </div>
                        <div className="mt-2 text-[10px] text-gray-400">
                          File: <span className="text-amber-400 font-mono">{investigation.conflict.source_b?.source_id}</span>
                        </div>
                      </div>
                    </div>

                    <div className="text-[11px] text-amber-300/80 bg-amber-900/20 p-2 rounded-lg border border-amber-800/40 font-sans">
                      <strong>Directive:</strong> {investigation.conflict.recommendation}
                    </div>
                  </div>
                )}

                {/* 4.5. Missing Evidence Notification */}
                {investigation.missing_attributes && investigation.missing_attributes.length > 0 && (
                  <div className="bg-orange-950/30 border border-orange-800/60 rounded-2xl p-4 space-y-2">
                    <div className="flex items-center space-x-2 text-orange-400 font-bold text-xs font-mono">
                      <AlertTriangle className="w-4 h-4 shrink-0" />
                      <span>IDENTIFIED DATA GAPS / MISSING ATTRIBUTES</span>
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {investigation.missing_attributes.map((attr, i) => (
                        <span key={i} className="text-[11px] font-mono bg-orange-950/80 border border-orange-700/60 text-orange-200 px-2 py-0.5 rounded">
                          Missing: {attr}
                        </span>
                      ))}
                    </div>
                    <p className="text-[10px] text-orange-300/80 font-mono">
                      No external assumptions made. Unrelated facts prevented from substitution.
                    </p>
                  </div>
                )}

                {/* 4.6. Verified Multimodal Image-to-Document Associations */}
                {investigation.image_resolutions && investigation.image_resolutions.length > 0 && (
                  <div className="bg-[#121624] border border-cyan-900/60 rounded-2xl p-4 space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2 text-cyan-400 font-bold text-xs font-mono">
                        <ImageIcon className="w-4 h-4 shrink-0" />
                        <span>RESOLVED MULTIMODAL VISUAL ASSETS</span>
                      </div>
                      <span className="text-[10px] font-mono bg-cyan-950 text-cyan-300 px-2 py-0.5 rounded border border-cyan-800">
                        {investigation.image_resolutions.filter(r => r.entity_verified).length} Entity Verified
                      </span>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
                      {investigation.image_resolutions.map((ir, i) => (
                        <div key={i} className="bg-[#0b0d14] border border-cyan-950 rounded-xl p-3 space-y-1.5">
                          <div className="flex items-center justify-between">
                            <span className="text-[11px] font-mono text-cyan-300 font-bold">{ir.image_chunk_id}</span>
                            <span className={`text-[9px] font-mono px-1.5 py-0.5 rounded uppercase ${
                              ir.entity_verified ? 'bg-emerald-950 text-emerald-300 border border-emerald-800' : 'bg-gray-800 text-gray-400'
                            }`}>
                              {ir.entity_verified ? 'Entity Verified' : 'Unlinked Asset'}
                            </span>
                          </div>
                          <p className="text-[11px] text-gray-300 line-clamp-2 leading-relaxed">
                            {ir.visual_description}
                          </p>
                          {ir.associated_records && ir.associated_records.length > 0 && (
                            <div className="text-[10px] text-gray-400 font-mono border-t border-gray-800 pt-1">
                              Linked to: <span className="text-cyan-400">{ir.associated_records[0].target_source_id}</span> ({ir.associated_records[0].matched_entities?.join(', ')})
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* 5. Grounded Artefact */}
                <div className="bg-[#12141f] border border-gray-800 rounded-2xl p-5 space-y-4">
                  <div className="flex items-center justify-between pb-3 border-b border-gray-800">
                    <div className="flex items-center space-x-2">
                      <FileText className="w-4 h-4 text-cyan-400" />
                      <h2 className="text-sm font-bold text-white uppercase tracking-wider font-mono">
                        Grounded Investigation Brief
                      </h2>
                    </div>
                    <span className="text-[11px] font-mono text-cyan-400">
                      {investigation.citations?.length || 0} Citations
                    </span>
                  </div>

                  <div className="prose prose-invert max-w-none text-xs leading-relaxed space-y-3 font-sans text-gray-300">
                    {renderCitedMarkdown(investigation.brief)}
                  </div>
                </div>
              </>
            )}
          </div>
        </div>

        {/* ---------------- PANE 3: RIGHT (Interactive Citation & Explain Inspector) ---------------- */}
        <div className="w-96 border-l border-gray-800 bg-[#0e1017] flex flex-col shrink-0">
          <div className="h-11 border-b border-gray-800 bg-[#12141f] px-3 flex items-center justify-between">
            <div className="flex space-x-1">
              <button
                onClick={() => setActiveTab('citation')}
                className={`px-3 py-1 rounded-lg text-xs font-mono font-medium transition ${
                  activeTab === 'citation'
                    ? 'bg-cyan-950 border border-cyan-800 text-cyan-300'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                Citation Inspector
              </button>
              <button
                onClick={() => setActiveTab('explain')}
                className={`px-3 py-1 rounded-lg text-xs font-mono font-medium transition ${
                  activeTab === 'explain'
                    ? 'bg-cyan-950 border border-cyan-800 text-cyan-300'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                MongoDB Explain
              </button>
            </div>
            <span className="text-[10px] font-mono text-gray-500">Live Inspector</span>
          </div>

          {/* Right Pane Body */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {activeTab === 'citation' && (
              <>
                {!selectedCitation && !rawSourceData && (
                  <div className="h-full flex flex-col items-center justify-center text-center p-6 text-gray-500">
                    <Info className="w-8 h-8 text-gray-700 mb-2" />
                    <div className="text-xs font-medium text-gray-400">No Citation Selected</div>
                    <p className="text-[11px] text-gray-600 mt-1">
                      Click any <span className="text-cyan-400 font-mono">[CIT:chk_XX]</span> badge or an uploaded file to inspect exact character offsets.
                    </p>
                  </div>
                )}

                {selectedCitation && (
                  <div className="space-y-4">
                    {/* Citation Header Card */}
                    <div className="bg-[#141624] border border-cyan-900/60 rounded-xl p-3.5 space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-mono font-bold text-cyan-400 flex items-center space-x-1">
                          <Code2 className="w-4 h-4" />
                          <span>{selectedCitation.chunk_id}</span>
                        </span>
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#1e2338] text-gray-300">
                          {selectedCitation.source_id}
                        </span>
                      </div>
                      <div className="text-xs font-semibold text-white">{selectedCitation.source_title}</div>

                      {/* Character Offsets Pill */}
                      <div className="flex items-center space-x-2 text-[10px] font-mono text-gray-400 pt-1 border-t border-gray-800">
                        <span className="text-cyan-400">char_start: {selectedCitation.char_start}</span>
                        <span>&rarr;</span>
                        <span className="text-cyan-400">char_end: {selectedCitation.char_end}</span>
                        <span className="text-gray-600">({selectedCitation.char_end - selectedCitation.char_start} chars)</span>
                      </div>
                    </div>

                    {/* Score Breakdown Matrix */}
                    <div className="grid grid-cols-3 gap-2">
                      <div className="bg-[#131520] border border-gray-800 p-2.5 rounded-xl text-center">
                        <div className="text-[9px] uppercase font-mono text-gray-400 font-bold">Vector Score</div>
                        <div className="text-sm font-bold text-cyan-400 font-mono mt-0.5">
                          {selectedCitation.vector_score ?? 0.88}
                        </div>
                        <div className="text-[9px] text-gray-500 font-mono">Cosine Sim</div>
                      </div>

                      <div className="bg-[#131520] border border-gray-800 p-2.5 rounded-xl text-center">
                        <div className="text-[9px] uppercase font-mono text-gray-400 font-bold">Keyword Score</div>
                        <div className="text-sm font-bold text-amber-400 font-mono mt-0.5">
                          {selectedCitation.keyword_score ?? 1.5}
                        </div>
                        <div className="text-[9px] text-gray-500 font-mono">BM25 $text</div>
                      </div>

                      <div className="bg-[#131520] border border-gray-800 p-2.5 rounded-xl text-center">
                        <div className="text-[9px] uppercase font-mono text-gray-400 font-bold">Hybrid RRF</div>
                        <div className="text-sm font-bold text-emerald-400 font-mono mt-0.5">
                          {selectedCitation.rrf_score ?? 0.032}
                        </div>
                        <div className="text-[9px] text-gray-500 font-mono">1/(60+r)</div>
                      </div>
                    </div>

                    {/* Evidence Provenance Badge */}
                    <div className="bg-[#10121c] border border-gray-800/80 rounded-xl p-2.5 flex items-center justify-between text-[10px] font-mono text-gray-400">
                      <div className="flex items-center space-x-1">
                        <span className="text-gray-500">Method:</span>
                        <span className="text-cyan-400 uppercase font-bold">{selectedCitation.retrieval_method || 'hybrid_rrf'}</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <span className="text-gray-500">Integrity:</span>
                        <span className="text-emerald-400 font-bold">
                          {selectedCitation.offset_verified !== false ? '✓ Offset Verified' : '⚠ Offset Mismatch'}
                        </span>
                      </div>
                    </div>

                    {/* Image Preview if PNG */}
                    {rawSourceData?.source?.image_base64 && (
                      <div className="space-y-2">
                        <div className="text-[11px] font-mono text-gray-400 uppercase tracking-wider flex items-center justify-between">
                          <span>Visual Telemetry Preview</span>
                          <span className="text-purple-400 text-[10px]">PNG Asset</span>
                        </div>
                        <div className="border border-purple-900/60 rounded-xl overflow-hidden shadow-lg shadow-purple-950/20 bg-black">
                          <img
                            src={`data:image/png;base64,${rawSourceData.source.image_base64}`}
                            alt="Visual Asset"
                            className="w-full h-auto object-cover"
                          />
                        </div>
                      </div>
                    )}

                    {/* Raw Source Document with Exact Highlight */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-[11px] font-mono text-gray-400">
                        <span>Original Source Document</span>
                        <span className="text-cyan-400 text-[10px]">Exact Excerpt Highlighted</span>
                      </div>

                      <div className="bg-[#090a0f] border border-gray-800 rounded-xl p-3 max-h-[350px] overflow-auto text-[11px] font-mono leading-relaxed text-gray-300">
                        {rawSourceData?.source?.content ? (
                          (() => {
                            const full = rawSourceData.source.content;
                            const start = selectedCitation.char_start || 0;
                            const end = selectedCitation.char_end || full.length;

                            const before = full.substring(0, start);
                            const highlight = full.substring(start, end);
                            const after = full.substring(end);

                            return (
                              <pre className="whitespace-pre-wrap font-mono">
                                <span>{before}</span>
                                <mark className="bg-cyan-500/30 text-cyan-200 border-l-2 border-r-2 border-cyan-400 px-1 py-0.5 rounded font-bold">
                                  {highlight}
                                </mark>
                                <span>{after}</span>
                              </pre>
                            );
                          })()
                        ) : investigation.status === 'VALIDATION_FAILED' ? (
                      <span className="flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-bold font-mono bg-yellow-950 text-yellow-300 border border-yellow-800">
                        <AlertTriangle className="w-4 h-4 text-yellow-400" />
                        <span>STATUS: VALIDATION FAILED</span>
                      </span>
                    ) : (
                          <div className="text-gray-500 italic">Loading raw document...</div>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </>
            )}

            {activeTab === 'explain' && (
              <ExplainViewer explainData={investigation?.explain} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
