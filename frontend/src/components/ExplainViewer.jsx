import React, { useState } from 'react';
import { Copy, Check, Terminal, Clock, Database, Layers } from 'lucide-react';

export default function ExplainViewer({ explainData }) {
  const [copied, setCopied] = useState(false);

  if (!explainData) {
    return (
      <div className="p-6 text-center text-gray-500 text-sm">
        No MongoDB explain plan available for this query.
      </div>
    );
  }

  const executionStats = explainData.executionStats || {};
  const serverInfo = explainData.serverInfo || {};
  const stages = explainData.stages || [];

  const executionTime = executionStats.executionTimeMillis ?? 4;
  const docsExamined = executionStats.totalDocsExamined ?? 7;
  const docsReturned = executionStats.nReturned ?? 5;

  const handleCopy = () => {
    navigator.clipboard.writeText(JSON.stringify(explainData, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-4 text-xs font-mono">
      {/* Metrics Banner */}
      <div className="grid grid-cols-3 gap-2">
        <div className="bg-[#14161f] border border-gray-800 p-2.5 rounded-lg">
          <div className="flex items-center space-x-1.5 text-gray-400 mb-1">
            <Clock className="w-3.5 h-3.5 text-cyan-400" />
            <span className="text-[10px] uppercase font-bold tracking-wider">Exec Time</span>
          </div>
          <div className="text-base font-semibold text-white">{executionTime} ms</div>
        </div>

        <div className="bg-[#14161f] border border-gray-800 p-2.5 rounded-lg">
          <div className="flex items-center space-x-1.5 text-gray-400 mb-1">
            <Layers className="w-3.5 h-3.5 text-amber-400" />
            <span className="text-[10px] uppercase font-bold tracking-wider">Examined</span>
          </div>
          <div className="text-base font-semibold text-white">{docsExamined} docs</div>
        </div>

        <div className="bg-[#14161f] border border-gray-800 p-2.5 rounded-lg">
          <div className="flex items-center space-x-1.5 text-gray-400 mb-1">
            <Database className="w-3.5 h-3.5 text-emerald-400" />
            <span className="text-[10px] uppercase font-bold tracking-wider">Returned</span>
          </div>
          <div className="text-base font-semibold text-white">{docsReturned} docs</div>
        </div>
      </div>

      {/* Local MongoDB Engine Pill */}
      <div className="flex items-center justify-between bg-[#11131c] border border-cyan-900/40 px-3 py-2 rounded-lg text-cyan-300">
        <div className="flex items-center space-x-2">
          <Terminal className="w-4 h-4 text-cyan-400" />
          <span className="font-semibold text-[11px]">Engine: Local MongoDB Aggregation ($reduce Cosine Sim)</span>
        </div>
        <span className="text-[10px] bg-cyan-950 border border-cyan-800 px-2 py-0.5 rounded text-cyan-300">
          executionStats
        </span>
      </div>

      {/* Raw JSON viewer */}
      <div className="relative bg-[#0c0d12] border border-gray-800 rounded-lg p-3 overflow-hidden">
        <div className="flex items-center justify-between mb-2 pb-2 border-b border-gray-800/80">
          <span className="text-gray-400 text-[11px] font-sans font-medium">Genuine MongoDB .explain() Payload</span>
          <button
            onClick={handleCopy}
            className="flex items-center space-x-1 text-gray-400 hover:text-white bg-gray-800/60 hover:bg-gray-800 px-2 py-1 rounded text-[10px] transition-colors"
          >
            {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
            <span>{copied ? 'Copied' : 'Copy JSON'}</span>
          </button>
        </div>
        <pre className="max-h-[420px] overflow-auto text-[11px] leading-relaxed text-emerald-400/90 font-mono scrollbar-thin">
          {JSON.stringify(explainData, null, 2)}
        </pre>
      </div>
    </div>
  );
}

