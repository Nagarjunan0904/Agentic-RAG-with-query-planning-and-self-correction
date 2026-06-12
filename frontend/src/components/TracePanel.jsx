import { useState } from 'react'

const NODE_DEFS = [
  { key: 'planner',          label: 'Planner',    icon: '🧠' },
  { key: 'vector_retriever', label: 'Vector',     icon: '🔎' },
  { key: 'bm25_retriever',   label: 'BM25',       icon: '📝' },
  { key: 'sql_agent',        label: 'SQL',        icon: '🗄️' },
  { key: 'reranker',         label: 'Reranker',   icon: '⚡' },
  { key: 'generator',        label: 'Generator',  icon: '✨' },
  { key: 'evaluator',        label: 'Evaluator',  icon: '🎯' },
]

const SOURCE_CONFIG = {
  msmarco:   { color: '#6366F1', bg: 'rgba(99,102,241,0.12)',  border: 'rgba(99,102,241,0.25)'  },
  wikipedia: { color: '#A855F7', bg: 'rgba(168,85,247,0.12)',  border: 'rgba(168,85,247,0.25)'  },
  sql:       { color: '#F59E0B', bg: 'rgba(245,158,11,0.12)',   border: 'rgba(245,158,11,0.25)'  },
}

export default function TracePanel({ result }) {
  const [open, setOpen] = useState(true)

  if (!result) {
    return (
      <div
        className="rounded-2xl border p-5"
        style={{ backgroundColor: '#1A1A24', borderColor: '#2A2A38' }}
      >
        <p className="text-xs font-semibold uppercase tracking-widest mb-3" style={{ color: '#4B5563' }}>
          Execution Trace
        </p>
        <p className="text-xs text-center py-6" style={{ color: '#4B5563' }}>
          Run a query to see the trace
        </p>
      </div>
    )
  }

  const { latency_ms = {}, reranked_docs = [] } = result
  const presentNodes = NODE_DEFS.filter((n) => latency_ms[n.key] != null)
  const maxLatency = Math.max(...presentNodes.map((n) => latency_ms[n.key]), 1)
  const totalMs = presentNodes.reduce((s, n) => s + latency_ms[n.key], 0)

  return (
    <div
      className="rounded-2xl border overflow-hidden"
      style={{ backgroundColor: '#1A1A24', borderColor: '#2A2A38' }}
    >
      {/* Header */}
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between px-5 py-4 transition-colors hover:brightness-110"
        style={{ borderBottom: open ? '1px solid #2A2A38' : 'none' }}
      >
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold uppercase tracking-widest" style={{ color: '#9CA3AF' }}>
            Execution Trace
          </span>
          <span
            className="px-2 py-0.5 rounded-full text-[10px] font-medium"
            style={{ backgroundColor: 'rgba(99,102,241,0.15)', color: '#818CF8' }}
          >
            {totalMs} ms total
          </span>
        </div>
        <svg
          className="w-4 h-4 transition-transform duration-300"
          style={{ color: '#6B7280', transform: open ? 'rotate(180deg)' : 'rotate(0deg)' }}
          fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Collapsible body */}
      <div style={{ maxHeight: open ? '1000px' : '0', overflow: 'hidden', transition: 'max-height 0.35s ease' }}>
        <div className="p-5 space-y-5">
          {/* Vertical timeline */}
          <div className="space-y-0">
            {presentNodes.map((node, idx) => {
              const ms = latency_ms[node.key]
              const pct = (ms / maxLatency) * 100
              const isLast = idx === presentNodes.length - 1
              return (
                <div key={node.key} className="flex gap-3">
                  {/* Timeline spine */}
                  <div className="flex flex-col items-center">
                    <div
                      className="w-8 h-8 rounded-lg flex items-center justify-center text-sm shrink-0 z-10"
                      style={{ backgroundColor: '#0F0F13', border: '1px solid #3A3A4A' }}
                    >
                      {node.icon}
                    </div>
                    {!isLast && (
                      <div className="w-px flex-1 my-1" style={{ backgroundColor: '#2A2A38', minHeight: 16 }} />
                    )}
                  </div>

                  {/* Node content */}
                  <div className={`flex-1 min-w-0 ${isLast ? 'pb-0' : 'pb-3'}`}>
                    <div className="flex items-center gap-2 h-8">
                      <span className="text-xs font-medium text-white">{node.label}</span>
                      <span
                        className="px-1.5 py-0.5 rounded text-[10px] font-mono"
                        style={{ backgroundColor: '#0F0F13', color: '#818CF8', border: '1px solid #2A2A38' }}
                      >
                        {ms}ms
                      </span>
                    </div>
                    <div className="mt-1 h-1.5 rounded-full" style={{ backgroundColor: '#0F0F13' }}>
                      <div
                        className="h-1.5 rounded-full"
                        style={{
                          width: `${pct}%`,
                          background: 'linear-gradient(90deg, #6366F1, #8B5CF6)',
                          transition: 'width 0.6s ease',
                        }}
                      />
                    </div>
                  </div>
                </div>
              )
            })}
          </div>

          {/* Retrieved docs */}
          {reranked_docs.length > 0 && (
            <div>
              <p className="text-[10px] font-semibold uppercase tracking-widest mb-3" style={{ color: '#6B7280' }}>
                Top Docs
              </p>
              <div className="space-y-2">
                {reranked_docs.map((doc, i) => {
                  const sc = SOURCE_CONFIG[doc.source] ?? { color: '#9CA3AF', bg: 'rgba(156,163,175,0.1)', border: 'rgba(156,163,175,0.2)' }
                  const scorePct = Math.round((doc.rerank_score ?? 0) * 100)
                  return (
                    <div
                      key={doc.chunk_id ?? i}
                      className="rounded-lg p-3 border"
                      style={{ backgroundColor: '#0F0F13', borderColor: '#2A2A38' }}
                    >
                      <div className="flex items-center gap-2 mb-1.5">
                        <span
                          className="px-1.5 py-0.5 rounded text-[10px] font-semibold border"
                          style={{ backgroundColor: sc.bg, borderColor: sc.border, color: sc.color }}
                        >
                          {doc.source}
                        </span>
                        <span className="ml-auto text-[10px] font-medium" style={{ color: sc.color }}>
                          {scorePct}%
                        </span>
                      </div>
                      <p className="text-[11px] leading-4 line-clamp-2" style={{ color: '#9CA3AF' }}>
                        {doc.text}
                      </p>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
