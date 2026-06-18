import { useState } from 'react'
import { postFeedback } from '../api/client'

const STRATEGY_CONFIG = {
  SEMANTIC:   { color: '#6366F1', bg: 'rgba(99,102,241,0.12)',  border: 'rgba(99,102,241,0.3)',  icon: '🔍', label: 'Semantic'   },
  KEYWORD:    { color: '#10B981', bg: 'rgba(16,185,129,0.12)',  border: 'rgba(16,185,129,0.3)',  icon: '🏷️', label: 'Keyword'    },
  STRUCTURED: { color: '#F59E0B', bg: 'rgba(245,158,11,0.12)',  border: 'rgba(245,158,11,0.3)',  icon: '🗄️', label: 'Structured' },
}

const SOURCE_CONFIG = {
  medquad:   { color: '#6366F1', bg: 'rgba(99,102,241,0.12)',  border: 'rgba(99,102,241,0.25)'  },
  medlineplus: { color: '#A855F7', bg: 'rgba(168,85,247,0.12)',  border: 'rgba(168,85,247,0.25)'  },
  sql:       { color: '#F59E0B', bg: 'rgba(245,158,11,0.12)',   border: 'rgba(245,158,11,0.25)'  },
}

function ConfidenceRing({ confidence, color }) {
  const r = 20
  const circumference = 2 * Math.PI * r
  const offset = circumference * (1 - Math.min(1, Math.max(0, confidence)))
  const pct = Math.round(confidence * 100)
  return (
    <div className="relative" style={{ width: 52, height: 52 }}>
      <svg width="52" height="52" viewBox="0 0 52 52">
        <circle cx="26" cy="26" r={r} fill="none" stroke="#2A2A38" strokeWidth="4" />
        <circle
          cx="26" cy="26" r={r} fill="none"
          stroke={color} strokeWidth="4" strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform="rotate(-90 26 26)"
          style={{ transition: 'stroke-dashoffset 0.7s ease' }}
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-xs font-bold text-white">{pct}%</span>
      </div>
    </div>
  )
}

export default function ResponseCard({ result, loading }) {
  const [feedback, setFeedback] = useState(null)

  if (!result && !loading) {
    return (
      <div
        className="rounded-2xl border border-dashed p-10 text-center"
        style={{ borderColor: '#2A2A38', backgroundColor: 'rgba(26,26,36,0.5)' }}
      >
        <div className="text-3xl mb-3">🤖</div>
        <p className="text-sm" style={{ color: '#6B7280' }}>Your answer will appear here</p>
        <p className="text-xs mt-1" style={{ color: '#4B5563' }}>Try one of the example queries above</p>
      </div>
    )
  }

  if (loading) {
    return (
      <div
        className="rounded-2xl border p-6 space-y-3"
        style={{ backgroundColor: '#1A1A24', borderColor: '#2A2A38' }}
      >
        <div className="h-3 rounded-full animate-pulse w-3/4" style={{ backgroundColor: '#2A2A38' }} />
        <div className="h-3 rounded-full animate-pulse w-full"  style={{ backgroundColor: '#2A2A38' }} />
        <div className="h-3 rounded-full animate-pulse w-5/6"   style={{ backgroundColor: '#2A2A38' }} />
        <div className="h-3 rounded-full animate-pulse w-2/3"   style={{ backgroundColor: '#2A2A38' }} />
      </div>
    )
  }

  const { answer, strategy, confidence, retry_count, reranked_docs = [] } = result
  const cfg = STRATEGY_CONFIG[strategy] ?? { color: '#9CA3AF', bg: 'rgba(156,163,175,0.1)', border: 'rgba(156,163,175,0.2)', icon: '❓', label: strategy }

  const handleFeedback = async (thumbs_up) => {
    setFeedback(thumbs_up)
    await postFeedback(crypto.randomUUID(), thumbs_up).catch(() => {})
  }

  return (
    <div
      className="rounded-2xl border overflow-hidden"
      style={{ backgroundColor: '#1A1A24', borderColor: '#2A2A38', borderLeftWidth: 3, borderLeftColor: cfg.color }}
    >
      {/* Header */}
      <div className="px-5 pt-5 pb-4 flex items-center gap-3 flex-wrap" style={{ borderBottom: '1px solid #2A2A38' }}>
        <span
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold border"
          style={{ backgroundColor: cfg.bg, borderColor: cfg.border, color: cfg.color }}
        >
          <span>{cfg.icon}</span>
          <span>{cfg.label}</span>
        </span>

        {retry_count > 0 && (
          <span
            className="flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium border"
            style={{ backgroundColor: 'rgba(251,146,60,0.1)', borderColor: 'rgba(251,146,60,0.3)', color: '#FB923C' }}
          >
            ↺ Re-retrieved (retry {retry_count})
          </span>
        )}

        <div className="ml-auto flex items-center gap-3">
          <span className="text-xs" style={{ color: '#6B7280' }}>Confidence</span>
          <ConfidenceRing confidence={confidence} color={cfg.color} />
        </div>
      </div>

      {/* Answer */}
      <div className="px-5 py-4">
        <p className="text-sm leading-7 whitespace-pre-wrap" style={{ color: '#F3F4F6' }}>{answer}</p>
      </div>

      {/* Source docs */}
      {reranked_docs.length > 0 && (
        <div className="px-5 pb-4 space-y-2">
          <p className="text-[10px] font-semibold uppercase tracking-widest mb-2" style={{ color: '#6B7280' }}>
            Sources
          </p>
          {reranked_docs.map((doc, i) => {
            const sc = SOURCE_CONFIG[doc.source] ?? { color: '#9CA3AF', bg: 'rgba(156,163,175,0.1)', border: 'rgba(156,163,175,0.2)' }
            const scorePct = Math.round((doc.rerank_score ?? 0) * 100)
            return (
              <div
                key={doc.chunk_id ?? i}
                className="rounded-xl p-3 border space-y-1.5"
                style={{ backgroundColor: 'rgba(15,15,19,0.6)', borderColor: '#2A2A38' }}
              >
                <div className="flex items-center gap-2">
                  <span
                    className="px-2 py-0.5 rounded-full text-[10px] font-semibold border"
                    style={{ backgroundColor: sc.bg, borderColor: sc.border, color: sc.color }}
                  >
                    {doc.source}
                  </span>
                  <div className="ml-auto flex items-center gap-1.5">
                    <div className="w-16 h-1 rounded-full" style={{ backgroundColor: '#2A2A38' }}>
                      <div className="h-1 rounded-full" style={{ width: `${scorePct}%`, backgroundColor: sc.color, transition: 'width 0.5s ease' }} />
                    </div>
                    <span className="text-[10px] font-medium" style={{ color: sc.color }}>{scorePct}%</span>
                  </div>
                </div>
                <p className="text-xs leading-5 line-clamp-2" style={{ color: '#9CA3AF' }}>{doc.text}</p>
              </div>
            )
          })}
        </div>
      )}

      {/* Feedback */}
      <div className="px-5 py-3 flex items-center gap-3" style={{ borderTop: '1px solid #2A2A38' }}>
        <span className="text-xs" style={{ color: '#6B7280' }}>Helpful?</span>
        {[true, false].map((val) => (
          <button
            key={String(val)}
            onClick={() => handleFeedback(val)}
            disabled={feedback !== null}
            className="text-base transition-all duration-200 disabled:cursor-default"
            style={{ opacity: feedback === null ? 0.45 : feedback === val ? 1 : 0.2 }}
            onMouseEnter={(e) => feedback === null && (e.target.style.opacity = '1')}
            onMouseLeave={(e) => feedback === null && (e.target.style.opacity = '0.45')}
            aria-label={val ? 'Thumbs up' : 'Thumbs down'}
          >
            {val ? '👍' : '👎'}
          </button>
        ))}
        {feedback !== null && (
          <span className="text-xs ml-1" style={{ color: '#10B981' }}>Thanks!</span>
        )}
      </div>
    </div>
  )
}
