import { useEffect, useState } from 'react'
import { getMetrics } from '../api/client'

const METRIC_CONFIG = [
  { key: 'faithfulness',      label: 'Faithfulness',       color: '#6366F1', glow: 'rgba(99,102,241,0.3)'  },
  { key: 'answer_relevancy',  label: 'Answer Relevancy',   color: '#10B981', glow: 'rgba(16,185,129,0.3)'  },
  { key: 'context_recall',    label: 'Context Recall',     color: '#F59E0B', glow: 'rgba(245,158,11,0.3)'  },
  { key: 'context_precision', label: 'Context Precision',  color: '#F43F5E', glow: 'rgba(244,63,94,0.3)'   },
]

function CircleArc({ score, color, glow }) {
  const size = 80
  const cx = size / 2
  const cy = size / 2
  const r = 30
  const circumference = 2 * Math.PI * r
  const offset = circumference * (1 - Math.min(1, Math.max(0, score)))
  const pct = Math.round(score * 100)

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="#2A2A38" strokeWidth="5" />
        <circle
          cx={cx} cy={cy} r={r} fill="none"
          stroke={color} strokeWidth="5" strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform={`rotate(-90 ${cx} ${cy})`}
          style={{
            transition: 'stroke-dashoffset 0.8s cubic-bezier(0.4,0,0.2,1)',
            filter: `drop-shadow(0 0 6px ${glow})`,
          }}
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-sm font-bold text-white">{pct}</span>
      </div>
    </div>
  )
}

function MetricCard({ label, score, color, glow }) {
  return (
    <div
      className="rounded-xl p-4 border flex flex-col items-center gap-3 transition-all duration-200"
      style={{
        backgroundColor: '#1A1A24',
        borderColor: '#2A2A38',
      }}
      onMouseEnter={(e) => { e.currentTarget.style.borderColor = color; e.currentTarget.style.boxShadow = `0 0 20px ${glow}` }}
      onMouseLeave={(e) => { e.currentTarget.style.borderColor = '#2A2A38'; e.currentTarget.style.boxShadow = 'none' }}
    >
      <CircleArc score={score} color={color} glow={glow} />
      <div className="text-center">
        <p className="text-xs font-medium text-white leading-tight">{label}</p>
        <p className="text-[10px] mt-0.5" style={{ color }}>
          {score >= 0.7 ? 'Strong' : score >= 0.4 ? 'Moderate' : 'Weak'}
        </p>
      </div>
    </div>
  )
}

export default function MetricsDashboard() {
  const [metrics, setMetrics] = useState(null)

  useEffect(() => {
    getMetrics().then(setMetrics).catch(() => setMetrics({}))
  }, [])

  if (!metrics) return null

  const isEmpty = Object.keys(metrics).length === 0

  return (
    <div
      className="rounded-2xl border p-5"
      style={{ backgroundColor: '#1A1A24', borderColor: '#2A2A38' }}
    >
      <div className="flex items-center gap-2 mb-4">
        <span className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: '#6B7280' }}>
          RAGAS Evaluation Metrics
        </span>
        {!isEmpty && (
          <span
            className="px-2 py-0.5 rounded-full text-[10px] font-medium"
            style={{ backgroundColor: 'rgba(16,185,129,0.12)', color: '#10B981', border: '1px solid rgba(16,185,129,0.25)' }}
          >
            Latest run
          </span>
        )}
      </div>

      {isEmpty ? (
        <div className="text-center py-8 space-y-2">
          <div className="text-2xl">📊</div>
          <p className="text-sm" style={{ color: '#6B7280' }}>Run RAGAS evaluation to see metrics</p>
          <p className="text-xs" style={{ color: '#4B5563' }}>
            Results will be read from <code className="px-1 rounded text-indigo-400" style={{ backgroundColor: '#0F0F13' }}>data/ragas_results.json</code>
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {METRIC_CONFIG.map(({ key, label, color, glow }) => (
            <MetricCard
              key={key}
              label={label}
              score={metrics[key] ?? 0}
              color={color}
              glow={glow}
            />
          ))}
        </div>
      )}
    </div>
  )
}
