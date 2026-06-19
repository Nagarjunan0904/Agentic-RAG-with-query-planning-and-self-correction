import { useState } from 'react'

const EXAMPLES = [
  { label: 'What are the symptoms of type 2 diabetes?',         type: 'SEMANTIC'    },
  { label: "How is Alzheimer's disease treated?",               type: 'SEMANTIC'    },
  { label: 'What is hypertension?',                             type: 'KEYWORD'     },
  { label: 'Find information on asthma',                        type: 'KEYWORD'     },
  { label: 'What is the most common condition among patients?', type: 'STRUCTURED'  },
  { label: 'What is the most prescribed medication?',           type: 'STRUCTURED'  },
]

const GROUPS = [
  { type: 'SEMANTIC',   label: 'Semantic',   color: '#6366F1', bg: 'rgba(99,102,241,0.12)',  border: 'rgba(99,102,241,0.35)' },
  { type: 'KEYWORD',    label: 'Keyword',    color: '#10B981', bg: 'rgba(16,185,129,0.12)',  border: 'rgba(16,185,129,0.35)' },
  { type: 'STRUCTURED', label: 'Structured', color: '#F59E0B', bg: 'rgba(245,158,11,0.12)',  border: 'rgba(245,158,11,0.35)' },
]

function Dots() {
  return (
    <span className="flex items-center gap-1">
      {[0, 150, 300].map((delay) => (
        <span
          key={delay}
          className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-bounce"
          style={{ animationDelay: `${delay}ms` }}
        />
      ))}
    </span>
  )
}

export default function QueryInput({ onSubmit, loading }) {
  const [value, setValue] = useState('')
  const [focused, setFocused] = useState(false)

  const submit = (text) => {
    const q = (text ?? value).trim()
    if (!q || loading) return
    onSubmit(q)
  }

  const handlePill = (label) => {
    setValue(label)
    onSubmit(label)
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div
      className="rounded-2xl p-5 space-y-4 border"
      style={{ backgroundColor: '#1A1A24', borderColor: '#2A2A38' }}
    >
      {/* Grouped pills */}
      <div className="space-y-2">
        {GROUPS.map(({ type, label, color, bg, border }) => (
          <div key={type} className="flex items-center gap-2 flex-wrap">
            <span
              className="text-[10px] font-semibold uppercase tracking-widest w-20 shrink-0"
              style={{ color }}
            >
              {label}
            </span>
            {EXAMPLES.filter((e) => e.type === type).map((ex) => (
              <button
                key={ex.label}
                onClick={() => handlePill(ex.label)}
                disabled={loading}
                className="px-3 py-1 rounded-full text-xs font-medium border transition-all cursor-pointer disabled:opacity-40 hover:brightness-125"
                style={{ backgroundColor: bg, borderColor: border, color }}
              >
                {ex.label}
              </button>
            ))}
          </div>
        ))}
      </div>

      {/* Calibration note */}
      <p className="text-[11px] text-center" style={{ color: '#4B5563' }}>
        Grounded in NIH medical literature (MedQuAD + MedlinePlus) and synthetic patient records (Synthea)
      </p>

      {/* Textarea + button */}
      <div className="flex gap-3">
        <div
          className="flex-1 rounded-xl border transition-all duration-200"
          style={{
            borderColor: focused ? '#6366F1' : '#2A2A38',
            boxShadow: focused ? '0 0 0 3px rgba(99,102,241,0.2), 0 0 20px rgba(99,102,241,0.1)' : 'none',
          }}
        >
          <textarea
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKey}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            disabled={loading}
            rows={3}
            placeholder="Ask anything… or click an example above"
            className="w-full resize-none rounded-xl px-4 py-3 text-sm leading-relaxed placeholder-[#4B5563] disabled:opacity-50"
            style={{ backgroundColor: '#0F0F13', color: '#F9FAFB', border: 'none' }}
          />
        </div>

        <button
          onClick={() => submit()}
          disabled={loading || !value.trim()}
          className="self-end px-5 py-3 rounded-xl text-sm font-semibold text-white transition-all duration-200 flex items-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed"
          style={{
            background: 'linear-gradient(135deg, #6366F1, #8B5CF6)',
            boxShadow: (!loading && value.trim()) ? '0 0 20px rgba(99,102,241,0.4)' : 'none',
          }}
        >
          {loading ? (
            <>
              <Dots />
              <span>Thinking</span>
            </>
          ) : (
            <>
              <span>Ask</span>
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12h15m0 0l-6.75-6.75M19.5 12l-6.75 6.75" />
              </svg>
            </>
          )}
        </button>
      </div>
    </div>
  )
}
