import { useState } from 'react'
import { postQuery } from './api/client'
import QueryInput from './components/QueryInput'
import ResponseCard from './components/ResponseCard'
import TracePanel from './components/TracePanel'
import MetricsDashboard from './components/MetricsDashboard'

export default function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleSubmit = async (question) => {
    setLoading(true)
    setError(null)
    try {
      const data = await postQuery(question)
      setResult(data)
    } catch (err) {
      setError(err.response?.data?.detail ?? err.message ?? 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen" style={{ backgroundColor: '#0F0F13' }}>
      {/* Header */}
      <header
        className="sticky top-0 z-10 px-6 py-4 flex items-center gap-4 border-b"
        style={{ backgroundColor: 'rgba(26,26,36,0.85)', borderColor: '#2A2A38', backdropFilter: 'blur(12px)' }}
      >
        <div className="flex items-center gap-3">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center text-base"
            style={{ background: 'linear-gradient(135deg, #6366F1, #8B5CF6)' }}
          >
            ⚡
          </div>
          <div>
            <h1 className="text-sm font-bold text-white leading-none">Agentic RAG</h1>
            <p className="text-xs mt-0.5" style={{ color: '#9CA3AF' }}>
              Query planning · Self-correction · Three retrieval strategies
            </p>
          </div>
        </div>
        <div className="ml-auto flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-emerald-400" style={{ boxShadow: '0 0 6px #10B981' }} />
          <span className="text-xs" style={{ color: '#9CA3AF' }}>API connected</span>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-6">
        {/* Two-panel layout */}
        <div className="flex gap-5 items-start">
          {/* Left — 60% */}
          <div className="flex-[3] min-w-0 space-y-4">
            <QueryInput onSubmit={handleSubmit} loading={loading} />

            {error && (
              <div
                className="rounded-xl px-4 py-3 text-sm border"
                style={{ backgroundColor: 'rgba(239,68,68,0.1)', borderColor: 'rgba(239,68,68,0.3)', color: '#FCA5A5' }}
              >
                ⚠ {error}
              </div>
            )}

            <ResponseCard result={result} loading={loading} />
          </div>

          {/* Right — 40% */}
          <div className="flex-[2] min-w-0 sticky top-24">
            <TracePanel result={result} />
          </div>
        </div>

        {/* Metrics — full width */}
        <div className="mt-5">
          <MetricsDashboard />
        </div>
      </div>
    </div>
  )
}
