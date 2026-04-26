import { useState, useRef, useEffect } from 'react'
import { useLocation } from 'react-router-dom'
import { MessageCircle, X, Send, Bot, User, Trash2, Sparkles } from 'lucide-react'

interface Message {
  role: 'user' | 'assistant'
  text: string
}

interface PageContext {
  page: string
  dataset: string
  engine_id?: number
  engine_score?: number
  engine_decision?: string
  engine_rul?: number
  engine_regime?: number
  active_tab?: string
  top_sensor?: string
  fleet_total?: number
  fleet_alerted?: number
}

// ── Module-level singletons ─────────────────────────────────────────────────

let _ctx: PageContext = { page: 'overview', dataset: 'FD001' }
export function setChatContext(ctx: Partial<PageContext>) {
  const next = { ..._ctx, ...ctx }
  const changed = (Object.keys(ctx) as (keyof PageContext)[]).some(k => _ctx[k] !== next[k])
  _ctx = next
  if (changed) window.dispatchEvent(new Event('chat-context-updated'))
}

// openChatWith: called by chart "Explain" buttons to open + auto-ask a question
let _trigger: ((question: string) => void) | null = null
export function openChatWith(question: string) {
  _trigger?.(question)
}

// ── Suggestion chips — per-page and per-tab ─────────────────────────────────

const TAB_SUGGESTIONS: Record<string, string[]> = {
  'Sensor Readings': [
    'Explain the Sensor Readings chart',
    'What does the gap between the two lines mean?',
    'Why did the alert fire at that cycle?',
  ],
  'Wear Signals': [
    'Explain the Wear Signals chart',
    'What do the ±3σ bands mean?',
    'Is this deviation level dangerous?',
  ],
  'Flight Conditions': [
    'Explain the Flight Conditions chart',
    'What do the different colours represent?',
    'How does operating condition affect health scoring?',
  ],
  'Sensor Integrity': [
    'Explain the Sensor Integrity chart',
    'When is the physics veto triggered?',
    'What does G > 26.30 mean in practice?',
  ],
  'Detection Log': [
    'Walk me through this detection log',
    'What does the causal reasoner step do?',
    'How is the final decision made?',
  ],
}

const PAGE_SUGGESTIONS: Record<string, string[]> = {
  overview: [
    'Explain the Fleet Status chart',
    'Why are some engines flagged red?',
    'What does RUL mean?',
  ],
  'engine-detail': [
    'Why is this engine flagged?',
    'What do these sensor readings mean?',
    'How many cycles until failure?',
  ],
  alerts: [
    'What does the risk score mean?',
    'When should I ground an engine?',
    'What is the causal score?',
  ],
  methodology: [
    'How does causal scoring work?',
    'What is the G-test?',
    'Why use α = 1.0 for FD002/FD004?',
  ],
}

function getSuggestions(pageKey: string): string[] {
  if (pageKey === 'engine-detail' && _ctx.active_tab) {
    return TAB_SUGGESTIONS[_ctx.active_tab] ?? PAGE_SUGGESTIONS['engine-detail']
  }
  return PAGE_SUGGESTIONS[pageKey] ?? PAGE_SUGGESTIONS['overview']
}

// ── Context label shown in the header ───────────────────────────────────────

function contextLabel(): string {
  const parts: string[] = []
  if (_ctx.engine_id !== undefined) parts.push(`Engine #${_ctx.engine_id}`)
  if (_ctx.active_tab) parts.push(_ctx.active_tab)
  else if (_ctx.fleet_total !== undefined) parts.push(`${_ctx.fleet_total} engines`)
  if (_ctx.dataset) parts.push(_ctx.dataset)
  return parts.join(' · ')
}

// ── API call ─────────────────────────────────────────────────────────────────

async function callChat(message: string, context: PageContext): Promise<string> {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, context }),
  })
  if (!res.ok) return res.status === 503
    ? 'AI assistant is temporarily unavailable — please try again in a moment.'
    : 'Something went wrong — please try again.'
  const data = await res.json()
  return data.reply ?? 'No response.'
}

// ── Component ────────────────────────────────────────────────────────────────

export function ChatWidget() {
  const [open, setOpen] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [, forceUpdate] = useState(0)
  const [pendingQuestion, setPendingQuestion] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  // Always-current ref so effects can call the latest sendMessage without stale closure
  const sendRef = useRef<(text: string) => void>(() => {})
  const location = useLocation()

  const pageKey = location.pathname.startsWith('/engines/')
    ? 'engine-detail'
    : location.pathname === '/alerts'
    ? 'alerts'
    : location.pathname === '/methodology'
    ? 'methodology'
    : location.pathname === '/about'
    ? 'about'
    : 'overview'

  // Re-render when page context changes (tab switch, engine change, etc.)
  useEffect(() => {
    const handler = () => forceUpdate(n => n + 1)
    window.addEventListener('chat-context-updated', handler)
    return () => window.removeEventListener('chat-context-updated', handler)
  }, [])

  // Keep sendRef current so effects never hold a stale sendMessage closure
  useEffect(() => { sendRef.current = sendMessage })

  // Register the global openChatWith trigger — works whether panel is open or not
  useEffect(() => {
    _trigger = (question: string) => {
      setOpen(true)
      setPendingQuestion(question)
    }
    return () => { _trigger = null }
  }, [])

  // Fire whenever a new question is pending (open state is irrelevant)
  // NOTE: no setTimeout — a cleanup return would cancel the timer when setPendingQuestion('')
  // causes a re-render, silently dropping the question. Call sendRef directly instead.
  useEffect(() => {
    if (!pendingQuestion) return
    const q = pendingQuestion
    setPendingQuestion('')
    sendRef.current(q)
  }, [pendingQuestion])

  // Focus input when panel opens manually (no pending question)
  useEffect(() => {
    if (open && !pendingQuestion) {
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }, [open])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  async function sendMessage(text: string) {
    if (!text.trim() || loading) return
    setMessages(prev => [...prev, { role: 'user', text: text.trim() }])
    setInput('')
    setLoading(true)
    const ctx: PageContext = { ..._ctx, page: pageKey }
    const reply = await callChat(text.trim(), ctx)
    setMessages(prev => [...prev, { role: 'assistant', text: reply }])
    setLoading(false)
  }

  const suggestions = getSuggestions(pageKey)
  const ctxLabel = contextLabel()

  return (
    <>
      {/* Floating button — anchored just past the sidebar (w-56 = 224px) so it never covers charts */}
      <button
        onClick={() => setOpen(o => !o)}
        className="fixed bottom-5 right-5 z-50 w-12 h-12 rounded-full shadow-lg flex items-center justify-center transition-all hover:scale-105 hover:shadow-xl"
        style={{ background: 'var(--accent)', color: '#fff' }}
        aria-label="Open AI assistant"
      >
        {open ? <X size={20} /> : <MessageCircle size={20} />}
      </button>

      {/* Chat panel — same left anchor, extends rightward into left portion of content */}
      {open && (
        <div
          className="fixed bottom-20 right-5 z-50 w-[340px] rounded-2xl shadow-2xl flex flex-col overflow-hidden"
          style={{ height: 460, background: 'var(--bg-card)', border: '1px solid var(--border)' }}
        >
          {/* Header */}
          <div className="px-4 py-3 flex items-center gap-2" style={{ background: 'var(--accent)' }}>
            <Bot size={16} color="#fff" />
            <span className="text-sm font-semibold text-white">AI Assistant</span>
            {messages.length > 0 && (
              <button
                onClick={() => setMessages([])}
                className="ml-auto p-1 rounded hover:bg-white/20 transition-colors"
                title="Clear chat"
              >
                <Trash2 size={13} color="#fff" />
              </button>
            )}
          </div>

          {/* Context strip */}
          {ctxLabel && (
            <div
              className="px-4 py-1.5 text-[10px] flex items-center gap-1.5 border-b"
              style={{ background: 'var(--accent-light)', borderColor: 'var(--border)', color: 'var(--text-muted)' }}
            >
              <Sparkles size={9} style={{ color: 'var(--accent)' }} />
              <span>I can see: <span className="font-medium" style={{ color: 'var(--accent)' }}>{ctxLabel}</span></span>
            </div>
          )}

          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-3 py-3 space-y-3">
            {messages.length === 0 && (
              <div className="space-y-2">
                <p className="text-xs text-center pt-1 pb-2" style={{ color: 'var(--text-muted)' }}>
                  Ask anything, or tap a suggestion:
                </p>
                <div className="flex flex-col gap-1.5">
                  {suggestions.map(s => (
                    <button
                      key={s}
                      onClick={() => sendMessage(s)}
                      className="w-full text-left text-xs px-3 py-2.5 rounded-xl border transition-all hover:border-[var(--accent)] hover:bg-[var(--accent-light)]"
                      style={{ borderColor: 'var(--border)', color: 'var(--text-primary)' }}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((m, i) => (
              <div key={i} className={`flex gap-2 ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                {m.role === 'assistant' && (
                  <div className="w-6 h-6 rounded-full flex items-center justify-center shrink-0 mt-0.5"
                    style={{ background: 'var(--accent)', color: '#fff' }}>
                    <Bot size={11} />
                  </div>
                )}
                <div
                  className="max-w-[82%] text-xs px-3 py-2.5 leading-relaxed"
                  style={m.role === 'user'
                    ? { background: 'var(--accent)', color: '#fff', borderRadius: '16px 16px 4px 16px' }
                    : { background: 'var(--bg)', color: 'var(--text-primary)', border: '1px solid var(--border)', borderRadius: '16px 16px 16px 4px' }
                  }
                >
                  {m.text}
                </div>
                {m.role === 'user' && (
                  <div className="w-6 h-6 rounded-full flex items-center justify-center shrink-0 mt-0.5"
                    style={{ background: 'var(--border)', color: 'var(--text-muted)' }}>
                    <User size={11} />
                  </div>
                )}
              </div>
            ))}

            {loading && (
              <div className="flex gap-2 justify-start">
                <div className="w-6 h-6 rounded-full flex items-center justify-center shrink-0"
                  style={{ background: 'var(--accent)', color: '#fff' }}>
                  <Bot size={11} />
                </div>
                <div className="text-xs px-3 py-2.5" style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: '16px 16px 16px 4px', color: 'var(--text-muted)' }}>
                  <span className="inline-flex gap-1 items-center">
                    <span className="w-1.5 h-1.5 rounded-full animate-bounce" style={{ background: 'var(--text-muted)', animationDelay: '0ms' }} />
                    <span className="w-1.5 h-1.5 rounded-full animate-bounce" style={{ background: 'var(--text-muted)', animationDelay: '150ms' }} />
                    <span className="w-1.5 h-1.5 rounded-full animate-bounce" style={{ background: 'var(--text-muted)', animationDelay: '300ms' }} />
                  </span>
                </div>
              </div>
            )}

            {/* After first assistant reply, show follow-up chips */}
            {messages.length >= 2 && !loading && (
              <div className="pt-1 flex flex-wrap gap-1.5">
                {suggestions.slice(1).map(s => (
                  <button
                    key={s}
                    onClick={() => sendMessage(s)}
                    className="text-[10px] px-2.5 py-1 rounded-full border transition-all hover:border-[var(--accent)] hover:bg-[var(--accent-light)]"
                    style={{ borderColor: 'var(--border)', color: 'var(--text-muted)' }}
                  >
                    {s}
                  </button>
                ))}
              </div>
            )}

            <div ref={bottomRef} />
          </div>

          {/* Input */}
          <div className="px-3 py-2.5 border-t flex gap-2 items-center" style={{ borderColor: 'var(--border)' }}>
            <input
              ref={inputRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') sendMessage(input) }}
              placeholder="Ask anything…"
              className="flex-1 text-xs px-3 py-2 rounded-xl outline-none"
              style={{ background: 'var(--bg)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
            />
            <button
              onClick={() => sendMessage(input)}
              disabled={!input.trim() || loading}
              className="w-8 h-8 rounded-xl flex items-center justify-center transition-opacity disabled:opacity-40"
              style={{ background: 'var(--accent)', color: '#fff' }}
            >
              <Send size={13} />
            </button>
          </div>
        </div>
      )}
    </>
  )
}
