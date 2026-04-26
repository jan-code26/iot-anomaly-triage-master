import { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { Dataset } from '@/lib/types'
import { Sidebar } from '@/components/layout/Sidebar'
import { TopBar } from '@/components/layout/TopBar'
import { Overview } from '@/pages/Overview'
import { Alerts } from '@/pages/Alerts'
import { Engines } from '@/pages/Engines'
import { EngineDetail } from '@/pages/EngineDetail'
import { Methodology } from '@/pages/Methodology'
import { About } from '@/pages/About'
import { ChatWidget } from '@/components/ui/ChatWidget'

const qc = new QueryClient({ defaultOptions: { queries: { staleTime: 60_000 } } })

export default function App() {
  const [dark, setDark] = useState(() => window.matchMedia('(prefers-color-scheme: dark)').matches)
  const [dataset, setDataset] = useState<Dataset>('FD001')

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
  }, [dark])

  return (
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <div className="flex min-h-screen w-full">
          <Sidebar />
          <div className="flex-1 flex flex-col min-w-0">
            <TopBar dataset={dataset} onDataset={setDataset} dark={dark} onToggleDark={() => setDark(d => !d)} />
            <main className="flex-1 overflow-y-auto" style={{ background: 'var(--bg)' }}>
              <Routes>
                <Route path="/" element={<Overview dataset={dataset} />} />
                <Route path="/alerts" element={<Alerts dataset={dataset} />} />
                <Route path="/engines" element={<Engines dataset={dataset} />} />
                <Route path="/engines/:id" element={<EngineDetail />} />
                <Route path="/methodology" element={<Methodology />} />
                <Route path="/about" element={<About />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </main>
          </div>
        </div>
        <ChatWidget />
      </BrowserRouter>
    </QueryClientProvider>
  )
}
