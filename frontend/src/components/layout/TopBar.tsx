import { Sun, Moon } from 'lucide-react'
import type { Dataset } from '@/lib/types'
import { DATASET_LABELS } from '@/lib/utils'

interface TopBarProps {
  dataset: Dataset
  onDataset: (d: Dataset) => void
  dark: boolean
  onToggleDark: () => void
}

const DATASETS: Dataset[] = ['FD001', 'FD002', 'FD003', 'FD004']

export function TopBar({ dataset, onDataset, dark, onToggleDark }: TopBarProps) {
  return (
    <header
      className="flex items-center gap-3 px-5 py-3 border-b sticky top-0 z-20"
      style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}
    >
      {/* Dataset switcher */}
      <div className="flex gap-1 flex-wrap">
        {DATASETS.map(ds => (
          <button
            key={ds}
            onClick={() => onDataset(ds)}
            className="px-3 py-1 rounded-md text-xs font-mono font-semibold transition-colors cursor-pointer"
            style={{
              background: dataset === ds ? 'var(--accent)' : 'var(--border)',
              color: dataset === ds ? '#fff' : 'var(--text-muted)',
            }}
          >
            {ds}
          </button>
        ))}
      </div>

      <div className="ml-2 text-xs hidden sm:block" style={{ color: 'var(--text-muted)' }}>
        {DATASET_LABELS[dataset].split(' — ')[1]}
      </div>

      <div className="ml-auto flex items-center gap-2">
        <button
          onClick={onToggleDark}
          className="p-2 rounded-md hover:bg-[var(--border)] transition-colors cursor-pointer"
          style={{ color: 'var(--text-muted)' }}
          aria-label="Toggle dark mode"
        >
          {dark ? <Sun size={15} /> : <Moon size={15} />}
        </button>
      </div>
    </header>
  )
}
