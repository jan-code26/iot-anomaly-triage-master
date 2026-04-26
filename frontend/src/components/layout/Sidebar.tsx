import { NavLink, useLocation } from 'react-router-dom'
import { LayoutDashboard, Bell, Cpu, BookOpen, Info, ChevronRight } from 'lucide-react'
import { cn } from '@/lib/utils'

const NAV = [
  { to: '/',            label: 'Overview',      icon: LayoutDashboard },
  { to: '/alerts',      label: 'Alerts',        icon: Bell },
  { to: '/engines',     label: 'Engines',       icon: Cpu },
  { to: '/methodology', label: 'Methodology',   icon: BookOpen },
  { to: '/about',       label: 'About / Paper', icon: Info },
]

export function Sidebar() {
  const loc = useLocation()
  return (
    <aside
      className="hidden lg:flex flex-col w-56 shrink-0 border-r min-h-screen"
      style={{ background: 'var(--bg-sidebar)', borderColor: 'var(--border)' }}
    >
      {/* Logo */}
      <div className="px-5 py-5 border-b" style={{ borderColor: 'var(--border)' }}>
        <span className="text-sm font-bold tracking-tight" style={{ color: 'var(--text-primary)' }}>
          AnomalyTriage
        </span>
        <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>IoT · CMAPSS</p>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-0.5">
        {NAV.map(({ to, label, icon: Icon }) => {
          const active = to === '/' ? loc.pathname === '/' : loc.pathname.startsWith(to)
          return (
            <NavLink
              key={to}
              to={to}
              className={cn(
                'flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors group',
                active
                  ? 'bg-[var(--accent-light)] font-medium'
                  : 'hover:bg-[var(--border)]'
              )}
              style={{ color: active ? 'var(--accent)' : 'var(--text-muted)' }}
            >
              <Icon size={15} />
              <span>{label}</span>
              {active && <ChevronRight size={12} className="ml-auto opacity-60" />}
            </NavLink>
          )
        })}
      </nav>

      {/* Footer */}
      <div className="px-5 py-4 border-t text-xs" style={{ borderColor: 'var(--border)', color: 'var(--text-muted)' }}>
        v1.0 · CMAPSS
      </div>
    </aside>
  )
}
