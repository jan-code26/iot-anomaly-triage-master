import { cn } from '@/lib/utils'

interface BadgeProps {
  children: React.ReactNode
  variant?: 'default' | 'outline' | 'ok' | 'warn' | 'alert' | 'veto' | 'indigo'
  className?: string
}

const variants: Record<string, string> = {
  default: 'bg-zinc-100 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300',
  outline: 'border border-[var(--border)] text-[var(--text-muted)]',
  ok:     'bg-green-50 text-green-700 dark:bg-green-950 dark:text-green-400',
  warn:   'bg-amber-50 text-amber-700 dark:bg-amber-950 dark:text-amber-400',
  alert:  'bg-red-50 text-red-700 dark:bg-red-950 dark:text-red-400',
  veto:   'bg-purple-50 text-purple-700 dark:bg-purple-950 dark:text-purple-400',
  indigo: 'bg-indigo-50 text-indigo-700 dark:bg-indigo-950 dark:text-indigo-400',
}

export function Badge({ children, variant = 'default', className }: BadgeProps) {
  return (
    <span className={cn(
      'inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium font-mono',
      variants[variant],
      className
    )}>
      {children}
    </span>
  )
}
