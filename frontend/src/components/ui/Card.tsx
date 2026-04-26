import { cn } from '@/lib/utils'

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode
}

export function Card({ className, children, ...props }: CardProps) {
  return (
    <div
      className={cn('rounded-xl border bg-[var(--bg-card)] shadow-sm', className)}
      style={{ borderColor: 'var(--border)' }}
      {...props}
    >
      {children}
    </div>
  )
}

export function CardHeader({ className, children, ...props }: CardProps) {
  return <div className={cn('px-5 pt-5 pb-3', className)} {...props}>{children}</div>
}

export function CardTitle({ className, children, ...props }: CardProps) {
  return <h3 className={cn('text-sm font-semibold text-[var(--text-muted)] uppercase tracking-wider', className)} {...props}>{children}</h3>
}

export function CardContent({ className, children, ...props }: CardProps) {
  return <div className={cn('px-5 pb-5', className)} {...props}>{children}</div>
}
