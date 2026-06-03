import { NavLink, Outlet } from 'react-router-dom'
import { Upload, Camera, Video, TrafficCone } from 'lucide-react'
import { cn } from '@/lib/utils'
import { ThemeToggle } from '@/components/ThemeToggle'

const NAV = [
  { to: '/', label: 'Tải ảnh', icon: Upload, end: true },
  { to: '/snapshot', label: 'Chụp ảnh', icon: Camera, end: false },
  { to: '/realtime', label: 'Thời gian thực', icon: Video, end: false },
]

/** Global layout: sticky header with nav + theme/lang toggles, routed body. */
export function AppShell() {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-30 border-b bg-background/80 backdrop-blur">
        <div className="container flex h-16 items-center justify-between gap-4">
          <div className="flex items-center gap-2 font-semibold">
            <span className="rounded-md bg-primary p-1.5 text-primary-foreground">
              <TrafficCone className="h-5 w-5" />
            </span>
            <span className="hidden sm:inline">
              Nhận diện biển báo giao thông
            </span>
          </div>

          <nav className="flex items-center gap-1">
            {NAV.map(({ to, label, icon: Icon, end }) => (
              <NavLink
                key={to}
                to={to}
                end={end}
                className={({ isActive }) =>
                  cn(
                    'flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors',
                    isActive
                      ? 'bg-primary/10 text-primary'
                      : 'text-muted-foreground hover:bg-accent hover:text-foreground',
                  )
                }
              >
                <Icon className="h-4 w-4" />
                <span className="hidden md:inline">{label}</span>
              </NavLink>
            ))}
          </nav>

          <div className="flex items-center gap-2">
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="container flex-1 py-8">
        <Outlet />
      </main>
    </div>
  )
}
