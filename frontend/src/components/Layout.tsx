import { Outlet, NavLink } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  Home, 
  Cpu, 
  BarChart3, 
  Network, 
  Brain, 
  Zap,
  GitMerge,
  Github,
  ExternalLink
} from 'lucide-react'

const navItems = [
  { path: '/', icon: Home, label: 'Home' },
  { path: '/architecture', icon: Cpu, label: 'Architecture' },
  { path: '/sparsity', icon: BarChart3, label: 'Sparsity' },
  { path: '/graph', icon: Network, label: 'Graph Brain' },
  { path: '/monosemanticity', icon: Brain, label: 'Monosemanticity' },
  { path: '/hebbian', icon: Zap, label: 'Hebbian' },
  { path: '/merge', icon: GitMerge, label: 'Model Merge' },
]

export function Layout() {
  return (
    <div className="min-h-screen flex">
      {/* Sidebar */}
      <motion.aside 
        initial={{ x: -100, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        className="w-64 bg-gray-900/50 backdrop-blur-xl border-r border-gray-800/50 flex flex-col"
      >
        {/* Logo */}
        <div className="p-6 border-b border-gray-800/50">
          <NavLink to="/" className="flex items-center gap-3 group">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-bdh-accent to-purple-600 flex items-center justify-center text-2xl group-hover:scale-110 transition-transform">
              🐉
            </div>
            <div>
              <h1 className="font-bold text-lg gradient-text">BDH Suite</h1>
              <p className="text-xs text-gray-500">Interpretability Explorer</p>
            </div>
          </NavLink>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-1">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 group ${
                  isActive
                    ? 'bg-bdh-accent/20 text-bdh-accent'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'
                }`
              }
            >
              {({ isActive }) => (
                <>
                  <item.icon 
                    size={20} 
                    className={`transition-transform group-hover:scale-110 ${
                      isActive ? 'text-bdh-accent' : ''
                    }`}
                  />
                  <span className="font-medium">{item.label}</span>
                  {isActive && (
                    <motion.div
                      layoutId="nav-indicator"
                      className="ml-auto w-1.5 h-1.5 rounded-full bg-bdh-accent"
                    />
                  )}
                </>
              )}
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="p-4 border-t border-gray-800/50 space-y-3">
          <a
            href="https://github.com/pathwaycom/bdh"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-gray-500 hover:text-gray-300 text-sm transition-colors"
          >
            <Github size={16} />
            <span>BDH Paper</span>
            <ExternalLink size={12} className="ml-auto" />
          </a>
          <div className="text-xs text-gray-600">
            KRITI 2026 · AI Interpretability
          </div>
        </div>
      </motion.aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  )
}
