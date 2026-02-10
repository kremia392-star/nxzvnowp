import { motion } from 'framer-motion'
import { NavLink } from 'react-router-dom'
import { 
  Cpu, 
  BarChart3, 
  Network, 
  Brain, 
  Zap,
  GitMerge,
  ArrowRight,
  Sparkles
} from 'lucide-react'

const features = [
  {
    path: '/architecture',
    icon: Cpu,
    title: 'Interactive Architecture',
    description: 'Explore the BDH architecture with animated data flow visualization',
    color: 'from-blue-500 to-cyan-500',
    insight: 'See where sparsity happens',
  },
  {
    path: '/sparsity',
    icon: BarChart3,
    title: 'Sparse Brain',
    description: 'Compare BDH\'s ~5% activation vs Transformer\'s ~95%',
    color: 'from-purple-500 to-pink-500',
    insight: '95% of neurons stay silent',
  },
  {
    path: '/graph',
    icon: Network,
    title: 'Graph Brain',
    description: '3D visualization of emergent network topology',
    color: 'from-green-500 to-emerald-500',
    insight: 'Brain-like connectivity',
  },
  {
    path: '/monosemanticity',
    icon: Brain,
    title: 'Monosemanticity',
    description: 'Discover synapses that encode specific concepts',
    color: 'from-orange-500 to-yellow-500',
    insight: 'Individual concept neurons',
  },
  {
    path: '/hebbian',
    icon: Zap,
    title: 'Hebbian Learning',
    description: 'Watch memory form during inference',
    color: 'from-red-500 to-orange-500',
    insight: 'Learning without backprop',
  },
  {
    path: '/merge',
    icon: GitMerge,
    title: 'Model Merging',
    description: 'Combine French + Portuguese specialists',
    color: 'from-indigo-500 to-purple-500',
    insight: 'Impossible in transformers',
  },
]

const stats = [
  { value: '~5%', label: 'Neurons Active', sublabel: 'vs ~95% in transformers' },
  { value: 'O(T)', label: 'Attention Complexity', sublabel: 'vs O(T²) in transformers' },
  { value: '∞', label: 'Context Length', sublabel: 'constant memory usage' },
  { value: '1', label: 'Synapse = 1 Concept', sublabel: 'monosemantic neurons' },
]

export function HomePage() {
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative py-20 px-8 overflow-hidden">
        {/* Background glow */}
        <div className="absolute inset-0 bg-gradient-radial from-bdh-accent/10 via-transparent to-transparent" />
        
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="relative max-w-4xl mx-auto text-center"
        >
          <motion.div 
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-bdh-accent/10 border border-bdh-accent/20 text-bdh-accent text-sm mb-6"
            initial={{ scale: 0.9 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2 }}
          >
            <Sparkles size={16} />
            <span>KRITI 2026 · AI Interpretability Challenge</span>
          </motion.div>

          <h1 className="text-5xl md:text-6xl font-bold mb-6">
            <span className="gradient-text">BDH Interpretability</span>
            <br />
            <span className="text-gray-300">Suite</span>
          </h1>

          <p className="text-xl text-gray-400 mb-8 max-w-2xl mx-auto">
            The definitive explorer for <span className="text-white">Baby Dragon Hatchling</span> — 
            a biologically-inspired architecture where AI reasoning becomes 
            <span className="text-bdh-accent"> visible and understandable</span>.
          </p>

          <div className="flex items-center justify-center gap-4">
            <NavLink to="/architecture" className="btn-primary flex items-center gap-2">
              Start Exploring
              <ArrowRight size={18} />
            </NavLink>
            <a 
              href="https://arxiv.org/abs/2509.26507" 
              target="_blank"
              rel="noopener noreferrer"
              className="btn-secondary"
            >
              Read the Paper
            </a>
          </div>
        </motion.div>
      </section>

      {/* Stats Section */}
      <section className="py-12 px-8 border-y border-gray-800/50">
        <div className="max-w-6xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-8">
          {stats.map((stat, index) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="text-center"
            >
              <div className="text-4xl font-bold gradient-text mb-1">
                {stat.value}
              </div>
              <div className="text-gray-300 font-medium">{stat.label}</div>
              <div className="text-gray-500 text-sm">{stat.sublabel}</div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-16 px-8">
        <div className="max-w-6xl mx-auto">
          <motion.h2
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-3xl font-bold text-center mb-12"
          >
            Explore BDH's <span className="gradient-text">Unique Properties</span>
          </motion.h2>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <motion.div
                key={feature.path}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
              >
                <NavLink
                  to={feature.path}
                  className="block glass-card p-6 h-full hover:border-bdh-accent/30 transition-all duration-300 group"
                >
                  <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${feature.color} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
                    <feature.icon size={24} className="text-white" />
                  </div>
                  
                  <h3 className="text-xl font-semibold mb-2 group-hover:text-bdh-accent transition-colors">
                    {feature.title}
                  </h3>
                  
                  <p className="text-gray-400 mb-4">
                    {feature.description}
                  </p>

                  <div className="flex items-center gap-2 text-sm text-bdh-accent">
                    <Sparkles size={14} />
                    <span>{feature.insight}</span>
                  </div>
                </NavLink>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-16 px-8 bg-gray-900/30">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-12">
            What Makes BDH <span className="gradient-text">Different</span>
          </h2>

          <div className="space-y-8">
            {[
              {
                title: 'Sparse by Design',
                description: 'Unlike transformers where nearly all neurons fire, BDH achieves ~95% sparsity through ReLU activation after expanding to neuron space. This isn\'t a regularization trick — it\'s architectural.',
              },
              {
                title: 'Monosemantic Neurons',
                description: 'Individual synapses reliably encode specific concepts. You can literally point to a synapse and say "this is the currency synapse" — it fires for dollar, euro, yen, and ONLY currencies.',
              },
              {
                title: 'Hebbian Learning',
                description: '"Neurons that fire together wire together" — BDH implements this during inference. It can learn new facts without backpropagation, forming memories through synaptic co-activation.',
              },
              {
                title: 'Composable Intelligence',
                description: 'Train specialists separately, merge them freely. A French translator and Portuguese translator can be concatenated into a polyglot — transformers fundamentally cannot do this.',
              },
            ].map((item, index) => (
              <motion.div
                key={item.title}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                className="flex gap-6"
              >
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-bdh-accent/20 border border-bdh-accent/40 flex items-center justify-center text-bdh-accent font-bold">
                  {index + 1}
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-2">{item.title}</h3>
                  <p className="text-gray-400">{item.description}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 px-8 text-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          className="max-w-2xl mx-auto"
        >
          <h2 className="text-3xl font-bold mb-4">
            Ready to <span className="gradient-text">See Inside the Dragon</span>?
          </h2>
          <p className="text-gray-400 mb-8">
            Start with the interactive architecture diagram to understand how BDH processes information.
          </p>
          <NavLink to="/architecture" className="btn-primary inline-flex items-center gap-2">
            Explore Architecture
            <ArrowRight size={18} />
          </NavLink>
        </motion.div>
      </section>
    </div>
  )
}
