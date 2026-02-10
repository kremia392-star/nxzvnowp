import { useState } from 'react'
import { motion } from 'framer-motion'
import { Brain, Search, Sparkles, ArrowRight } from 'lucide-react'

// Sample monosemantic synapse data
const SAMPLE_SYNAPSES = {
  currencies: [
    { layer: 3, head: 2, neuron: 4521, selectivity: 8.7, examples: ['dollar', 'euro', 'pound', 'yen', 'franc'] },
    { layer: 5, head: 1, neuron: 1893, selectivity: 7.2, examples: ['USD', 'EUR', 'GBP', 'currency'] },
    { layer: 4, head: 3, neuron: 6234, selectivity: 6.8, examples: ['peso', 'krona', 'zloty', 'ruble'] },
  ],
  countries: [
    { layer: 2, head: 0, neuron: 3102, selectivity: 9.1, examples: ['France', 'Germany', 'Spain', 'Italy'] },
    { layer: 4, head: 2, neuron: 5678, selectivity: 8.4, examples: ['Poland', 'Czech', 'Hungary', 'Romania'] },
    { layer: 3, head: 1, neuron: 2341, selectivity: 7.9, examples: ['European', 'Union', 'EU', 'member'] },
  ],
  languages: [
    { layer: 1, head: 3, neuron: 7890, selectivity: 9.5, examples: ['English', 'French', 'German', 'Spanish'] },
    { layer: 2, head: 2, neuron: 4567, selectivity: 8.1, examples: ['Portuguese', 'Italian', 'Polish', 'Dutch'] },
  ],
  legal: [
    { layer: 5, head: 0, neuron: 1234, selectivity: 7.6, examples: ['amendment', 'regulation', 'directive'] },
    { layer: 6, head: 2, neuron: 8901, selectivity: 7.3, examples: ['treaty', 'article', 'clause', 'provision'] },
  ],
}

const CONCEPT_CATEGORIES = [
  { id: 'currencies', name: 'Currencies', icon: '💰', color: 'from-yellow-500 to-orange-500' },
  { id: 'countries', name: 'Countries', icon: '🌍', color: 'from-blue-500 to-cyan-500' },
  { id: 'languages', name: 'Languages', icon: '🗣️', color: 'from-purple-500 to-pink-500' },
  { id: 'legal', name: 'Legal Terms', icon: '⚖️', color: 'from-green-500 to-emerald-500' },
]

export function MonosemanticityPage() {
  const [selectedConcept, setSelectedConcept] = useState<string>('currencies')
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedSynapse, setSelectedSynapse] = useState<any>(null)

  const synapses = SAMPLE_SYNAPSES[selectedConcept as keyof typeof SAMPLE_SYNAPSES] || []

  return (
    <div className="min-h-screen p-8">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-3xl font-bold mb-2">
          <span className="gradient-text">Monosemanticity</span> Dashboard
        </h1>
        <p className="text-gray-400">
          Discover synapses that reliably encode specific concepts. 
          Unlike polysemantic transformer neurons, BDH synapses are interpretable.
        </p>
      </motion.div>

      {/* Concept Categories */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8"
      >
        {CONCEPT_CATEGORIES.map((category) => (
          <button
            key={category.id}
            onClick={() => setSelectedConcept(category.id)}
            className={`glass-card p-4 text-left transition-all ${
              selectedConcept === category.id
                ? 'ring-2 ring-bdh-accent'
                : 'hover:border-gray-700'
            }`}
          >
            <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${category.color} flex items-center justify-center text-xl mb-2`}>
              {category.icon}
            </div>
            <h3 className="font-semibold">{category.name}</h3>
            <p className="text-sm text-gray-400">
              {SAMPLE_SYNAPSES[category.id as keyof typeof SAMPLE_SYNAPSES]?.length || 0} synapses
            </p>
          </button>
        ))}
      </motion.div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Synapse List */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="lg:col-span-1 glass-card p-6"
        >
          <div className="flex items-center gap-2 mb-4">
            <Brain size={20} className="text-bdh-accent" />
            <h2 className="text-lg font-semibold">Discovered Synapses</h2>
          </div>

          {/* Search */}
          <div className="relative mb-4">
            <Search size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search synapses..."
              className="input-field pl-10"
            />
          </div>

          {/* Synapse Cards */}
          <div className="space-y-3 max-h-[400px] overflow-y-auto">
            {synapses.map((synapse, idx) => (
              <motion.button
                key={idx}
                onClick={() => setSelectedSynapse(synapse)}
                className={`w-full p-4 rounded-xl text-left transition-all ${
                  selectedSynapse === synapse
                    ? 'bg-bdh-accent/20 border border-bdh-accent/40'
                    : 'bg-gray-800/50 hover:bg-gray-800 border border-transparent'
                }`}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.05 }}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-mono text-sm text-gray-400">
                    L{synapse.layer}_H{synapse.head}_N{synapse.neuron}
                  </span>
                  <span className="text-bdh-accent font-bold">
                    {synapse.selectivity.toFixed(1)}
                  </span>
                </div>
                <div className="flex flex-wrap gap-1">
                  {synapse.examples.slice(0, 3).map((ex: string) => (
                    <span key={ex} className="px-2 py-0.5 bg-gray-700 rounded text-xs text-gray-300">
                      {ex}
                    </span>
                  ))}
                  {synapse.examples.length > 3 && (
                    <span className="text-xs text-gray-500">+{synapse.examples.length - 3}</span>
                  )}
                </div>
              </motion.button>
            ))}
          </div>
        </motion.div>

        {/* Synapse Detail */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="lg:col-span-2 glass-card p-6"
        >
          {selectedSynapse ? (
            <>
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-xl font-bold mb-1">
                    Synapse <span className="text-bdh-accent font-mono">
                      L{selectedSynapse.layer}_H{selectedSynapse.head}_N{selectedSynapse.neuron}
                    </span>
                  </h2>
                  <p className="text-gray-400">
                    This synapse consistently activates for <span className="text-white">{selectedConcept}</span>
                  </p>
                </div>
                <div className="text-right">
                  <div className="text-3xl font-bold text-bdh-accent">
                    {selectedSynapse.selectivity.toFixed(1)}
                  </div>
                  <div className="text-sm text-gray-400">selectivity score</div>
                </div>
              </div>

              {/* Activation Examples */}
              <div className="mb-6">
                <h3 className="text-sm font-semibold text-gray-400 mb-3">ACTIVATING INPUTS</h3>
                <div className="flex flex-wrap gap-2">
                  {selectedSynapse.examples.map((ex: string) => (
                    <motion.div
                      key={ex}
                      className="px-4 py-2 bg-bdh-accent/20 border border-bdh-accent/40 rounded-lg"
                      initial={{ scale: 0.9, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                    >
                      <span className="text-bdh-accent font-medium">{ex}</span>
                    </motion.div>
                  ))}
                </div>
              </div>

              {/* Non-activating */}
              <div className="mb-6">
                <h3 className="text-sm font-semibold text-gray-400 mb-3">DOES NOT ACTIVATE FOR</h3>
                <div className="flex flex-wrap gap-2">
                  {['apple', 'running', 'democracy', 'mountain', 'blue'].map((ex) => (
                    <div
                      key={ex}
                      className="px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-gray-500"
                    >
                      {ex}
                    </div>
                  ))}
                </div>
              </div>

              {/* Activation Visualization */}
              <div className="p-4 bg-gray-800/50 rounded-xl">
                <h3 className="text-sm font-semibold text-gray-400 mb-3">ACTIVATION PATTERN</h3>
                <div className="flex items-end gap-1 h-32">
                  {Array.from({ length: 20 }).map((_, i) => {
                    const isInCategory = i < 8
                    const height = isInCategory 
                      ? 60 + Math.random() * 40 
                      : 5 + Math.random() * 15
                    return (
                      <motion.div
                        key={i}
                        className={`flex-1 rounded-t ${isInCategory ? 'bg-bdh-accent' : 'bg-gray-700'}`}
                        initial={{ height: 0 }}
                        animate={{ height: `${height}%` }}
                        transition={{ delay: i * 0.03, duration: 0.3 }}
                      />
                    )
                  })}
                </div>
                <div className="flex justify-between mt-2 text-xs text-gray-500">
                  <span>← In-category inputs</span>
                  <span>Out-of-category inputs →</span>
                </div>
              </div>
            </>
          ) : (
            <div className="h-full flex items-center justify-center text-gray-500">
              <div className="text-center">
                <Brain size={48} className="mx-auto mb-4 opacity-50" />
                <p>Select a synapse to view details</p>
              </div>
            </div>
          )}
        </motion.div>
      </div>

      {/* Key Insight */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="mt-6 glass-card p-6"
      >
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Sparkles size={20} className="text-bdh-accent" />
          Why This Matters
        </h3>
        <p className="text-gray-400">
          In transformers, neurons are <span className="text-red-400">polysemantic</span> — 
          a single neuron might activate for "cats", "the letter C", and "circular shapes". 
          This makes interpretation nearly impossible. BDH's architecture produces{' '}
          <span className="text-bdh-accent font-medium">monosemantic synapses</span> — 
          each synapse reliably encodes ONE concept. You can literally point to a synapse 
          and say "this is the currency synapse." This is{' '}
          <span className="text-white">interpretability by design</span>.
        </p>
      </motion.div>
    </div>
  )
}
