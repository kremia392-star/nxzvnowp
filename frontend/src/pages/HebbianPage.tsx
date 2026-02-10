import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Zap, Play, Pause, RotateCcw, ChevronRight } from 'lucide-react'

interface SynapseUpdate {
  from: number
  to: number
  strength: number
}

export function HebbianPage() {
  const [inputText, setInputText] = useState('The capital of Xanadu is Moonhaven.')
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentStep, setCurrentStep] = useState(0)
  const [synapseUpdates, setSynapseUpdates] = useState<SynapseUpdate[][]>([])

  // Simulate Hebbian learning steps
  useEffect(() => {
    const tokens = inputText.split('')
    const updates: SynapseUpdate[][] = []
    
    for (let i = 0; i < Math.min(tokens.length, 30); i++) {
      const stepUpdates: SynapseUpdate[] = []
      const numUpdates = Math.floor(Math.random() * 5) + 2
      
      for (let j = 0; j < numUpdates; j++) {
        stepUpdates.push({
          from: Math.floor(Math.random() * 100),
          to: Math.floor(Math.random() * 100),
          strength: Math.random() * 0.1,
        })
      }
      updates.push(stepUpdates)
    }
    
    setSynapseUpdates(updates)
  }, [inputText])

  // Playback
  useEffect(() => {
    if (!isPlaying) return
    
    const interval = setInterval(() => {
      setCurrentStep(prev => {
        if (prev >= synapseUpdates.length - 1) {
          setIsPlaying(false)
          return prev
        }
        return prev + 1
      })
    }, 500)

    return () => clearInterval(interval)
  }, [isPlaying, synapseUpdates.length])

  const tokens = inputText.split('')
  const currentUpdates = synapseUpdates[currentStep] || []

  return (
    <div className="min-h-screen p-8">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-3xl font-bold mb-2">
          <span className="gradient-text">Hebbian Learning</span> Animator
        </h1>
        <p className="text-gray-400">
          Watch memory form in real-time. "Neurons that fire together, wire together" — 
          BDH implements this during inference, no backpropagation needed.
        </p>
      </motion.div>

      {/* Input */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card p-6 mb-6"
      >
        <label className="text-sm text-gray-400 mb-2 block">Input sequence (novel fact)</label>
        <input
          type="text"
          value={inputText}
          onChange={(e) => { setInputText(e.target.value); setCurrentStep(0) }}
          className="input-field mb-4"
        />

        {/* Token timeline */}
        <div className="flex flex-wrap gap-1 mb-4">
          {tokens.slice(0, 40).map((token, idx) => (
            <motion.span
              key={idx}
              className={`px-2 py-1 rounded font-mono text-sm transition-all ${
                idx === currentStep
                  ? 'bg-bdh-accent text-white scale-110'
                  : idx < currentStep
                  ? 'bg-gray-700 text-gray-300'
                  : 'bg-gray-800 text-gray-500'
              }`}
              animate={idx === currentStep ? { scale: [1, 1.1, 1] } : {}}
              transition={{ duration: 0.3 }}
            >
              {token === ' ' ? '␣' : token}
            </motion.span>
          ))}
        </div>

        {/* Controls */}
        <div className="flex items-center gap-4">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="btn-primary flex items-center gap-2"
          >
            {isPlaying ? <Pause size={18} /> : <Play size={18} />}
            {isPlaying ? 'Pause' : 'Play'}
          </button>
          <button
            onClick={() => { setCurrentStep(0); setIsPlaying(false) }}
            className="btn-secondary flex items-center gap-2"
          >
            <RotateCcw size={18} />
            Reset
          </button>
          <input
            type="range"
            min={0}
            max={Math.max(0, synapseUpdates.length - 1)}
            value={currentStep}
            onChange={(e) => setCurrentStep(parseInt(e.target.value))}
            className="flex-1"
          />
          <span className="text-gray-400 font-mono">
            {currentStep + 1}/{synapseUpdates.length}
          </span>
        </div>
      </motion.div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Synapse Updates Visualization */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="glass-card p-6"
        >
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Zap size={20} className="text-bdh-accent" />
            Synapse Updates (Token {currentStep + 1})
          </h3>

          <div className="space-y-3">
            <AnimatePresence mode="wait">
              {currentUpdates.map((update, idx) => (
                <motion.div
                  key={`${currentStep}-${idx}`}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  transition={{ delay: idx * 0.1 }}
                  className="flex items-center gap-4 p-3 bg-gray-800/50 rounded-lg"
                >
                  <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded-full bg-blue-500/20 border border-blue-500/40 flex items-center justify-center text-xs font-mono text-blue-400">
                      {update.from}
                    </div>
                    <ChevronRight size={16} className="text-bdh-accent" />
                    <div className="w-8 h-8 rounded-full bg-green-500/20 border border-green-500/40 flex items-center justify-center text-xs font-mono text-green-400">
                      {update.to}
                    </div>
                  </div>
                  <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full bg-gradient-to-r from-bdh-accent to-purple-400"
                      initial={{ width: 0 }}
                      animate={{ width: `${update.strength * 100}%` }}
                      transition={{ duration: 0.3 }}
                    />
                  </div>
                  <span className="text-sm text-gray-400 font-mono">
                    +{update.strength.toFixed(4)}
                  </span>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>

          <p className="text-sm text-gray-500 mt-4">
            Each update shows: neuron_from → neuron_to with strength increase
          </p>
        </motion.div>

        {/* Cumulative Synapse Matrix */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="glass-card p-6"
        >
          <h3 className="text-lg font-semibold mb-4">Cumulative Synapse Strength</h3>
          
          {/* Mini heatmap visualization */}
          <div className="aspect-square bg-gray-800/50 rounded-lg p-4 mb-4">
            <div className="grid grid-cols-10 gap-0.5 h-full">
              {Array.from({ length: 100 }).map((_, idx) => {
                // Simulate cumulative strength
                const row = Math.floor(idx / 10)
                const col = idx % 10
                const baseStrength = Math.random() * 0.3
                const addedStrength = currentStep * 0.02 * (
                  (row + col) % 3 === 0 ? 1 : 0.2
                )
                const strength = Math.min(1, baseStrength + addedStrength)
                
                return (
                  <motion.div
                    key={idx}
                    className="rounded-sm"
                    style={{
                      backgroundColor: `rgba(139, 92, 246, ${strength})`,
                    }}
                    animate={{ opacity: [0.5, 1, 0.5] }}
                    transition={{ 
                      duration: 2, 
                      repeat: Infinity, 
                      delay: idx * 0.01,
                      ease: "easeInOut"
                    }}
                  />
                )
              })}
            </div>
          </div>

          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-500">Weak</span>
            <div className="flex-1 mx-4 h-2 rounded-full bg-gradient-to-r from-gray-800 via-bdh-accent/50 to-bdh-accent" />
            <span className="text-gray-500">Strong</span>
          </div>
        </motion.div>
      </div>

      {/* Learning Demo */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="mt-6 glass-card p-6"
      >
        <h3 className="text-lg font-semibold mb-4">Inference-Time Learning Demo</h3>
        
        <div className="grid md:grid-cols-3 gap-6">
          <div className="p-4 bg-gray-800/50 rounded-lg">
            <div className="text-sm text-gray-400 mb-2">Step 1: Before Exposure</div>
            <div className="p-3 bg-gray-900 rounded font-mono text-sm">
              Q: "The capital of Xanadu is?"<br/>
              A: <span className="text-red-400">[unknown/random]</span>
            </div>
          </div>
          
          <div className="p-4 bg-bdh-accent/10 border border-bdh-accent/30 rounded-lg">
            <div className="text-sm text-bdh-accent mb-2">Step 2: Single Exposure</div>
            <div className="p-3 bg-gray-900 rounded font-mono text-sm">
              Input: "The capital of Xanadu is Moonhaven."<br/>
              <span className="text-bdh-accent">→ Hebbian update occurs</span>
            </div>
          </div>
          
          <div className="p-4 bg-green-500/10 border border-green-500/30 rounded-lg">
            <div className="text-sm text-green-400 mb-2">Step 3: After Learning</div>
            <div className="p-3 bg-gray-900 rounded font-mono text-sm">
              Q: "The capital of Xanadu is?"<br/>
              A: <span className="text-green-400">"Moonhaven"</span>
            </div>
          </div>
        </div>

        <p className="text-gray-400 mt-4">
          <span className="text-white font-medium">Key insight:</span> BDH learns new facts 
          during inference through Hebbian synaptic updates. No gradient descent, no 
          backpropagation, no fine-tuning. Transformers fundamentally cannot do this — 
          they require retraining to learn new information.
        </p>
      </motion.div>
    </div>
  )
}
