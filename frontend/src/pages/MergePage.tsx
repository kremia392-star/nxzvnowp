import { useState } from 'react'
import { motion } from 'framer-motion'
import { GitMerge, ArrowRight, Check, Sparkles, Zap } from 'lucide-react'

export function MergePage() {
  const [mergeStep, setMergeStep] = useState<0 | 1 | 2 | 3>(0)

  const steps = [
    { title: 'French Specialist', desc: 'En-Fr translation' },
    { title: 'Portuguese Specialist', desc: 'En-Pt translation' },
    { title: 'Merge', desc: 'Concatenate' },
    { title: 'Polyglot', desc: 'Both languages' },
  ]

  return (
    <div className="min-h-screen p-8">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-3xl font-bold mb-2">
          <span className="gradient-text">Model Merging</span> Explorer
        </h1>
        <p className="text-gray-400">
          Combine separately trained specialists into a unified model — 
          impossible with transformers, natural with BDH.
        </p>
      </motion.div>

      {/* Step Progress */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card p-6 mb-6"
      >
        <div className="flex items-center justify-between">
          {steps.map((step, idx) => (
            <div key={idx} className="flex items-center">
              <motion.button
                onClick={() => setMergeStep(idx as 0 | 1 | 2 | 3)}
                className="flex flex-col items-center"
                whileHover={{ scale: 1.05 }}
              >
                <div className={`w-14 h-14 rounded-full flex items-center justify-center mb-2 transition-all ${
                  idx < mergeStep 
                    ? 'bg-green-500 text-white' 
                    : idx === mergeStep 
                    ? 'bg-bdh-accent text-white ring-4 ring-bdh-accent/30' 
                    : 'bg-gray-800 text-gray-500'
                }`}>
                  {idx < mergeStep ? (
                    <Check size={24} />
                  ) : idx === 0 ? (
                    <span className="text-2xl">🇫🇷</span>
                  ) : idx === 1 ? (
                    <span className="text-2xl">🇵🇹</span>
                  ) : idx === 2 ? (
                    <GitMerge size={24} />
                  ) : (
                    <span className="text-2xl">🌍</span>
                  )}
                </div>
                <span className={`text-sm font-medium ${idx <= mergeStep ? 'text-white' : 'text-gray-500'}`}>
                  {step.title}
                </span>
                <span className="text-xs text-gray-500">{step.desc}</span>
              </motion.button>
              
              {idx < steps.length - 1 && (
                <div className={`mx-6 h-0.5 w-16 ${idx < mergeStep ? 'bg-green-500' : 'bg-gray-700'}`} />
              )}
            </div>
          ))}
        </div>
      </motion.div>

      {/* Visual Merge Diagram */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="glass-card p-8 mb-6"
      >
        <div className="relative h-80">
          {/* French Model */}
          <motion.div
            className="absolute left-0 top-0 w-64 h-40 rounded-xl bg-blue-500/10 border-2 border-blue-500/50 p-4"
            animate={{
              x: mergeStep >= 2 ? 100 : 0,
              y: mergeStep >= 2 ? 70 : 0,
              scale: mergeStep >= 2 ? 0.8 : 1,
              opacity: mergeStep >= 3 ? 0 : 1,
            }}
            transition={{ duration: 0.5 }}
          >
            <div className="flex items-center gap-2 mb-3">
              <span className="text-2xl">🇫🇷</span>
              <span className="font-semibold text-blue-400">French Specialist</span>
            </div>
            <div className="text-sm text-gray-400 mb-2">Neurons: 0 - 8,191</div>
            <div className="grid grid-cols-8 gap-1">
              {Array.from({ length: 32 }).map((_, i) => (
                <div key={i} className="w-3 h-3 rounded-sm bg-blue-500/50" />
              ))}
            </div>
            <div className="text-xs text-gray-500 mt-2">Trained on Europarl En-Fr</div>
          </motion.div>

          {/* Portuguese Model */}
          <motion.div
            className="absolute right-0 top-0 w-64 h-40 rounded-xl bg-green-500/10 border-2 border-green-500/50 p-4"
            animate={{
              x: mergeStep >= 2 ? -100 : 0,
              y: mergeStep >= 2 ? 70 : 0,
              scale: mergeStep >= 2 ? 0.8 : 1,
              opacity: mergeStep >= 3 ? 0 : 1,
            }}
            transition={{ duration: 0.5 }}
          >
            <div className="flex items-center gap-2 mb-3">
              <span className="text-2xl">🇵🇹</span>
              <span className="font-semibold text-green-400">Portuguese Specialist</span>
            </div>
            <div className="text-sm text-gray-400 mb-2">Neurons: 0 - 8,191</div>
            <div className="grid grid-cols-8 gap-1">
              {Array.from({ length: 32 }).map((_, i) => (
                <div key={i} className="w-3 h-3 rounded-sm bg-green-500/50" />
              ))}
            </div>
            <div className="text-xs text-gray-500 mt-2">Trained on Europarl En-Pt</div>
          </motion.div>

          {/* Merge Operation */}
          <motion.div
            className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2"
            animate={{
              scale: mergeStep === 2 ? [1, 1.2, 1] : 1,
              opacity: mergeStep >= 2 ? 1 : 0,
            }}
            transition={{ duration: 0.5 }}
          >
            <div className="w-20 h-20 rounded-full bg-bdh-accent/20 border-2 border-bdh-accent flex items-center justify-center">
              <GitMerge size={32} className="text-bdh-accent" />
            </div>
          </motion.div>

          {/* Merged Model */}
          <motion.div
            className="absolute left-1/2 bottom-0 -translate-x-1/2 w-80 h-48 rounded-xl bg-purple-500/10 border-2 border-purple-500/50 p-4"
            initial={{ opacity: 0, y: 50 }}
            animate={{
              opacity: mergeStep >= 3 ? 1 : 0,
              y: mergeStep >= 3 ? 0 : 50,
            }}
            transition={{ duration: 0.5 }}
          >
            <div className="flex items-center gap-2 mb-3">
              <span className="text-2xl">🌍</span>
              <span className="font-semibold text-purple-400">Merged Polyglot</span>
            </div>
            <div className="text-sm text-gray-400 mb-2">Neurons: 0 - 16,383</div>
            <div className="grid grid-cols-16 gap-0.5">
              {Array.from({ length: 64 }).map((_, i) => (
                <div 
                  key={i} 
                  className={`w-2 h-2 rounded-sm ${
                    i < 32 ? 'bg-blue-500/50' : 'bg-green-500/50'
                  }`} 
                />
              ))}
            </div>
            <div className="flex gap-4 mt-3 text-xs">
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 rounded-sm bg-blue-500" />
                <span className="text-gray-400">French (0-8191)</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 rounded-sm bg-green-500" />
                <span className="text-gray-400">Portuguese (8192-16383)</span>
              </div>
            </div>
          </motion.div>
        </div>
      </motion.div>

      {/* Stats Comparison */}
      <div className="grid md:grid-cols-3 gap-6 mb-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="glass-card p-6"
        >
          <div className="flex items-center gap-2 mb-4">
            <span className="text-2xl">🇫🇷</span>
            <h3 className="font-semibold">French Model</h3>
          </div>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Parameters</span>
              <span className="font-mono">25M</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Neurons/Head</span>
              <span className="font-mono">8,192</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">BLEU (En→Fr)</span>
              <span className="font-mono text-green-400">32.4</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">BLEU (En→Pt)</span>
              <span className="font-mono text-red-400">2.1</span>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="glass-card p-6"
        >
          <div className="flex items-center gap-2 mb-4">
            <span className="text-2xl">🇵🇹</span>
            <h3 className="font-semibold">Portuguese Model</h3>
          </div>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Parameters</span>
              <span className="font-mono">25M</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Neurons/Head</span>
              <span className="font-mono">8,192</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">BLEU (En→Fr)</span>
              <span className="font-mono text-red-400">1.8</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">BLEU (En→Pt)</span>
              <span className="font-mono text-green-400">29.7</span>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="glass-card p-6 ring-2 ring-bdh-accent/30"
        >
          <div className="flex items-center gap-2 mb-4">
            <span className="text-2xl">🌍</span>
            <h3 className="font-semibold text-bdh-accent">Merged Polyglot</h3>
          </div>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Parameters</span>
              <span className="font-mono">~50M</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Neurons/Head</span>
              <span className="font-mono">16,384</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">BLEU (En→Fr)</span>
              <span className="font-mono text-green-400">31.9</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">BLEU (En→Pt)</span>
              <span className="font-mono text-green-400">28.8</span>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Key Insight */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="glass-card p-6"
      >
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Sparkles size={20} className="text-bdh-accent" />
          Why This Matters
        </h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-red-400 mb-2">❌ Transformers Can't Do This</h4>
            <p className="text-gray-400 text-sm">
              Transformer weights are densely interconnected. You cannot simply concatenate 
              two transformer models — the result would be meaningless. Any attempt at 
              "merging" requires careful fine-tuning, distillation, or other complex procedures.
            </p>
          </div>
          <div>
            <h4 className="font-medium text-green-400 mb-2">✅ BDH Does It Naturally</h4>
            <p className="text-gray-400 text-sm">
              BDH's sparse, modular architecture means neurons operate independently. 
              Concatenating two models is as simple as stacking their neuron spaces. 
              No fine-tuning needed — the merged model immediately works for both domains.
            </p>
          </div>
        </div>
        <div className="mt-4 p-4 bg-bdh-accent/10 border border-bdh-accent/30 rounded-lg">
          <p className="text-bdh-accent font-medium">
            <Zap size={16} className="inline mr-2" />
            Implication: Train specialists for specific tasks, merge them freely. 
            This enables <span className="text-white">modular AI development</span> — 
            a paradigm impossible with current transformer architectures.
          </p>
        </div>
      </motion.div>
    </div>
  )
}
