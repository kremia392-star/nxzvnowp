import { useState } from "react";
import { motion } from "framer-motion";
import { BarChart3, Zap, RefreshCw, AlertCircle } from "lucide-react";

export function SparsityPage() {
  const [inputText, setInputText] = useState(
    "The European Parliament adopted the resolution.",
  );
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isLiveMode, setIsLiveMode] = useState(false);
  const [comparisonData, setComparisonData] = useState<{
    bdh: { sparsity: number; activeNeurons: number; totalNeurons: number };
    transformer: {
      sparsity: number;
      activeNeurons: number;
      totalNeurons: number;
    };
  } | null>(null);

  // Demo data fallback
  const demoData = {
    bdh: { sparsity: 0.947, activeNeurons: 1732, totalNeurons: 32768 },
    transformer: { sparsity: 0.05, activeNeurons: 31130, totalNeurons: 32768 },
  };

  const handleAnalyze = async () => {
    setIsLoading(true);
    setError(null);

    try {
      // Try API first
      const response = await fetch("/api/inference/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: inputText,
          model_name: "french",
        }),
      });

      if (response.ok) {
        const data = await response.json();
        const totalNeurons =
          data.num_heads * data.neurons_per_head * data.num_layers;
        const activeNeurons = Math.round(
          totalNeurons * (1 - data.overall_sparsity),
        );

        setComparisonData({
          bdh: {
            sparsity: data.overall_sparsity,
            activeNeurons,
            totalNeurons,
          },
          // Transformer comparison (simulated - they don't have real sparsity)
          transformer: {
            sparsity: 0.05,
            activeNeurons: Math.round(totalNeurons * 0.95),
            totalNeurons,
          },
        });
        setIsLiveMode(true);
        setIsLoading(false);
        return;
      } else {
        const errData = await response.json().catch(() => ({}));
        setError(`API Error: ${errData.detail || response.statusText}`);
      }
    } catch (err) {
      setError("Backend offline - showing demo data");
    }

    // Fall back to demo data
    setComparisonData(demoData);
    setIsLiveMode(false);
    setIsLoading(false);
  };

  return (
    <div className="min-h-screen p-8">
      {/* Error display */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-4 p-4 bg-yellow-500/20 border border-yellow-500/50 rounded-lg text-yellow-300 flex items-center gap-2"
        >
          <AlertCircle size={18} />
          {error}
        </motion.div>
      )}

      {/* Mode indicator */}
      {comparisonData && (
        <div className="mb-4 flex items-center gap-2">
          <span
            className={`px-2 py-1 rounded text-xs font-medium ${
              isLiveMode
                ? "bg-green-500/20 text-green-400 border border-green-500/50"
                : "bg-yellow-500/20 text-yellow-400 border border-yellow-500/50"
            }`}
          >
            {isLiveMode ? "🟢 Live API" : "🟡 Demo Mode"}
          </span>
        </div>
      )}

      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-3xl font-bold mb-2">
          <span className="gradient-text">Sparse Brain</span> Comparator
        </h1>
        <p className="text-gray-400">
          Compare BDH's ~5% activation rate with Transformer's ~95%. Watch how
          different the brain activity looks.
        </p>
      </motion.div>

      {/* Input */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card p-6 mb-8"
      >
        <div className="flex gap-4">
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Enter text to compare..."
            className="input-field flex-1"
          />
          <button
            onClick={handleAnalyze}
            className="btn-primary flex items-center gap-2"
            disabled={isLoading}
          >
            {isLoading ? (
              <RefreshCw className="animate-spin" size={18} />
            ) : (
              <Zap size={18} />
            )}
            Analyze
          </button>
        </div>
      </motion.div>

      {/* Comparison */}
      <div className="grid md:grid-cols-2 gap-8">
        {/* BDH */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="glass-card p-6"
        >
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-bdh-accent to-purple-600 flex items-center justify-center text-2xl">
              🐉
            </div>
            <div>
              <h2 className="text-xl font-bold">BDH</h2>
              <p className="text-gray-400 text-sm">Baby Dragon Hatchling</p>
            </div>
          </div>

          {/* Sparsity visualization */}
          <div className="mb-6">
            <div className="flex justify-between mb-2">
              <span className="text-gray-400">Sparsity</span>
              <span className="text-bdh-accent font-bold">
                {comparisonData
                  ? `${(comparisonData.bdh.sparsity * 100).toFixed(1)}%`
                  : "--"}
              </span>
            </div>
            <div className="h-4 bg-gray-800 rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-gradient-to-r from-bdh-accent to-purple-500"
                initial={{ width: 0 }}
                animate={{
                  width: comparisonData
                    ? `${comparisonData.bdh.sparsity * 100}%`
                    : "0%",
                }}
                transition={{ duration: 1, ease: "easeOut" }}
              />
            </div>
          </div>

          {/* Neuron grid */}
          <div className="mb-4">
            <p className="text-gray-400 text-sm mb-2">
              Neuron activations (sample of 400)
            </p>
            <div className="grid grid-cols-20 gap-0.5">
              {Array.from({ length: 400 }).map((_, i) => (
                <motion.div
                  key={i}
                  className={`w-2 h-2 rounded-sm ${
                    i <
                    (comparisonData
                      ? 400 * (1 - comparisonData.bdh.sparsity)
                      : 20)
                      ? "bg-bdh-accent"
                      : "bg-gray-800"
                  }`}
                  initial={{ opacity: 0, scale: 0 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: i * 0.002 }}
                />
              ))}
            </div>
          </div>

          <div className="text-center p-4 bg-gray-800/50 rounded-lg">
            <div className="text-3xl font-bold text-bdh-accent">
              {comparisonData
                ? comparisonData.bdh.activeNeurons.toLocaleString()
                : "--"}
            </div>
            <div className="text-gray-400 text-sm">active neurons</div>
            <div className="text-gray-500 text-xs mt-1">
              out of {comparisonData?.bdh.totalNeurons.toLocaleString() || "--"}
            </div>
          </div>
        </motion.div>

        {/* Transformer */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="glass-card p-6"
        >
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-orange-500 to-red-600 flex items-center justify-center">
              <BarChart3 size={24} className="text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold">Transformer</h2>
              <p className="text-gray-400 text-sm">Standard architecture</p>
            </div>
          </div>

          {/* Sparsity visualization */}
          <div className="mb-6">
            <div className="flex justify-between mb-2">
              <span className="text-gray-400">Sparsity</span>
              <span className="text-orange-400 font-bold">
                {comparisonData
                  ? `${(comparisonData.transformer.sparsity * 100).toFixed(1)}%`
                  : "--"}
              </span>
            </div>
            <div className="h-4 bg-gray-800 rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-gradient-to-r from-orange-500 to-red-500"
                initial={{ width: 0 }}
                animate={{
                  width: comparisonData
                    ? `${comparisonData.transformer.sparsity * 100}%`
                    : "0%",
                }}
                transition={{ duration: 1, ease: "easeOut" }}
              />
            </div>
          </div>

          {/* Neuron grid */}
          <div className="mb-4">
            <p className="text-gray-400 text-sm mb-2">
              Neuron activations (sample of 400)
            </p>
            <div className="grid grid-cols-20 gap-0.5">
              {Array.from({ length: 400 }).map((_, i) => (
                <motion.div
                  key={i}
                  className={`w-2 h-2 rounded-sm ${
                    i <
                    (comparisonData
                      ? 400 * (1 - comparisonData.transformer.sparsity)
                      : 380)
                      ? "bg-orange-500"
                      : "bg-gray-800"
                  }`}
                  initial={{ opacity: 0, scale: 0 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: i * 0.002 }}
                />
              ))}
            </div>
          </div>

          <div className="text-center p-4 bg-gray-800/50 rounded-lg">
            <div className="text-3xl font-bold text-orange-400">
              {comparisonData
                ? comparisonData.transformer.activeNeurons.toLocaleString()
                : "--"}
            </div>
            <div className="text-gray-400 text-sm">active neurons</div>
            <div className="text-gray-500 text-xs mt-1">
              out of{" "}
              {comparisonData?.transformer.totalNeurons.toLocaleString() ||
                "--"}
            </div>
          </div>
        </motion.div>
      </div>

      {/* Insight */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="mt-8 glass-card p-6"
      >
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Zap size={20} className="text-bdh-accent" />
          Key Insight
        </h3>
        <p className="text-gray-400">
          BDH achieves{" "}
          <span className="text-bdh-accent font-bold">~95% sparsity</span>{" "}
          through architectural design, not regularization. After projecting to
          neuron space (D→N), the ReLU activation naturally kills most signals.
          This means each active neuron carries{" "}
          <span className="text-white">
            meaningful, interpretable information
          </span>{" "}
          — unlike transformers where the dense activations make interpretation
          nearly impossible.
        </p>
      </motion.div>
    </div>
  );
}
