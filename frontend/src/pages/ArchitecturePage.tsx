import { useState, useEffect, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Play,
  Pause,
  RotateCcw,
  Info,
  Zap,
  SkipForward,
  SkipBack,
  ChevronRight,
} from "lucide-react";
import { BDHArchitectureDiagram } from "@/features/architecture/BDHArchitectureDiagram";
import { usePlaybackStore } from "@/stores/playbackStore";

// Must match STEPS in BDHArchitectureDiagram
const NUM_ARCH_STEPS = 13; // 0..12
const STEP_DURATION = 2000; // ms per architecture step

export function ArchitecturePage() {
  const [inputText, setInputText] = useState("The capital of France is Paris");
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentLayer, setCurrentLayer] = useState(0);
  const [showTooltips, setShowTooltips] = useState(true);

  // === Sequential animation state ===
  const [currentTokenIdx, setCurrentTokenIdx] = useState(0);
  const [currentStep, setCurrentStep] = useState(0);
  const [stepProgress, setStepProgress] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const stepStartRef = useRef<number>(0);

  const {
    playbackData,
    isLoading,
    error,
    mode,
    loadPlayback,
    setFrame,
    reset,
  } = usePlaybackStore();

  // Total number of tokens
  const numTokens = playbackData?.input_chars?.length ?? 0;

  // Find frame index for a specific token and layer
  const findFrameIndex = useCallback(
    (tokenIdx: number, layer: number): number => {
      if (!playbackData) return 0;
      const idx = playbackData.frames.findIndex(
        (f) => f.token_idx === tokenIdx && f.layer === layer,
      );
      return idx >= 0 ? idx : 0;
    },
    [playbackData],
  );

  // Current frame data for display
  const currentFrameData = playbackData
    ? playbackData.frames[findFrameIndex(currentTokenIdx, currentLayer)]
    : undefined;

  // Sync store frame when token/layer changes
  useEffect(() => {
    if (!playbackData) return;
    const idx = findFrameIndex(currentTokenIdx, currentLayer);
    setFrame(idx);
  }, [currentTokenIdx, currentLayer, playbackData, findFrameIndex, setFrame]);

  // Load playback data on mount
  useEffect(() => {
    loadPlayback(inputText);
  }, []);

  // === Master animation timer ===
  useEffect(() => {
    if (!isPlaying || !playbackData) {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      return;
    }

    stepStartRef.current = Date.now();
    setStepProgress(0);

    timerRef.current = setInterval(() => {
      const elapsed = Date.now() - stepStartRef.current;
      const progress = Math.min(elapsed / STEP_DURATION, 1);
      setStepProgress(progress);

      if (progress >= 1) {
        // Step complete — advance
        setCurrentStep((prev) => {
          const nextStep = prev + 1;
          if (nextStep >= NUM_ARCH_STEPS) {
            // All steps done for this token — advance token
            setCurrentTokenIdx((prevToken) => {
              const nextToken = prevToken + 1;
              if (nextToken >= numTokens) {
                // All tokens done — stop playing
                setIsPlaying(false);
                return prevToken;
              }
              return nextToken;
            });
            stepStartRef.current = Date.now();
            setStepProgress(0);
            return 0; // reset to step 0 for new token
          }
          stepStartRef.current = Date.now();
          setStepProgress(0);
          return nextStep;
        });
      }
    }, 30);

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [isPlaying, playbackData, numTokens]);

  const handleRun = () => {
    setIsPlaying(false);
    setCurrentTokenIdx(0);
    setCurrentStep(0);
    setStepProgress(0);
    reset();
    loadPlayback(inputText);
  };

  const handlePlayPause = () => {
    if (isPlaying) {
      setIsPlaying(false);
    } else {
      // If we finished all tokens, restart
      if (
        currentTokenIdx >= numTokens - 1 &&
        currentStep >= NUM_ARCH_STEPS - 1
      ) {
        setCurrentTokenIdx(0);
        setCurrentStep(0);
      }
      setIsPlaying(true);
    }
  };

  const handleReset = () => {
    setIsPlaying(false);
    setCurrentTokenIdx(0);
    setCurrentStep(0);
    setStepProgress(0);
  };

  const handleNextToken = () => {
    setIsPlaying(false);
    if (currentTokenIdx < numTokens - 1) {
      setCurrentTokenIdx((p) => p + 1);
      setCurrentStep(0);
      setStepProgress(0);
    }
  };

  const handlePrevToken = () => {
    setIsPlaying(false);
    if (currentTokenIdx > 0) {
      setCurrentTokenIdx((p) => p - 1);
      setCurrentStep(0);
      setStepProgress(0);
    }
  };

  const handleNextStep = () => {
    setIsPlaying(false);
    if (currentStep < NUM_ARCH_STEPS - 1) {
      setCurrentStep((p) => p + 1);
      setStepProgress(1);
    } else if (currentTokenIdx < numTokens - 1) {
      setCurrentTokenIdx((p) => p + 1);
      setCurrentStep(0);
      setStepProgress(0);
    }
  };

  const handleStepClick = (stepIdx: number) => {
    setIsPlaying(false);
    setCurrentStep(stepIdx);
    setStepProgress(1);
  };

  const handleTokenClick = (tokenIdx: number) => {
    setIsPlaying(false);
    setCurrentTokenIdx(tokenIdx);
    setCurrentStep(0);
    setStepProgress(0);
  };

  return (
    <div className="min-h-screen p-8">
      {/* Loading overlay */}
      <AnimatePresence>
        {isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-gray-900/80 backdrop-blur-sm"
          >
            <div className="flex flex-col items-center gap-4">
              <div className="relative w-16 h-16">
                <div className="absolute inset-0 rounded-full border-4 border-gray-700" />
                <div className="absolute inset-0 rounded-full border-4 border-t-bdh-accent border-r-transparent border-b-transparent border-l-transparent animate-spin" />
              </div>
              <p className="text-gray-300 text-sm font-medium">
                Running inference on model...
              </p>
              <p className="text-gray-500 text-xs">
                Extracting sparse activations
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error display */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-4 p-4 bg-red-500/20 border border-red-500/50 rounded-lg text-red-300"
        >
          <strong>Error:</strong> {error}
        </motion.div>
      )}

      {/* Mode indicator */}
      {playbackData && (
        <div className="mb-4 flex items-center gap-2">
          <span
            className={`px-2 py-1 rounded text-xs font-medium ${
              mode === "live"
                ? "bg-green-500/20 text-green-400 border border-green-500/50"
                : "bg-yellow-500/20 text-yellow-400 border border-yellow-500/50"
            }`}
          >
            {mode === "live" ? "🟢 Live API" : "🟡 Demo Mode"}
          </span>
          {mode === "playback" && (
            <span className="text-xs text-gray-500">
              Using sample data - backend may be offline
            </span>
          )}
        </div>
      )}

      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-3xl font-bold mb-2">
          Interactive <span className="gradient-text">Architecture</span>
        </h1>
        <p className="text-gray-400">
          Explore BDH's data flow with animated visualizations. Watch how ~95%
          of paths go dark at the ReLU — that's sparsity in action.
        </p>
      </motion.div>

      {/* Input Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="glass-card p-6 mb-6"
      >
        <div className="flex gap-4">
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Enter text to visualize..."
            className="input-field flex-1"
          />
          <button onClick={handleRun} className="btn-primary">
            <Zap size={18} className="mr-2" />
            Run
          </button>
        </div>

        {playbackData && (
          <div className="mt-4 flex items-center gap-4">
            {/* Token display - click to select */}
            <div className="flex-1 flex flex-wrap gap-1">
              {playbackData.input_chars.map((char, idx) => (
                <motion.span
                  key={idx}
                  onClick={() => handleTokenClick(idx)}
                  className={`px-2 py-1 rounded font-mono text-sm cursor-pointer transition-all hover:ring-2 hover:ring-bdh-accent/50 ${
                    currentTokenIdx === idx
                      ? "bg-bdh-accent text-white shadow-lg shadow-bdh-accent/30"
                      : idx < currentTokenIdx
                        ? "bg-gray-700 text-gray-300 hover:bg-gray-600"
                        : "bg-gray-800 text-gray-500 hover:bg-gray-700"
                  }`}
                  initial={{ scale: 0.8 }}
                  animate={{ scale: 1 }}
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                  transition={{ delay: idx * 0.02 }}
                >
                  {char === " " ? "␣" : char}
                </motion.span>
              ))}
            </div>

            {/* Playback controls */}
            <div className="flex items-center gap-2">
              <button
                onClick={handlePrevToken}
                disabled={currentTokenIdx === 0}
                className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors disabled:opacity-30"
                title="Previous token"
              >
                <SkipBack size={18} />
              </button>
              <button
                onClick={handlePlayPause}
                className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors"
                title={isPlaying ? "Pause" : "Play (step-by-step)"}
              >
                {isPlaying ? <Pause size={20} /> : <Play size={20} />}
              </button>
              <button
                onClick={handleNextStep}
                className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors"
                title="Next step"
              >
                <ChevronRight size={20} />
              </button>
              <button
                onClick={handleNextToken}
                disabled={currentTokenIdx >= numTokens - 1}
                className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors disabled:opacity-30"
                title="Next token"
              >
                <SkipForward size={18} />
              </button>
              <button
                onClick={handleReset}
                className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors"
                title="Reset"
              >
                <RotateCcw size={18} />
              </button>
            </div>
          </div>
        )}

        {/* Progress indicator */}
        {playbackData && (
          <div className="mt-3 flex items-center gap-3 text-xs text-gray-400">
            <span>
              Token{" "}
              <span className="text-purple-400 font-bold">
                {currentTokenIdx + 1}
              </span>
              /{numTokens}
            </span>
            <span className="text-gray-600">|</span>
            <span>
              Step{" "}
              <span className="text-purple-400 font-bold">
                {currentStep + 1}
              </span>
              /{NUM_ARCH_STEPS}
            </span>
            <span className="text-gray-600">|</span>
            <span>
              Layer{" "}
              <span className="text-purple-400 font-bold">
                {currentLayer + 1}
              </span>
            </span>
            <div className="flex-1" />
            {isPlaying && (
              <span className="text-green-400 flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                Playing
              </span>
            )}
          </div>
        )}
      </motion.div>

      {/* Main Diagram */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="glass-card p-6 mb-6 overflow-x-auto"
      >
        <BDHArchitectureDiagram
          frameData={currentFrameData}
          playbackData={playbackData ?? undefined}
          currentLayer={currentLayer}
          isAnimating={isPlaying}
          currentStep={currentStep}
          onStepChange={handleStepClick}
        />
      </motion.div>

      {/* Layer selector */}
      {playbackData && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="glass-card p-6"
        >
          <h3 className="text-lg font-semibold mb-4">Layer Selection</h3>
          <div className="flex gap-2 flex-wrap">
            {Array.from({ length: playbackData.num_layers }, (_, i) => (
              <button
                key={i}
                onClick={() => {
                  setCurrentLayer(i);
                  setIsPlaying(false);
                }}
                className={`px-4 py-2 rounded-lg font-mono transition-all ${
                  currentLayer === i
                    ? "bg-bdh-accent text-white shadow-lg shadow-bdh-accent/30"
                    : "bg-gray-800 hover:bg-gray-700 text-gray-400"
                }`}
              >
                L{i}
              </button>
            ))}
          </div>

          {/* Sparsity indicator */}
          {currentFrameData && (
            <div className="mt-4 grid grid-cols-2 gap-4">
              <div className="p-4 rounded-lg bg-gray-800/50">
                <div className="text-sm text-gray-400 mb-1">X Sparsity</div>
                <div className="text-2xl font-bold text-bdh-accent">
                  {(currentFrameData.x_sparsity * 100).toFixed(1)}%
                </div>
                <div className="mt-2 h-2 bg-gray-700 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-bdh-accent"
                    initial={{ width: 0 }}
                    animate={{ width: `${currentFrameData.x_sparsity * 100}%` }}
                    transition={{ duration: 0.3 }}
                  />
                </div>
              </div>
              <div className="p-4 rounded-lg bg-gray-800/50">
                <div className="text-sm text-gray-400 mb-1">Y Sparsity</div>
                <div className="text-2xl font-bold text-green-400">
                  {(currentFrameData.y_sparsity * 100).toFixed(1)}%
                </div>
                <div className="mt-2 h-2 bg-gray-700 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-green-400"
                    initial={{ width: 0 }}
                    animate={{ width: `${currentFrameData.y_sparsity * 100}%` }}
                    transition={{ duration: 0.3 }}
                  />
                </div>
              </div>
            </div>
          )}
        </motion.div>
      )}

      {/* Key Insights Panel */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="mt-6 glass-card p-6"
      >
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Zap size={20} className="text-bdh-accent" />
          Key Insights
        </h3>
        <div className="grid md:grid-cols-2 gap-4">
          <InsightCard
            title="Sparsification at ReLU"
            description="Watch the yellow encoder path expand from D=256 to N=8192 dimensions, then ReLU kills ~95% of activations. The diagram literally shows paths going dark."
          />
          <InsightCard
            title="Linear Attention"
            description="The blue attention block computes ϱ += x^T v — a rank-1 update that's O(T) not O(T²). This is why BDH scales to unlimited context."
          />
          <InsightCard
            title="Hebbian State"
            description="The attention state ϱ accumulates co-activation patterns. This IS the Hebbian memory — neurons that fire together strengthen their connection."
          />
          <InsightCard
            title="Gating (x × y)"
            description="The element-wise multiplication of sparse x and y creates even sparser output. Only paths active in BOTH survive — biological-like signal gating."
          />
        </div>
      </motion.div>
    </div>
  );
}

function InsightCard({
  title,
  description,
}: {
  title: string;
  description: string;
}) {
  return (
    <div className="p-4 rounded-lg bg-gray-800/50 border border-gray-700/50">
      <h4 className="font-medium text-bdh-accent mb-2">{title}</h4>
      <p className="text-sm text-gray-400">{description}</p>
    </div>
  );
}
