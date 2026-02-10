import { useEffect, useState, useId, useMemo } from "react";
import { motion } from "framer-motion";

// =============================================================================
// DATA INTERFACES
// =============================================================================

interface FrameData {
  token_idx: number;
  token_char: string;
  token_byte: number;
  layer: number;
  x_sparsity: number;
  y_sparsity: number;
  x_active_count?: number;
  y_active_count?: number;
  x_active: Array<{ indices: number[]; values: number[] }>;
  y_active: Array<{ indices: number[]; values: number[] }>;
  x_top_neurons?: Array<{ head: number; neuron: number; value: number }>;
  y_top_neurons?: Array<{ head: number; neuron: number; value: number }>;
  x_pre_relu?: {
    mean: number;
    std: number;
    max: number;
    min: number;
    positive_count: number;
    total: number;
    histogram?: Array<{ start: number; end: number; count: number }>;
  };
  y_pre_relu?: {
    mean: number;
    std: number;
    max: number;
    min: number;
    positive_count: number;
    total: number;
    histogram?: Array<{ start: number; end: number; count: number }>;
  };
  gating?: {
    x_only: number;
    y_only: number;
    both: number;
    survival_rate: number;
  };
  attention?: number[];
  attention_stats?: {
    top_attended: Array<{ token_idx: number; char: string; weight: number }>;
  };
  attention_weights?: Array<{
    token_idx: number;
    char: string;
    weight: number;
  }>;
  embedding?: {
    byte_value: number;
    norm: number;
    mean: number;
    std: number;
    vector_ds?: number[];
  };
  x_activation_grid?: number[][];
  y_activation_grid?: number[][];
  hadamard_grid?: number[][];
  a_star_ds?: number[];
  a_star_norm?: number;
}

interface PlaybackData {
  input_text: string;
  input_chars: string[];
  num_layers: number;
  num_heads: number;
  neurons_per_head: number;
  embedding_dim?: number;
  total_neurons?: number;
  frames: FrameData[];
  predictions?: Array<Array<{ byte: number; char: string; prob: number }>>;
  rho_matrices?: Record<number, number[][]>;
}

interface Props {
  frameData?: FrameData;
  playbackData?: PlaybackData;
  showTooltips?: boolean;
  currentLayer: number;
  isAnimating: boolean;
  currentStep?: number;
  onStepChange?: (step: number) => void;
}

// =============================================================================
// CONSTANTS
// =============================================================================

const STEPS = [
  { id: 0, name: "Input Token", key: "input" },
  { id: 1, name: "Embedding", key: "embedding" },
  { id: 2, name: "LayerNorm", key: "layernorm" },
  { id: 3, name: "Linear Dₓ", key: "linear_dx" },
  { id: 4, name: "ReLU (x)", key: "relu_x" },
  { id: 5, name: "ρ Memory", key: "rho" },
  { id: 6, name: "a* Readout", key: "a_star" },
  { id: 7, name: "Linear Dᵧ", key: "linear_dy" },
  { id: 8, name: "ReLU (y)", key: "relu_y" },
  { id: 9, name: "Hadamard x⊙y", key: "hadamard" },
  { id: 10, name: "Decoder", key: "decoder" },
  { id: 11, name: "Residual", key: "residual" },
  { id: 12, name: "Output", key: "output" },
] as const;

const STEP_DURATION = 2500;

// =============================================================================
// COLOR UTILITIES
// =============================================================================

/** Blue (negative) → dark → red/orange (positive) */
function divergingColor(v: number, maxAbs: number): string {
  const norm = maxAbs > 0 ? v / maxAbs : 0;
  if (norm < 0) {
    const t = Math.min(1, -norm);
    return `rgb(${Math.round(20 + t * 10)}, ${Math.round(30 + t * 100)}, ${Math.round(60 + t * 195)})`;
  }
  const t = Math.min(1, norm);
  return `rgb(${Math.round(60 + t * 195)}, ${Math.round(30 + t * 60)}, ${Math.round(20)})`;
}

/** Dark → bright cyan for attention/ρ values */
function rhoColor(v: number, maxAbs: number): string {
  if (maxAbs <= 0) return "#0F172A";
  const norm = v / maxAbs;
  if (norm <= 0) {
    const t = Math.min(1, Math.abs(norm));
    return `rgb(${Math.round(15 + t * 5)}, ${Math.round(20 + t * 10)}, ${Math.round(40 + t * 60)})`;
  }
  const t = Math.min(1, norm);
  return `rgb(${Math.round(15 + t * 20)}, ${Math.round(40 + t * 180)}, ${Math.round(60 + t * 195)})`;
}

/** Dark → bright activation color */
function activationColor(
  v: number,
  maxVal: number,
  hue: "red" | "green" | "cyan" = "red",
): string {
  if (v <= 0) return "#0F172A";
  const t = Math.min(1, v / (maxVal || 0.001));
  switch (hue) {
    case "red":
      return `rgb(${Math.round(30 + t * 225)}, ${Math.round(15 + t * 50)}, ${Math.round(15 + t * 25)})`;
    case "green":
      return `rgb(${Math.round(15 + t * 25)}, ${Math.round(30 + t * 225)}, ${Math.round(25 + t * 105)})`;
    case "cyan":
      return `rgb(${Math.round(15 + t * 25)}, ${Math.round(30 + t * 190)}, ${Math.round(50 + t * 205)})`;
  }
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function BDHArchitectureDiagram({
  frameData,
  playbackData,
  currentLayer,
  isAnimating,
  currentStep: externalStep,
  onStepChange,
}: Props) {
  const [internalStep, setInternalStep] = useState(0);
  const [fillProgress, setFillProgress] = useState(1);
  const uniqueId = useId();
  const currentStep = externalStep ?? internalStep;

  const config = useMemo(
    () => ({
      d: playbackData?.embedding_dim ?? 256,
      n: playbackData?.neurons_per_head ?? 8192,
      h: playbackData?.num_heads ?? 4,
      total: playbackData?.total_neurons ?? 32768,
    }),
    [playbackData],
  );

  const predictions = useMemo(() => {
    if (!playbackData?.predictions || !frameData) return null;
    return playbackData.predictions[frameData.token_idx];
  }, [playbackData, frameData]);

  // Get the ρ matrix for the current layer
  const rhoMatrix = useMemo(() => {
    if (!playbackData?.rho_matrices || frameData === undefined) return null;
    return playbackData.rho_matrices[currentLayer] ?? null;
  }, [playbackData, currentLayer, frameData]);

  useEffect(() => {
    if (!isAnimating) {
      setFillProgress(1);
      return;
    }
    setFillProgress(0);
    const startTime = Date.now();
    const fillInterval = setInterval(() => {
      setFillProgress(Math.min((Date.now() - startTime) / STEP_DURATION, 1));
    }, 30);
    const stepTimeout = setTimeout(() => {
      const next = (currentStep + 1) % STEPS.length;
      onStepChange ? onStepChange(next) : setInternalStep(next);
    }, STEP_DURATION);
    return () => {
      clearInterval(fillInterval);
      clearTimeout(stepTimeout);
    };
  }, [isAnimating, currentStep, onStepChange]);

  const isActive = (s: number) => currentStep >= s;
  const isCurrent = (s: number) => currentStep === s;
  const getProgress = (s: number) =>
    currentStep > s ? 1 : currentStep === s ? fillProgress : 0;

  /* ---- Layout constants ---- */
  const W = 780;
  const LX = 30; // left column x
  const LW = 310; // left column width
  const RX = 420; // right column x
  const RW = 330; // right column width
  const CX = W / 2; // center x

  return (
    <div className="flex gap-6">
      {/* ===== LEFT SIDEBAR ===== */}
      <div className="w-48 flex-shrink-0">
        <div className="glass-card p-3 sticky top-4">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">
            Architecture Flow
          </h3>
          <div className="space-y-1">
            {STEPS.map((step, idx) => (
              <button
                key={step.id}
                onClick={() =>
                  onStepChange ? onStepChange(idx) : setInternalStep(idx)
                }
                className={`w-full flex items-center gap-2 px-2 py-1.5 rounded text-xs transition-all ${
                  idx < currentStep
                    ? "bg-purple-600/30 text-purple-300"
                    : idx === currentStep
                      ? "bg-purple-600 text-white font-semibold"
                      : "bg-gray-800/50 text-gray-500 hover:bg-gray-700/50"
                }`}
              >
                <div
                  className={`w-5 h-5 rounded flex items-center justify-center text-[10px] font-bold ${
                    idx < currentStep
                      ? "bg-purple-500 text-white"
                      : idx === currentStep
                        ? "bg-white text-purple-600"
                        : "bg-gray-700 text-gray-400"
                  }`}
                >
                  {idx}
                </div>
                <span className="truncate">{step.name}</span>
                {idx === currentStep && isAnimating && (
                  <div className="ml-auto w-8 h-1 bg-gray-700 rounded overflow-hidden">
                    <div
                      className="h-full bg-purple-400 transition-all"
                      style={{ width: `${fillProgress * 100}%` }}
                    />
                  </div>
                )}
              </button>
            ))}
          </div>
          <div className="mt-4 pt-3 border-t border-gray-700">
            <div className="text-xs text-gray-400 mb-1">Layer</div>
            <div className="text-lg font-bold text-purple-400">
              {currentLayer + 1} / {playbackData?.num_layers ?? 6}
            </div>
          </div>
        </div>
      </div>

      {/* ===== SVG ARCHITECTURE DIAGRAM ===== */}
      <div className="flex-1 min-w-0">
        <svg
          viewBox={`0 0 ${W} 1380`}
          className="w-full"
          style={{ maxHeight: "1200px" }}
        >
          <defs>
            <pattern
              id={`grid-${uniqueId}`}
              width="25"
              height="25"
              patternUnits="userSpaceOnUse"
            >
              <path
                d="M 25 0 L 0 0 0 25"
                fill="none"
                stroke="#1F2937"
                strokeWidth="0.5"
                opacity="0.3"
              />
            </pattern>
            <linearGradient
              id={`gp-${uniqueId}`}
              x1="0%"
              y1="0%"
              x2="0%"
              y2="100%"
            >
              <stop offset="0%" stopColor="#8B5CF6" />
              <stop offset="100%" stopColor="#6D28D9" />
            </linearGradient>
            <linearGradient
              id={`go-${uniqueId}`}
              x1="0%"
              y1="0%"
              x2="0%"
              y2="100%"
            >
              <stop offset="0%" stopColor="#F59E0B" />
              <stop offset="100%" stopColor="#D97706" />
            </linearGradient>
            <linearGradient
              id={`gr-${uniqueId}`}
              x1="0%"
              y1="0%"
              x2="0%"
              y2="100%"
            >
              <stop offset="0%" stopColor="#EF4444" />
              <stop offset="100%" stopColor="#DC2626" />
            </linearGradient>
            <linearGradient
              id={`gc-${uniqueId}`}
              x1="0%"
              y1="0%"
              x2="0%"
              y2="100%"
            >
              <stop offset="0%" stopColor="#06B6D4" />
              <stop offset="100%" stopColor="#0891B2" />
            </linearGradient>
            <linearGradient
              id={`gg-${uniqueId}`}
              x1="0%"
              y1="0%"
              x2="0%"
              y2="100%"
            >
              <stop offset="0%" stopColor="#10B981" />
              <stop offset="100%" stopColor="#059669" />
            </linearGradient>
          </defs>

          <rect width="100%" height="100%" fill={`url(#grid-${uniqueId})`} />

          {/* ===== TITLE ===== */}
          <text
            x={CX}
            y="25"
            textAnchor="middle"
            fill="#E5E7EB"
            fontSize="15"
            fontWeight="bold"
          >
            BDH Architecture — Layer {currentLayer + 1}
          </text>
          {frameData && (
            <text
              x={CX}
              y="44"
              textAnchor="middle"
              fill="#9CA3AF"
              fontSize="12"
            >
              Token: "<tspan fill="#F59E0B">{frameData.token_char}</tspan>"
              (byte {frameData.token_byte}, position {frameData.token_idx})
            </text>
          )}

          {/* ===== EMBEDDING (y=60, h=110) ===== */}
          <g transform={`translate(${CX - 220}, 60)`}>
            <ArchBox
              width={440}
              height={110}
              title="Embedding"
              gradient={`url(#gp-${uniqueId})`}
              isActive={isActive(1)}
              isCurrent={isCurrent(1) && isAnimating}
              progress={getProgress(1)}
            >
              {frameData?.embedding ? (
                <g>
                  <text
                    x="22"
                    y="36"
                    fill="#C4B5FD"
                    fontSize="11"
                    fontFamily="monospace"
                  >
                    byte {frameData.embedding.byte_value} → v* ∈ ℝ{config.d}
                  </text>
                  {frameData.embedding.vector_ds ? (
                    <HeatmapStrip
                      values={frameData.embedding.vector_ds}
                      x={22}
                      y={44}
                      width={396}
                      height={26}
                    />
                  ) : (
                    <rect
                      x="22"
                      y="44"
                      width="396"
                      height="26"
                      fill="#1F2937"
                      rx="2"
                    />
                  )}
                  <text x="22" y="86" fill="#6B7280" fontSize="8">
                    ← negative (blue)
                  </text>
                  <text
                    x="418"
                    y="86"
                    textAnchor="end"
                    fill="#6B7280"
                    fontSize="8"
                  >
                    positive (red) →
                  </text>
                  <text
                    x="22"
                    y="100"
                    fill="#9CA3AF"
                    fontSize="10"
                    fontFamily="monospace"
                  >
                    ‖v*‖={frameData.embedding.norm.toFixed(2)} μ=
                    {frameData.embedding.mean.toFixed(3)} σ=
                    {frameData.embedding.std.toFixed(3)}
                  </text>
                </g>
              ) : (
                <text
                  x="220"
                  y="60"
                  textAnchor="middle"
                  fill="#9CA3AF"
                  fontSize="12"
                >
                  w_t → v* ∈ ℝ^{config.d}
                </text>
              )}
            </ArchBox>
          </g>

          <FlowArrow x1={CX} y1={170} x2={CX} y2={185} active={isActive(1)} />

          {/* ===== LAYERNORM (y=185, h=48) ===== */}
          <g transform={`translate(${CX - 195}, 185)`}>
            <ArchBox
              width={390}
              height={48}
              title="LayerNorm"
              gradient={`url(#go-${uniqueId})`}
              isActive={isActive(2)}
              isCurrent={isCurrent(2) && isAnimating}
              progress={getProgress(2)}
            >
              <text
                x="195"
                y="34"
                textAnchor="middle"
                fill="#FCD34D"
                fontSize="11"
                fontFamily="monospace"
              >
                v* = (v* − μ) / σ
              </text>
            </ArchBox>
          </g>

          {/* ===== BRANCH ARROWS ===== */}
          <FlowArrow x1={CX} y1={233} x2={CX} y2={248} active={isActive(2)} />
          {/* Left branch to x-path */}
          <path
            d={`M ${CX} 248 L ${CX} 255 L ${LX + LW / 2} 255 L ${LX + LW / 2} 268`}
            stroke={isActive(2) ? "#8B5CF6" : "#374151"}
            strokeWidth="2"
            fill="none"
          />
          {/* Right branch to attention */}
          <path
            d={`M ${CX} 248 L ${CX} 255 L ${RX + RW / 2} 255 L ${RX + RW / 2} 268`}
            stroke={isActive(5) ? "#06B6D4" : "#374151"}
            strokeWidth="2"
            fill="none"
            strokeDasharray="4 2"
          />
          <text x={LX + LW / 2 + 30} y="264" fill="#9CA3AF" fontSize="9">
            to x-path
          </text>
          <text x={RX + RW / 2 - 80} y="264" fill="#67E8F9" fontSize="9">
            to attention (v*)
          </text>

          {/* ================================================================ */}
          {/* LEFT COLUMN: x-path                                              */}
          {/* ================================================================ */}

          {/* ===== LINEAR Dₓ (y=268, h=120) ===== */}
          <g transform={`translate(${LX}, 268)`}>
            <ArchBox
              width={LW}
              height={120}
              title="Linear Dₓ"
              gradient={`url(#go-${uniqueId})`}
              isActive={isActive(3)}
              isCurrent={isCurrent(3) && isAnimating}
              progress={getProgress(3)}
              shape="trapezoid"
            >
              {frameData?.x_pre_relu?.histogram ? (
                <g>
                  <text
                    x={LW / 2}
                    y="34"
                    textAnchor="middle"
                    fill="#FCD34D"
                    fontSize="10"
                    fontFamily="monospace"
                  >
                    x = v* @ E ({config.d}→{config.n})
                  </text>
                  <HistogramViz
                    bins={frameData.x_pre_relu.histogram}
                    x={20}
                    y={42}
                    width={LW - 40}
                    height={44}
                  />
                  <text
                    x="20"
                    y="98"
                    fill="#9CA3AF"
                    fontSize="9"
                    fontFamily="monospace"
                  >
                    range [{frameData.x_pre_relu.min.toFixed(1)},{" "}
                    {frameData.x_pre_relu.max.toFixed(1)}]
                  </text>
                  <text
                    x="20"
                    y="112"
                    fill="#F59E0B"
                    fontSize="9"
                    fontFamily="monospace"
                  >
                    {frameData.x_pre_relu.positive_count}/
                    {frameData.x_pre_relu.total} positive (
                    {(
                      (frameData.x_pre_relu.positive_count /
                        frameData.x_pre_relu.total) *
                      100
                    ).toFixed(1)}
                    %)
                  </text>
                </g>
              ) : (
                <text
                  x={LW / 2}
                  y="65"
                  textAnchor="middle"
                  fill="#FCD34D"
                  fontSize="11"
                  fontFamily="monospace"
                >
                  x = v* @ E ({config.d}→{config.n})
                </text>
              )}
            </ArchBox>
          </g>

          <FlowArrow
            x1={LX + LW / 2}
            y1={388}
            x2={LX + LW / 2}
            y2={405}
            active={isActive(3)}
          />

          {/* ===== RELU x (y=405, h=155) ===== */}
          <g transform={`translate(${LX}, 405)`}>
            <ArchBox
              width={LW}
              height={155}
              title="ReLU (x-path)"
              gradient={`url(#gr-${uniqueId})`}
              isActive={isActive(4)}
              isCurrent={isCurrent(4) && isAnimating}
              progress={getProgress(4)}
            >
              {frameData?.x_activation_grid ? (
                <g>
                  <text
                    x={LW / 2}
                    y="32"
                    textAnchor="middle"
                    fill="#FCA5A5"
                    fontSize="10"
                    fontFamily="monospace"
                  >
                    x_sparse = max(0, x)
                  </text>
                  <NeuronGrid
                    grid={frameData.x_activation_grid}
                    x={18}
                    y={40}
                    width={LW - 36}
                    height={52}
                    hue="red"
                  />
                  {/* Sparsity bar */}
                  <rect
                    x="18"
                    y="98"
                    width={LW - 36}
                    height="7"
                    fill="#1E293B"
                    rx="3"
                  />
                  <rect
                    x="18"
                    y="98"
                    width={(LW - 36) * (1 - frameData.x_sparsity)}
                    height="7"
                    fill="#EF4444"
                    rx="3"
                  />
                  <text
                    x={LW / 2}
                    y="118"
                    textAnchor="middle"
                    fill="#FFF"
                    fontSize="11"
                    fontWeight="bold"
                  >
                    {(frameData.x_sparsity * 100).toFixed(1)}% sparse
                  </text>
                  <text
                    x={LW / 2}
                    y="132"
                    textAnchor="middle"
                    fill="#9CA3AF"
                    fontSize="9"
                  >
                    {frameData.x_active_count ??
                      Math.round(
                        (1 - frameData.x_sparsity) * config.total,
                      )}{" "}
                    / {config.total} neurons active
                  </text>
                  {frameData.x_top_neurons &&
                    frameData.x_top_neurons.length > 0 && (
                      <text
                        x="18"
                        y="146"
                        fill="#F87171"
                        fontSize="8"
                        fontFamily="monospace"
                      >
                        strongest: [H{frameData.x_top_neurons[0]?.head},
                        {frameData.x_top_neurons[0]?.neuron}]=
                        {frameData.x_top_neurons[0]?.value.toFixed(2)}
                      </text>
                    )}
                </g>
              ) : frameData ? (
                <g>
                  <text
                    x={LW / 2}
                    y="55"
                    textAnchor="middle"
                    fill="#FCA5A5"
                    fontSize="11"
                  >
                    Sparsity: {(frameData.x_sparsity * 100).toFixed(1)}%
                  </text>
                  <text
                    x={LW / 2}
                    y="75"
                    textAnchor="middle"
                    fill="#9CA3AF"
                    fontSize="10"
                  >
                    {frameData.x_active_count ?? "?"} / {config.total} active
                  </text>
                </g>
              ) : (
                <text
                  x={LW / 2}
                  y="70"
                  textAnchor="middle"
                  fill="#FCA5A5"
                  fontSize="11"
                >
                  x_sparse = ReLU(x) → ~95% zeros
                </text>
              )}
            </ArchBox>
            {/* x_l label */}
            <text
              x={LW / 2}
              y="172"
              textAnchor="middle"
              fill="#C4B5FD"
              fontSize="12"
              fontWeight="bold"
            >
              x
              <tspan baselineShift="sub" fontSize="9">
                l
              </tspan>
            </text>
          </g>

          {/* Connection: x_l → ρ Memory (x as query) */}
          <path
            d={`M ${LX + LW / 2} 582 L ${LX + LW / 2} 600 L ${RX + RW / 2} 600 L ${RX + RW / 2} 530`}
            stroke={isActive(5) ? "#8B5CF6" : "#374151"}
            strokeWidth="2"
            fill="none"
          />
          <text x={CX} y="615" textAnchor="middle" fill="#9CA3AF" fontSize="9">
            x
            <tspan baselineShift="sub" fontSize="7">
              l
            </tspan>{" "}
            as query into ρ
          </text>

          {/* Connection: x_l down to Hadamard */}
          <path
            d={`M ${LX + LW / 2} 600 L ${LX + LW / 2} 1020 L ${CX - 60} 1020`}
            stroke={isActive(9) ? "#8B5CF6" : "#374151"}
            strokeWidth="2"
            fill="none"
          />

          {/* ================================================================ */}
          {/* RIGHT COLUMN: attention → y-path                                 */}
          {/* ================================================================ */}

          {/* ===== ρ MEMORY STATE (y=268, h=255) ===== */}
          <g transform={`translate(${RX}, 268)`}>
            <ArchBox
              width={RW}
              height={255}
              title="ρ Memory State"
              gradient={`url(#gc-${uniqueId})`}
              isActive={isActive(5)}
              isCurrent={isCurrent(5) && isAnimating}
              progress={getProgress(5)}
            >
              <text
                x={RW / 2}
                y="30"
                textAnchor="middle"
                fill="#67E8F9"
                fontSize="10"
                fontFamily="monospace"
              >
                ρ += x
                <tspan baselineShift="sub" fontSize="7">
                  l
                </tspan>
                <tspan baselineShift="super" fontSize="7">
                  T
                </tspan>{" "}
                ⊗ v* (Hebbian update)
              </text>
              {rhoMatrix && frameData ? (
                <RhoMatrixViz
                  matrix={rhoMatrix}
                  currentT={frameData.token_idx}
                  tokenChars={playbackData?.input_chars ?? []}
                  x={15}
                  y={38}
                  width={RW - 30}
                  height={170}
                />
              ) : (
                <text
                  x={RW / 2}
                  y="130"
                  textAnchor="middle"
                  fill="#67E8F9"
                  fontSize="10"
                >
                  Attention score matrix accumulates
                </text>
              )}
              {rhoMatrix && frameData && (
                <text
                  x={RW / 2}
                  y="225"
                  textAnchor="middle"
                  fill="#6B7280"
                  fontSize="8"
                >
                  Row {frameData.token_idx} = how token "{frameData.token_char}"
                  attends to past
                </text>
              )}
              {!rhoMatrix && (
                <text
                  x={RW / 2}
                  y="150"
                  textAnchor="middle"
                  fill="#475569"
                  fontSize="9"
                >
                  Each row shows attention from token to all past tokens.
                </text>
              )}
              {rhoMatrix && frameData && (
                <text
                  x={RW / 2}
                  y="240"
                  textAnchor="middle"
                  fill="#475569"
                  fontSize="8"
                >
                  ↑ cols = keys (past tokens) · rows = queries
                </text>
              )}
            </ArchBox>
          </g>

          <FlowArrow
            x1={RX + RW / 2}
            y1={523}
            x2={RX + RW / 2}
            y2={540}
            active={isActive(5)}
          />

          {/* ===== a* READOUT (y=540, h=90) ===== */}
          <g transform={`translate(${RX}, 540)`}>
            <ArchBox
              width={RW}
              height={90}
              title="a* Readout"
              gradient={`url(#gc-${uniqueId})`}
              isActive={isActive(6)}
              isCurrent={isCurrent(6) && isAnimating}
              progress={getProgress(6)}
            >
              <text
                x={RW / 2}
                y="30"
                textAnchor="middle"
                fill="#67E8F9"
                fontSize="10"
                fontFamily="monospace"
              >
                a* = Σ ρ[t,i] · v*
                <tspan baselineShift="sub" fontSize="7">
                  i
                </tspan>{" "}
                → ℝ{config.d}
              </text>
              {frameData?.a_star_ds ? (
                <g>
                  <HeatmapStrip
                    values={frameData.a_star_ds}
                    x={18}
                    y={38}
                    width={RW - 36}
                    height={20}
                  />
                  <text
                    x="18"
                    y="72"
                    fill="#9CA3AF"
                    fontSize="9"
                    fontFamily="monospace"
                  >
                    ‖a*‖ = {frameData.a_star_norm?.toFixed(3) ?? "—"}
                  </text>
                  <text
                    x={RW - 18}
                    y="72"
                    textAnchor="end"
                    fill="#475569"
                    fontSize="8"
                  >
                    {frameData.token_idx === 0
                      ? "no past tokens → a*≈0"
                      : "weighted sum of past embeddings"}
                  </text>
                </g>
              ) : (
                <text
                  x={RW / 2}
                  y="55"
                  textAnchor="middle"
                  fill="#475569"
                  fontSize="9"
                >
                  Output of reading from ρ memory
                </text>
              )}
            </ArchBox>
          </g>

          <FlowArrow
            x1={RX + RW / 2}
            y1={630}
            x2={RX + RW / 2}
            y2={648}
            active={isActive(6)}
          />

          {/* ===== LINEAR Dᵧ (y=648, h=120) ===== */}
          <g transform={`translate(${RX}, 648)`}>
            <ArchBox
              width={RW}
              height={120}
              title="Linear Dᵧ"
              gradient={`url(#go-${uniqueId})`}
              isActive={isActive(7)}
              isCurrent={isCurrent(7) && isAnimating}
              progress={getProgress(7)}
              shape="trapezoid"
            >
              {frameData?.y_pre_relu?.histogram ? (
                <g>
                  <text
                    x={RW / 2}
                    y="34"
                    textAnchor="middle"
                    fill="#FCD34D"
                    fontSize="10"
                    fontFamily="monospace"
                  >
                    y = a* @ Ev ({config.d}→{config.n})
                  </text>
                  <HistogramViz
                    bins={frameData.y_pre_relu.histogram}
                    x={20}
                    y={42}
                    width={RW - 40}
                    height={44}
                  />
                  <text
                    x="20"
                    y="98"
                    fill="#9CA3AF"
                    fontSize="9"
                    fontFamily="monospace"
                  >
                    range [{frameData.y_pre_relu.min.toFixed(1)},{" "}
                    {frameData.y_pre_relu.max.toFixed(1)}]
                  </text>
                  <text
                    x="20"
                    y="112"
                    fill="#F59E0B"
                    fontSize="9"
                    fontFamily="monospace"
                  >
                    {frameData.y_pre_relu.positive_count}/
                    {frameData.y_pre_relu.total} positive (
                    {(
                      (frameData.y_pre_relu.positive_count /
                        frameData.y_pre_relu.total) *
                      100
                    ).toFixed(1)}
                    %)
                  </text>
                </g>
              ) : (
                <text
                  x={RW / 2}
                  y="65"
                  textAnchor="middle"
                  fill="#FCD34D"
                  fontSize="11"
                  fontFamily="monospace"
                >
                  y = a* @ Ev ({config.d}→{config.n})
                </text>
              )}
            </ArchBox>
          </g>

          <FlowArrow
            x1={RX + RW / 2}
            y1={768}
            x2={RX + RW / 2}
            y2={785}
            active={isActive(7)}
          />

          {/* ===== RELU y (y=785, h=155) ===== */}
          <g transform={`translate(${RX}, 785)`}>
            <ArchBox
              width={RW}
              height={155}
              title="ReLU (y-path)"
              gradient={`url(#gr-${uniqueId})`}
              isActive={isActive(8)}
              isCurrent={isCurrent(8) && isAnimating}
              progress={getProgress(8)}
            >
              {frameData?.y_activation_grid ? (
                <g>
                  <text
                    x={RW / 2}
                    y="32"
                    textAnchor="middle"
                    fill="#FCA5A5"
                    fontSize="10"
                    fontFamily="monospace"
                  >
                    y_sparse = max(0, y)
                  </text>
                  <NeuronGrid
                    grid={frameData.y_activation_grid}
                    x={18}
                    y={40}
                    width={RW - 36}
                    height={52}
                    hue="green"
                  />
                  <rect
                    x="18"
                    y="98"
                    width={RW - 36}
                    height="7"
                    fill="#1E293B"
                    rx="3"
                  />
                  <rect
                    x="18"
                    y="98"
                    width={(RW - 36) * (1 - frameData.y_sparsity)}
                    height="7"
                    fill="#10B981"
                    rx="3"
                  />
                  <text
                    x={RW / 2}
                    y="118"
                    textAnchor="middle"
                    fill="#FFF"
                    fontSize="11"
                    fontWeight="bold"
                  >
                    {(frameData.y_sparsity * 100).toFixed(1)}% sparse
                  </text>
                  <text
                    x={RW / 2}
                    y="132"
                    textAnchor="middle"
                    fill="#9CA3AF"
                    fontSize="9"
                  >
                    {frameData.y_active_count ??
                      Math.round(
                        (1 - frameData.y_sparsity) * config.total,
                      )}{" "}
                    / {config.total} neurons active
                  </text>
                  {frameData.y_top_neurons &&
                    frameData.y_top_neurons.length > 0 && (
                      <text
                        x="18"
                        y="146"
                        fill="#F87171"
                        fontSize="8"
                        fontFamily="monospace"
                      >
                        strongest: [H{frameData.y_top_neurons[0]?.head},
                        {frameData.y_top_neurons[0]?.neuron}]=
                        {frameData.y_top_neurons[0]?.value.toFixed(2)}
                      </text>
                    )}
                </g>
              ) : frameData ? (
                <g>
                  <text
                    x={RW / 2}
                    y="55"
                    textAnchor="middle"
                    fill="#FCA5A5"
                    fontSize="11"
                  >
                    Sparsity: {(frameData.y_sparsity * 100).toFixed(1)}%
                  </text>
                  <text
                    x={RW / 2}
                    y="75"
                    textAnchor="middle"
                    fill="#9CA3AF"
                    fontSize="10"
                  >
                    {frameData.y_active_count ?? "?"} / {config.total} active
                  </text>
                </g>
              ) : (
                <text
                  x={RW / 2}
                  y="70"
                  textAnchor="middle"
                  fill="#FCA5A5"
                  fontSize="11"
                >
                  y_sparse = ReLU(y)
                </text>
              )}
            </ArchBox>
            {/* y_l label */}
            <text
              x={RW / 2}
              y="172"
              textAnchor="middle"
              fill="#10B981"
              fontSize="12"
              fontWeight="bold"
            >
              y
              <tspan baselineShift="sub" fontSize="9">
                l
              </tspan>
            </text>
          </g>

          {/* Connection: y_l → Hadamard */}
          <path
            d={`M ${RX + RW / 2} 962 L ${RX + RW / 2} 1020 L ${CX + 60} 1020`}
            stroke={isActive(9) ? "#10B981" : "#374151"}
            strokeWidth="2"
            fill="none"
          />

          {/* ================================================================ */}
          {/* CENTER: merge, decode, output                                    */}
          {/* ================================================================ */}

          {/* ===== HADAMARD (y=995) ===== */}
          <g transform={`translate(${CX - 60}, 995)`}>
            <motion.circle
              cx={60}
              cy={28}
              r={28}
              fill={isActive(9) ? "#164E63" : "#1F2937"}
              stroke={isActive(9) ? "#06B6D4" : "#374151"}
              strokeWidth="2"
              animate={
                isCurrent(9) && isAnimating ? { scale: [1, 1.05, 1] } : {}
              }
              transition={{ duration: 1, repeat: Infinity }}
            />
            <text
              x="60"
              y="34"
              textAnchor="middle"
              fill={isActive(9) ? "#67E8F9" : "#6B7280"}
              fontSize="22"
            >
              ⊙
            </text>
            <text
              x="60"
              y="70"
              textAnchor="middle"
              fill="#E5E7EB"
              fontSize="10"
              fontWeight="bold"
            >
              x ⊙ y gating
            </text>
            {frameData?.gating && (
              <>
                <text
                  x="60"
                  y="84"
                  textAnchor="middle"
                  fill="#22D3EE"
                  fontSize="11"
                  fontWeight="bold"
                  fontFamily="monospace"
                >
                  {(frameData.gating.survival_rate * 100).toFixed(0)}% survive
                </text>
                <text
                  x="60"
                  y="98"
                  textAnchor="middle"
                  fill="#9CA3AF"
                  fontSize="9"
                >
                  {frameData.gating.both} neurons pass both gates
                </text>
                <text
                  x="60"
                  y="110"
                  textAnchor="middle"
                  fill="#6B7280"
                  fontSize="8"
                >
                  x-only: {frameData.gating.x_only} | y-only:{" "}
                  {frameData.gating.y_only}
                </text>
              </>
            )}
          </g>

          <FlowArrow x1={CX} y1={1110} x2={CX} y2={1125} active={isActive(9)} />

          {/* ===== DECODER D (y=1125, h=55) ===== */}
          <g transform={`translate(${CX - 160}, 1125)`}>
            <ArchBox
              width={320}
              height={55}
              title="Decoder D"
              gradient={`url(#gg-${uniqueId})`}
              isActive={isActive(10)}
              isCurrent={isCurrent(10) && isAnimating}
              progress={getProgress(10)}
              shape="trapezoidInv"
            >
              <text
                x="160"
                y="38"
                textAnchor="middle"
                fill="#6EE7B7"
                fontSize="11"
                fontFamily="monospace"
              >
                Δv* = (x⊙y) @ D ({config.n}→{config.d})
              </text>
            </ArchBox>
          </g>

          <FlowArrow
            x1={CX}
            y1={1180}
            x2={CX}
            y2={1195}
            active={isActive(10)}
          />

          {/* ===== RESIDUAL (y=1195) ===== */}
          <g transform={`translate(${CX - 25}, 1195)`}>
            <motion.circle
              cx={25}
              cy={20}
              r={20}
              fill={isActive(11) ? "#312E81" : "#1F2937"}
              stroke={isActive(11) ? "#8B5CF6" : "#374151"}
              strokeWidth="2"
              animate={
                isCurrent(11) && isAnimating ? { scale: [1, 1.05, 1] } : {}
              }
              transition={{ duration: 1, repeat: Infinity }}
            />
            <text
              x="25"
              y="26"
              textAnchor="middle"
              fill={isActive(11) ? "#C4B5FD" : "#6B7280"}
              fontSize="18"
              fontWeight="bold"
            >
              ⊕
            </text>
            <text
              x="25"
              y="52"
              textAnchor="middle"
              fill="#9CA3AF"
              fontSize="10"
            >
              v* + Δv*
            </text>
          </g>

          {/* Skip connection line */}
          <path
            d={`M ${CX} 248 L ${W - 30} 248 L ${W - 30} 1215 L ${CX + 25} 1215`}
            stroke={isActive(11) ? "#8B5CF6" : "#374151"}
            strokeWidth="1.5"
            fill="none"
            strokeDasharray="4 2"
            opacity="0.5"
          />

          {/* ===== OUTPUT PREDICTIONS ===== */}
          {predictions && (
            <g transform={`translate(${CX - 240}, 1265)`}>
              <text
                x="240"
                y="0"
                textAnchor="middle"
                fill="#E5E7EB"
                fontSize="11"
                fontWeight="bold"
              >
                Next token predictions:
              </text>
              <g transform="translate(0, 10)">
                {predictions.slice(0, 5).map((p, i) => (
                  <g key={i} transform={`translate(${i * 96}, 0)`}>
                    <rect
                      x="0"
                      y="0"
                      width="90"
                      height="26"
                      rx="5"
                      fill={i === 0 ? "#7C3AED" : "#374151"}
                      opacity={i === 0 ? 1 : 0.7}
                    />
                    <text
                      x="45"
                      y="12"
                      textAnchor="middle"
                      fill="#FFF"
                      fontSize="11"
                      fontWeight="bold"
                    >
                      "{p.char}"
                    </text>
                    <text
                      x="45"
                      y="22"
                      textAnchor="middle"
                      fill="#D1D5DB"
                      fontSize="8"
                    >
                      {(p.prob * 100).toFixed(1)}%
                    </text>
                  </g>
                ))}
              </g>
            </g>
          )}
        </svg>
      </div>
    </div>
  );
}

// =============================================================================
// VISUALIZATION SUB-COMPONENTS
// =============================================================================

/** Renders a small T×T heatmap of the ρ (attention score) matrix */
function RhoMatrixViz({
  matrix,
  currentT,
  tokenChars,
  x,
  y,
  width,
  height,
}: {
  matrix: number[][];
  currentT: number;
  tokenChars: string[];
  x: number;
  y: number;
  width: number;
  height: number;
}) {
  const T = Math.min(currentT + 1, matrix.length);
  if (T <= 0) return null;

  // Extract the sub-matrix up to the current token
  const subMatrix = matrix.slice(0, T).map((row) => row.slice(0, T));

  // Find max absolute value for normalizing colors
  const maxAbs = Math.max(
    0.001,
    ...subMatrix.flatMap((row) => row.map((v) => Math.abs(v))),
  );

  // Layout
  const labelW = Math.min(22, width * 0.1);
  const labelH = Math.min(14, height * 0.08);
  const gridW = width - labelW;
  const gridH = height - labelH;
  const cellW = gridW / T;
  const cellH = gridH / T;
  const showLabels = T <= 20 && cellW > 8;

  return (
    <g transform={`translate(${x}, ${y})`}>
      {/* Background */}
      <rect
        x={labelW}
        y={0}
        width={gridW}
        height={gridH}
        fill="#0F172A"
        rx="3"
      />

      {/* Matrix cells */}
      {subMatrix.map((row, i) =>
        row.map((val, j) => {
          // Only lower triangle has values (causal mask with diagonal=-1)
          const isAboveDiag = j >= i;
          return (
            <rect
              key={`${i}-${j}`}
              x={labelW + j * cellW + 0.3}
              y={i * cellH + 0.3}
              width={Math.max(1, cellW - 0.6)}
              height={Math.max(1, cellH - 0.6)}
              fill={isAboveDiag ? "#0A0F1A" : rhoColor(val, maxAbs)}
              rx="0.5"
            />
          );
        }),
      )}

      {/* Highlight current token's row (the new update) */}
      {currentT < T && (
        <rect
          x={labelW}
          y={currentT * cellH}
          width={gridW}
          height={cellH}
          fill="none"
          stroke="#F59E0B"
          strokeWidth="1.5"
          rx="1"
        />
      )}

      {/* Row labels (token chars) — left axis */}
      {showLabels &&
        tokenChars.slice(0, T).map((c, i) => (
          <text
            key={`rl-${i}`}
            x={labelW - 3}
            y={i * cellH + cellH / 2 + 3}
            textAnchor="end"
            fill={i === currentT ? "#F59E0B" : "#6B7280"}
            fontSize={Math.min(8, cellH - 1)}
            fontFamily="monospace"
            fontWeight={i === currentT ? "bold" : "normal"}
          >
            {c === " " ? "␣" : c.length > 1 ? "·" : c}
          </text>
        ))}

      {/* Column labels (token chars) — top axis */}
      {showLabels &&
        tokenChars.slice(0, T).map((c, j) => (
          <text
            key={`cl-${j}`}
            x={labelW + j * cellW + cellW / 2}
            y={gridH + labelH - 2}
            textAnchor="middle"
            fill="#6B7280"
            fontSize={Math.min(7, cellW - 1)}
            fontFamily="monospace"
          >
            {c === " " ? "␣" : c.length > 1 ? "·" : c}
          </text>
        ))}

      {/* Border */}
      <rect
        x={labelW}
        y={0}
        width={gridW}
        height={gridH}
        fill="none"
        stroke="#374151"
        strokeWidth="0.5"
        rx="3"
      />

      {/* Legend */}
      <text x={labelW} y={gridH + labelH + 10} fill="#475569" fontSize="7">
        dark = low attn
      </text>
      <text
        x={labelW + gridW}
        y={gridH + labelH + 10}
        textAnchor="end"
        fill="#22D3EE"
        fontSize="7"
      >
        bright cyan = high attn
      </text>
      <rect
        x={labelW + gridW - 4}
        y={gridH + labelH + 4}
        width="3"
        height="3"
        fill="#22D3EE"
        rx="0.5"
      />
    </g>
  );
}

/** Renders a 1D heatmap strip — each cell colored by value (blue=neg, red=pos) */
function HeatmapStrip({
  values,
  x,
  y,
  width,
  height,
}: {
  values: number[];
  x: number;
  y: number;
  width: number;
  height: number;
}) {
  const cellW = width / values.length;
  const maxAbs = Math.max(0.001, ...values.map((v) => Math.abs(v)));
  return (
    <g transform={`translate(${x}, ${y})`}>
      <rect width={width} height={height} fill="#0F172A" rx="2" />
      {values.map((v, i) => (
        <rect
          key={i}
          x={i * cellW}
          width={cellW}
          height={height}
          fill={divergingColor(v, maxAbs)}
          rx="0.5"
        />
      ))}
      <rect
        width={width}
        height={height}
        fill="none"
        stroke="#374151"
        strokeWidth="0.5"
        rx="2"
      />
    </g>
  );
}

/** Renders a histogram — negative bins in blue, positive in orange */
function HistogramViz({
  bins,
  x,
  y,
  width,
  height,
}: {
  bins: Array<{ start: number; end: number; count: number }>;
  x: number;
  y: number;
  width: number;
  height: number;
}) {
  const maxCount = Math.max(1, ...bins.map((b) => b.count));
  const barW = width / bins.length;
  const zeroIdx = bins.findIndex((b) => b.end >= 0);
  return (
    <g transform={`translate(${x}, ${y})`}>
      <rect width={width} height={height} fill="#0F172A" rx="2" />
      {bins.map((bin, i) => {
        const barH = (bin.count / maxCount) * (height - 2);
        const isPositive = bin.start >= 0;
        return (
          <rect
            key={i}
            x={i * barW + 0.5}
            y={height - barH - 1}
            width={barW - 1}
            height={barH}
            fill={isPositive ? "#F59E0B" : "#3B82F6"}
            opacity={0.85}
            rx="0.5"
          />
        );
      })}
      {zeroIdx >= 0 && (
        <line
          x1={zeroIdx * barW}
          y1={0}
          x2={zeroIdx * barW}
          y2={height}
          stroke="#EF4444"
          strokeWidth="1.5"
          strokeDasharray="2 2"
        />
      )}
      <rect
        width={width}
        height={height}
        fill="none"
        stroke="#374151"
        strokeWidth="0.5"
        rx="2"
      />
      <text
        x={width / 2}
        y={-3}
        textAnchor="middle"
        fill="#6B7280"
        fontSize="7"
      >
        ← negative (killed) | positive (survives) →
      </text>
    </g>
  );
}

/** Renders a neuron activation grid — rows=heads, cols=neuron bins */
function NeuronGrid({
  grid,
  x,
  y,
  width,
  height,
  hue = "red",
}: {
  grid: number[][];
  x: number;
  y: number;
  width: number;
  height: number;
  hue?: "red" | "green" | "cyan";
}) {
  const numHeads = grid.length;
  const bins = grid[0]?.length || 0;
  if (bins === 0) return null;
  const maxVal = Math.max(0.001, ...grid.flat());
  const labelW = 20;
  const cellW = (width - labelW) / bins;
  const cellH = height / numHeads;
  return (
    <g transform={`translate(${x}, ${y})`}>
      <rect
        x={labelW}
        width={width - labelW}
        height={height}
        fill="#0F172A"
        rx="2"
      />
      {grid.map((row, h) => (
        <g key={h}>
          <text
            x="0"
            y={h * cellH + cellH / 2 + 3}
            fill="#9CA3AF"
            fontSize="8"
            fontFamily="monospace"
          >
            H{h}
          </text>
          {row.map((val, b) => (
            <rect
              key={b}
              x={labelW + b * cellW}
              y={h * cellH}
              width={Math.max(0.5, cellW - 0.3)}
              height={cellH - 0.5}
              fill={activationColor(val, maxVal, hue)}
            />
          ))}
        </g>
      ))}
      <rect
        x={labelW}
        width={width - labelW}
        height={height}
        fill="none"
        stroke="#374151"
        strokeWidth="0.5"
        rx="2"
      />
    </g>
  );
}

// =============================================================================
// STRUCTURAL COMPONENTS
// =============================================================================

interface ArchBoxProps {
  width: number;
  height: number;
  title: string;
  gradient: string;
  isActive: boolean;
  isCurrent: boolean;
  progress: number;
  shape?: "rect" | "trapezoid" | "trapezoidInv";
  children?: React.ReactNode;
}

function ArchBox({
  width,
  height,
  title,
  gradient,
  isActive,
  isCurrent,
  progress,
  shape = "rect",
  children,
}: ArchBoxProps) {
  const offset = 15;
  const getPath = () => {
    switch (shape) {
      case "trapezoid":
        return `M 0 0 L ${width} 0 L ${width - offset} ${height} L ${offset} ${height} Z`;
      case "trapezoidInv":
        return `M ${offset} 0 L ${width - offset} 0 L ${width} ${height} L 0 ${height} Z`;
      default:
        return `M 0 0 L ${width} 0 L ${width} ${height} L 0 ${height} Z`;
    }
  };
  const fillHeight = height * progress;
  const fillY = height - fillHeight;
  const clipId = `clip-${title.replace(/\s+/g, "-")}-${Math.random().toString(36).substr(2, 5)}`;

  return (
    <g>
      <path d={getPath()} fill="#111827" stroke="#1F2937" strokeWidth="1" />
      <defs>
        <clipPath id={clipId}>
          <rect x="0" y={fillY} width={width} height={fillHeight} />
        </clipPath>
      </defs>
      <g clipPath={`url(#${clipId})`}>
        <motion.path
          d={getPath()}
          fill={isActive ? gradient : "#1F2937"}
          opacity={isActive ? 0.3 : 0}
          animate={isCurrent ? { opacity: [0.2, 0.4, 0.2] } : {}}
          transition={{ duration: 1.5, repeat: Infinity }}
        />
      </g>
      <path
        d={getPath()}
        fill="none"
        stroke={isActive ? "#8B5CF6" : "#374151"}
        strokeWidth="2"
      />
      <text
        x={width / 2}
        y="18"
        textAnchor="middle"
        fill={isActive ? "#E5E7EB" : "#6B7280"}
        fontSize="12"
        fontWeight="bold"
      >
        {title}
      </text>
      {children}
    </g>
  );
}

function FlowArrow({
  x1,
  y1,
  x2,
  y2,
  active,
}: {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  active: boolean;
}) {
  return (
    <line
      x1={x1}
      y1={y1}
      x2={x2}
      y2={y2}
      stroke={active ? "#8B5CF6" : "#374151"}
      strokeWidth={active ? 2 : 1.5}
    />
  );
}
