import { create } from "zustand";

// Types for our data structures
interface TimelinePoint {
  iteration: number;
  checkpoint: string;
  file: string;
  sparsity: number;
  x_sparsity: number;
  y_sparsity: number;
  sparsity_by_layer: number[];
  graph_density: number;
  total_edges?: number;
  concepts_detected?: number;
}

interface EvolutionTimeline {
  timeline: TimelinePoint[];
  num_checkpoints: number;
  iterations: number[];
  metrics: string[];
}

interface KeyframeData {
  iteration: number;
  checkpoint: string;
  sparsity: {
    overall_sparsity: number;
    overall_x_sparsity: number;
    overall_y_sparsity: number;
    sparsity_by_layer: number[];
  };
  graph: {
    density: number;
    total_edges: number;
  };
  playback_preview?: {
    input_text: string;
    overall_sparsity: number;
    num_frames: number;
    frames: PlaybackFrame[];
  };
}

interface PlaybackFrame {
  token_idx: number;
  token_byte: number;
  token_char: string;
  layer: number;
  x_active: Array<{ indices: number[]; values: number[] }>;
  y_active: Array<{ indices: number[]; values: number[] }>;
  x_sparsity: number;
  y_sparsity: number;
  // Extended interpretability fields
  x_active_count?: number;
  y_active_count?: number;
  x_top_neurons?: Array<{ head: number; neuron: number; value: number }>;
  y_top_neurons?: Array<{ head: number; neuron: number; value: number }>;
  x_pre_relu?: {
    mean: number;
    std: number;
    max: number;
    min: number;
    positive_count: number;
    total: number;
  };
  y_pre_relu?: {
    mean: number;
    std: number;
    max: number;
    min: number;
    positive_count: number;
    total: number;
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
  embedding?: {
    byte_value: number;
    norm: number;
    mean: number;
    std: number;
  };
}

interface ConceptEvolution {
  concepts: {
    [key: string]: Array<{
      iteration: number;
      avg_activation: number;
      consistent_neurons: number;
    }>;
  };
  available_concepts: string[];
}

interface Manifest {
  model_name: string;
  num_checkpoints: number;
  checkpoint_files: string[];
  config: {
    n_layer?: number;
    n_embd?: number;
    n_head?: number;
    n_neurons?: number;
  };
  has_full_checkpoints?: boolean;
}

// Store state
interface EvolutionState {
  // Data
  currentModel: "french" | "portuguese" | "merged";
  manifest: Manifest | null;
  timeline: EvolutionTimeline | null;
  keyframes: KeyframeData[];
  conceptEvolution: ConceptEvolution | null;

  // Current selection
  currentIteration: number;
  currentKeyframeIndex: number;
  isPlaying: boolean;
  playbackSpeed: number; // ms per frame

  // Loading state
  isLoading: boolean;
  loadingProgress: number;
  error: string | null;

  // Cached full checkpoint data
  loadedCheckpoints: { [key: string]: any };

  // Actions
  setModel: (model: "french" | "portuguese" | "merged") => Promise<void>;
  loadModelData: (model: string) => Promise<void>;
  setIteration: (iteration: number) => void;
  setKeyframeIndex: (index: number) => void;
  play: () => void;
  pause: () => void;
  togglePlay: () => void;
  nextKeyframe: () => void;
  prevKeyframe: () => void;
  setPlaybackSpeed: (speed: number) => void;
  loadFullCheckpoint: (checkpoint: string) => Promise<any>;

  // Computed
  getCurrentKeyframe: () => KeyframeData | null;
  getIterationData: (iteration: number) => TimelinePoint | null;
  getSparsityAtIteration: (iteration: number) => number;
  getClosestKeyframeIndex: (iteration: number) => number;
}

// Helper to fetch JSON
async function fetchJSON<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.statusText}`);
  }
  return response.json();
}

export const useEvolutionStore = create<EvolutionState>((set, get) => ({
  // Initial state
  currentModel: "french",
  manifest: null,
  timeline: null,
  keyframes: [],
  conceptEvolution: null,
  currentIteration: 0,
  currentKeyframeIndex: 0,
  isPlaying: false,
  playbackSpeed: 500,
  isLoading: false,
  loadingProgress: 0,
  error: null,
  loadedCheckpoints: {},

  setModel: async (model) => {
    set({ currentModel: model });
    await get().loadModelData(model);
  },

  loadModelData: async (model) => {
    set({ isLoading: true, loadingProgress: 0, error: null });

    const basePath = `/playback/${model}`;

    try {
      // Load manifest first
      set({ loadingProgress: 10 });
      const manifest = await fetchJSON<Manifest>(`${basePath}/manifest.json`);
      set({ manifest, loadingProgress: 30 });

      // Load evolution timeline
      const timeline = await fetchJSON<EvolutionTimeline>(
        `${basePath}/evolution_timeline.json`,
      );
      set({ timeline, loadingProgress: 50 });

      // Load keyframe summary
      const keyframeSummary = await fetchJSON<{ keyframes: KeyframeData[] }>(
        `${basePath}/keyframe_summary.json`,
      );
      set({ keyframes: keyframeSummary.keyframes, loadingProgress: 70 });

      // Load concept evolution
      try {
        const concepts = await fetchJSON<ConceptEvolution>(
          `${basePath}/concept_evolution.json`,
        );
        set({ conceptEvolution: concepts, loadingProgress: 90 });
      } catch {
        // Concept evolution might not exist
        set({ conceptEvolution: null });
      }

      // Set initial iteration to the last one
      const lastIteration =
        timeline.iterations[timeline.iterations.length - 1] || 0;
      set({
        currentIteration: lastIteration,
        currentKeyframeIndex: keyframeSummary.keyframes.length - 1,
        loadingProgress: 100,
        isLoading: false,
      });
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : "Unknown error",
        isLoading: false,
      });
    }
  },

  setIteration: (iteration) => {
    const { keyframes } = get();
    // Find closest keyframe
    let closestIndex = 0;
    let closestDist = Infinity;
    keyframes.forEach((kf, i) => {
      const dist = Math.abs(kf.iteration - iteration);
      if (dist < closestDist) {
        closestDist = dist;
        closestIndex = i;
      }
    });
    set({ currentIteration: iteration, currentKeyframeIndex: closestIndex });
  },

  setKeyframeIndex: (index) => {
    const { keyframes } = get();
    if (index >= 0 && index < keyframes.length) {
      set({
        currentKeyframeIndex: index,
        currentIteration: keyframes[index].iteration,
      });
    }
  },

  play: () => set({ isPlaying: true }),
  pause: () => set({ isPlaying: false }),
  togglePlay: () => set((state) => ({ isPlaying: !state.isPlaying })),

  nextKeyframe: () => {
    const { currentKeyframeIndex, keyframes } = get();
    if (currentKeyframeIndex < keyframes.length - 1) {
      get().setKeyframeIndex(currentKeyframeIndex + 1);
    }
  },

  prevKeyframe: () => {
    const { currentKeyframeIndex } = get();
    if (currentKeyframeIndex > 0) {
      get().setKeyframeIndex(currentKeyframeIndex - 1);
    }
  },

  setPlaybackSpeed: (speed) => set({ playbackSpeed: speed }),

  loadFullCheckpoint: async (checkpoint) => {
    const { currentModel, loadedCheckpoints } = get();

    // Return cached if available
    if (loadedCheckpoints[checkpoint]) {
      return loadedCheckpoints[checkpoint];
    }

    // Load from server
    const url = `/playback/${currentModel}/checkpoints/${checkpoint}.json`;
    try {
      const data = await fetchJSON(url);
      set((state) => ({
        loadedCheckpoints: { ...state.loadedCheckpoints, [checkpoint]: data },
      }));
      return data;
    } catch {
      // Try without checkpoints subdirectory
      const altUrl = `/playback/${currentModel}/${checkpoint}.json`;
      const data = await fetchJSON(altUrl);
      set((state) => ({
        loadedCheckpoints: { ...state.loadedCheckpoints, [checkpoint]: data },
      }));
      return data;
    }
  },

  getCurrentKeyframe: () => {
    const { keyframes, currentKeyframeIndex } = get();
    return keyframes[currentKeyframeIndex] || null;
  },

  getIterationData: (iteration) => {
    const { timeline } = get();
    if (!timeline) return null;
    return timeline.timeline.find((t) => t.iteration === iteration) || null;
  },

  getSparsityAtIteration: (iteration) => {
    const data = get().getIterationData(iteration);
    return data?.sparsity || 0;
  },

  getClosestKeyframeIndex: (iteration) => {
    const { keyframes } = get();
    let closest = 0;
    let minDist = Infinity;
    keyframes.forEach((kf, i) => {
      const dist = Math.abs(kf.iteration - iteration);
      if (dist < minDist) {
        minDist = dist;
        closest = i;
      }
    });
    return closest;
  },
}));

// ============================================================================
// ORIGINAL PLAYBACK STORE (for single checkpoint playback)
// ============================================================================

interface PlaybackData {
  input_text: string;
  input_tokens: number[];
  input_chars: string[];
  num_layers: number;
  num_heads: number;
  neurons_per_head: number;
  frames: PlaybackFrame[];
  overall_sparsity?: number;
  sparsity_by_layer?: number[];
  // Extended fields for interpretability
  embedding_dim?: number;
  total_neurons?: number;
  predictions?: Array<Array<{ byte: number; char: string; prob: number }>>;
}

interface PlaybackState {
  playbackData: PlaybackData | null;
  currentFrame: number;
  isLoading: boolean;
  error: string | null;
  mode: "live" | "playback";
  currentModel: string;

  loadPlayback: (text: string, modelName?: string) => Promise<void>;
  loadFromCheckpoint: (
    model: string,
    checkpoint: string,
    exampleIdx?: number,
  ) => Promise<void>;
  setModel: (model: string) => void;
  setFrame: (frame: number) => void;
  nextFrame: () => void;
  prevFrame: () => void;
  reset: () => void;
}

// Sample data for demo mode
const SAMPLE_PLAYBACK: PlaybackData = {
  input_text: "The capital of France is Paris",
  input_tokens: [84, 104, 101, 32, 99, 97, 112, 105, 116, 97, 108],
  input_chars: ["T", "h", "e", " ", "c", "a", "p", "i", "t", "a", "l"],
  num_layers: 8,
  num_heads: 4,
  neurons_per_head: 8192,
  frames: Array.from({ length: 11 * 8 }, (_, i) => ({
    token_idx: i % 11,
    token_byte: [84, 104, 101, 32, 99, 97, 112, 105, 116, 97, 108][i % 11],
    token_char: ["T", "h", "e", " ", "c", "a", "p", "i", "t", "a", "l"][i % 11],
    layer: Math.floor(i / 11),
    x_active: Array.from({ length: 4 }, () => ({
      indices: Array.from({ length: 50 }, () =>
        Math.floor(Math.random() * 8192),
      ),
      values: Array.from({ length: 50 }, () => Math.random()),
    })),
    y_active: Array.from({ length: 4 }, () => ({
      indices: Array.from({ length: 30 }, () =>
        Math.floor(Math.random() * 8192),
      ),
      values: Array.from({ length: 30 }, () => Math.random()),
    })),
    x_sparsity: 0.93 + Math.random() * 0.04,
    y_sparsity: 0.94 + Math.random() * 0.03,
  })),
  overall_sparsity: 0.947,
  sparsity_by_layer: [0.94, 0.95, 0.96, 0.95, 0.94, 0.95, 0.96, 0.95],
};

export const usePlaybackStore = create<PlaybackState>((set, get) => ({
  playbackData: null,
  currentFrame: 0,
  isLoading: false,
  error: null,
  mode: "playback",
  currentModel: "french",

  loadPlayback: async (text: string, modelName?: string) => {
    const model = modelName || get().currentModel;
    set({ isLoading: true, error: null });

    try {
      // Try API with a 15-second timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 15000);

      const response = await fetch("/api/visualization/playback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, model_name: model }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (response.ok) {
        const data = await response.json();
        set({
          playbackData: data,
          currentFrame: 0,
          isLoading: false,
          mode: "live",
          error: null,
        });
        return;
      } else {
        const errData = await response
          .json()
          .catch(() => ({ detail: "Unknown error" }));
        console.warn("API error:", errData);
        set({ error: `API Error: ${errData.detail || response.statusText}` });
      }
    } catch (err: any) {
      if (err?.name === "AbortError") {
        console.warn("API request timed out");
        set({ error: "Request timed out - showing demo data" });
      } else {
        console.warn("API not available:", err);
        set({ error: "Backend offline - showing demo data" });
      }
    }

    // Fall back to sample data with the input text adjusted
    const chars = Array.from(text);
    const numTokens = chars.length;
    const numLayers = SAMPLE_PLAYBACK.num_layers;
    const numHeads = SAMPLE_PLAYBACK.num_heads;
    const neuronsPerHead = SAMPLE_PLAYBACK.neurons_per_head;

    // Generate frames for ALL tokens across ALL layers
    const frames: PlaybackFrame[] = [];
    for (let layer = 0; layer < numLayers; layer++) {
      for (let t = 0; t < numTokens; t++) {
        const charCode = chars[t].charCodeAt(0);
        frames.push({
          token_idx: t,
          token_byte: charCode,
          token_char: chars[t],
          layer,
          x_active: Array.from({ length: numHeads }, () => ({
            indices: Array.from({ length: 50 }, () =>
              Math.floor(Math.random() * neuronsPerHead),
            ),
            values: Array.from({ length: 50 }, () => Math.random()),
          })),
          y_active: Array.from({ length: numHeads }, () => ({
            indices: Array.from({ length: 30 }, () =>
              Math.floor(Math.random() * neuronsPerHead),
            ),
            values: Array.from({ length: 30 }, () => Math.random()),
          })),
          x_sparsity: 0.93 + Math.random() * 0.04,
          y_sparsity: 0.94 + Math.random() * 0.03,
        });
      }
    }

    const sampleWithText: PlaybackData = {
      ...SAMPLE_PLAYBACK,
      input_text: text,
      input_tokens: chars.map((c) => c.charCodeAt(0)),
      input_chars: chars.map((c) => (c === " " ? " " : c)),
      frames,
    };
    set({
      playbackData: sampleWithText,
      currentFrame: 0,
      isLoading: false,
      mode: "playback",
    });
  },

  setModel: (model: string) => {
    set({ currentModel: model });
  },

  loadFromCheckpoint: async (
    model: string,
    checkpoint: string,
    exampleIdx: number = 0,
  ) => {
    set({ isLoading: true, error: null });

    try {
      const url = `/playback/${model}/${checkpoint}.json`;
      const response = await fetch(url);

      if (!response.ok) {
        throw new Error(`Failed to load ${url}`);
      }

      const data = await response.json();

      // Extract playback data from checkpoint
      if (data.playback && data.playback[exampleIdx]) {
        set({
          playbackData: data.playback[exampleIdx].data,
          currentFrame: 0,
          isLoading: false,
          mode: "playback",
        });
      } else {
        throw new Error("No playback data in checkpoint");
      }
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : "Unknown error",
        isLoading: false,
      });
    }
  },

  setFrame: (frame: number) => {
    const { playbackData } = get();
    if (!playbackData) return;
    const maxFrame = playbackData.frames.length - 1;
    set({ currentFrame: Math.max(0, Math.min(frame, maxFrame)) });
  },

  nextFrame: () => {
    const { playbackData, currentFrame } = get();
    if (!playbackData) return;
    const maxFrame = playbackData.frames.length - 1;
    set({ currentFrame: currentFrame < maxFrame ? currentFrame + 1 : 0 });
  },

  prevFrame: () => {
    const { playbackData, currentFrame } = get();
    if (!playbackData) return;
    const maxFrame = playbackData.frames.length - 1;
    set({ currentFrame: currentFrame > 0 ? currentFrame - 1 : maxFrame });
  },

  reset: () => set({ currentFrame: 0 }),
}));
