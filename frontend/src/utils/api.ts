import axios from "axios";

const API_BASE = "/api";

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});

// Inference endpoints
export const inference = {
  run: (text: string, modelName = "french") =>
    api.post("/inference/run", { text, model_name: modelName }),

  generate: (prompt: string, modelName = "french", maxTokens = 50) =>
    api.post("/inference/generate", {
      prompt,
      model_name: modelName,
      max_tokens: maxTokens,
    }),

  extractDetailed: (text: string, modelName = "french") =>
    api.post("/inference/extract-detailed", { text, model_name: modelName }),
};

// Analysis endpoints
export const analysis = {
  sparsity: (texts: string[], modelName = "french") =>
    api.post("/analysis/sparsity", { texts, model_name: modelName }),

  probeConcept: (
    conceptName: string,
    examples: string[],
    modelName = "french",
  ) =>
    api.post("/analysis/probe-concept", {
      concept_name: conceptName,
      examples,
      model_name: modelName,
    }),

  compare: (text: string, modelNames: string[]) =>
    api.post("/analysis/compare", { text, model_names: modelNames }),

  getConceptCategories: () => api.get("/analysis/concept-categories"),
};

// Model endpoints
export const models = {
  list: () => api.get("/models/list"),

  getInfo: (modelName: string) => api.get(`/models/${modelName}`),

  load: (modelName: string, checkpointPath?: string) =>
    api.post("/models/load", {
      model_name: modelName,
      checkpoint_path: checkpointPath,
    }),

  unload: (modelName: string) => api.post(`/models/${modelName}/unload`),

  getGraph: (modelName: string, threshold = 0.01) =>
    api.get(`/models/${modelName}/graph`, { params: { threshold } }),
};

// Visualization endpoints
export const visualization = {
  playback: (text: string, modelName = "french", includeAttention = false) =>
    api.post("/visualization/playback", {
      text,
      model_name: modelName,
      include_attention: includeAttention,
    }),

  hebbianTrack: (text: string, modelName = "french") =>
    api.post("/visualization/hebbian-track", { text, model_name: modelName }),

  getArchitectureSpec: () => api.get("/visualization/architecture-spec"),

  getColorScheme: () => api.get("/visualization/color-scheme"),
};

// Utility function to load playback from static JSON
export async function loadPlaybackJSON(filename: string) {
  const response = await fetch(`/playback/${filename}`);
  if (!response.ok) throw new Error(`Failed to load ${filename}`);
  return response.json();
}

// Health check
export const health = () => api.get("/health");
