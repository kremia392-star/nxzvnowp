"""
Visualization API Routes

Endpoints for visualization data:
- Playback data for frontend
- Graph layout computation
- Hebbian state tracking
- Architecture specification
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field
import torch
import numpy as np

router = APIRouter()


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class PlaybackRequest(BaseModel):
    """Request for playback data generation."""
    text: str
    model_name: str = Field(default="french")
    include_attention: bool = Field(default=True)


class HebbianTrackRequest(BaseModel):
    """Request for Hebbian state tracking."""
    text: str
    model_name: str = Field(default="french")


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post("/playback")
async def generate_playback(request: PlaybackRequest, req: Request):
    """
    Generate playback data for frontend animation with interpretability data.
    """
    import time
    t0 = time.perf_counter()
    
    model_service = req.app.state.model_service
    
    try:
        model = model_service.get_or_load(request.model_name)
        config = model_service.get_config(request.model_name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model error: {e}")
    
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "training"))
    from bdh import ExtractionConfig
    
    tokens = torch.tensor(
        [list(request.text.encode('utf-8'))],
        dtype=torch.long,
        device=model_service.device
    )
    T = tokens.shape[1]
    
    extraction_config = ExtractionConfig(
        capture_sparse_activations=True,
        capture_attention_patterns=request.include_attention,
        capture_pre_relu=True,
    )
    
    t1 = time.perf_counter()
    
    # Run model and extract activations
    with torch.no_grad():
        with model.extraction_mode(extraction_config) as buffer:
            logits, _ = model(tokens)
            
            # Get embedding
            embed_output = model.embed(tokens)
            
            # Transfer to CPU
            layer_data_cpu = {}
            for layer_idx in sorted(buffer.x_sparse.keys()):
                x_cpu = buffer.x_sparse[layer_idx][0].cpu()
                y_cpu = buffer.y_sparse[layer_idx][0].cpu()
                
                x_pre = None
                y_pre = None
                attn_cpu = None
                attn_out_cpu = None
                
                if layer_idx in buffer.x_pre_relu:
                    x_pre = buffer.x_pre_relu[layer_idx][0].cpu()
                if layer_idx in buffer.y_pre_relu:
                    y_pre = buffer.y_pre_relu[layer_idx][0].cpu()
                if request.include_attention and layer_idx in buffer.attention_scores:
                    attn_cpu = buffer.attention_scores[layer_idx][0].cpu()
                if layer_idx in buffer.attention_output:
                    attn_out_cpu = buffer.attention_output[layer_idx][0].cpu()
                
                layer_data_cpu[layer_idx] = {
                    'x_sparse': x_cpu,
                    'y_sparse': y_cpu,
                    'x_pre_relu': x_pre,
                    'y_pre_relu': y_pre,
                    'attention': attn_cpu,
                    'attention_output': attn_out_cpu,
                }
            
            logits_cpu = logits[0].cpu()
            embed_cpu = embed_output[0].cpu()
    
    t2 = time.perf_counter()
    
    # Build token metadata
    token_bytes = tokens[0].cpu().tolist()
    input_chars = []
    for b in token_bytes:
        try:
            input_chars.append(chr(b) if 32 <= b < 127 else f"\\x{b:02x}")
        except:
            input_chars.append(f"\\x{b:02x}")
    
    # Build frames
    frames = []
    n_head = config.n_head
    n_neurons = config.n_neurons
    TOP_K = 15
    GRID_BINS = 64
    
    # Build ρ matrices (head-averaged, full T×T per layer) for global response
    rho_matrices = {}
    for layer_idx, layer_data in layer_data_cpu.items():
        attn_raw = layer_data['attention']
        if attn_raw is not None:
            # attn shape: (nh, T, T) — head-average to (T, T)
            rho_avg = attn_raw.numpy().mean(axis=0)
            rho_matrices[layer_idx] = np.round(rho_avg, 4).tolist()
    
    for layer_idx, layer_data in layer_data_cpu.items():
        x_sparse = layer_data['x_sparse'].numpy()
        y_sparse = layer_data['y_sparse'].numpy()
        x_pre_relu = layer_data['x_pre_relu'].numpy() if layer_data['x_pre_relu'] is not None else None
        y_pre_relu = layer_data['y_pre_relu'].numpy() if layer_data['y_pre_relu'] is not None else None
        attn = layer_data['attention'].numpy() if layer_data['attention'] is not None else None
        attn_out = layer_data['attention_output'].numpy() if layer_data['attention_output'] is not None else None
        
        x_total = n_head * n_neurons
        y_total = n_head * n_neurons
        
        for t in range(T):
            x_active = []
            y_active = []
            x_top_neurons = []
            y_top_neurons = []
            all_x_active = set()
            all_y_active = set()
            
            for h in range(n_head):
                x_h = x_sparse[h, t]
                y_h = y_sparse[h, t]
                
                x_nz = np.flatnonzero(x_h > 0)
                y_nz = np.flatnonzero(y_h > 0)
                
                for idx in x_nz:
                    all_x_active.add((h, int(idx)))
                for idx in y_nz:
                    all_y_active.add((h, int(idx)))
                
                # Top neurons per head
                if len(x_nz) > 0:
                    top_k = min(5, len(x_nz))
                    top_idx = x_nz[np.argsort(x_h[x_nz])[-top_k:]][::-1]
                    for idx in top_idx:
                        x_top_neurons.append({
                            "head": int(h),
                            "neuron": int(idx),
                            "value": round(float(x_h[idx]), 3)
                        })
                
                if len(y_nz) > 0:
                    top_k = min(5, len(y_nz))
                    top_idx = y_nz[np.argsort(y_h[y_nz])[-top_k:]][::-1]
                    for idx in top_idx:
                        y_top_neurons.append({
                            "head": int(h),
                            "neuron": int(idx),
                            "value": round(float(y_h[idx]), 3)
                        })
                
                # Limit for basic active list
                if len(x_nz) > TOP_K:
                    top = np.argpartition(x_h[x_nz], -TOP_K)[-TOP_K:]
                    x_nz = x_nz[top]
                if len(y_nz) > TOP_K:
                    top = np.argpartition(y_h[y_nz], -TOP_K)[-TOP_K:]
                    y_nz = y_nz[top]
                
                x_active.append({
                    "indices": x_nz.tolist(),
                    "values": np.round(x_h[x_nz], 3).tolist(),
                })
                y_active.append({
                    "indices": y_nz.tolist(),
                    "values": np.round(y_h[y_nz], 3).tolist(),
                })
            
            # Activation grids (downsampled per-head neuron activity)
            bin_size = n_neurons // GRID_BINS
            x_grid = []
            y_grid = []
            hadamard_grid_data = []
            for h in range(n_head):
                x_h_full = x_sparse[h, t]
                y_h_full = y_sparse[h, t]
                x_row = []
                y_row = []
                h_row = []
                for b in range(GRID_BINS):
                    s, e = b * bin_size, (b + 1) * bin_size
                    x_bin = x_h_full[s:e]
                    y_bin = y_h_full[s:e]
                    x_mx = float(x_bin.max())
                    y_mx = float(y_bin.max())
                    x_row.append(round(x_mx, 3) if x_mx > 0 else 0)
                    y_row.append(round(y_mx, 3) if y_mx > 0 else 0)
                    h_row.append(int(((x_bin > 0) & (y_bin > 0)).sum()))
                x_grid.append(x_row)
                y_grid.append(y_row)
                hadamard_grid_data.append(h_row)
            
            # Gating
            gated = all_x_active & all_y_active
            x_only = len(all_x_active - all_y_active)
            y_only = len(all_y_active - all_x_active)
            both = len(gated)
            
            # Sparsity
            x_active_count = len(all_x_active)
            y_active_count = len(all_y_active)
            x_sparsity = round(1 - (x_active_count / x_total), 4)
            y_sparsity = round(1 - (y_active_count / y_total), 4)
            
            # Pre-relu stats (with histogram for visualization)
            x_pre_stats = None
            if x_pre_relu is not None:
                x_pre_t = x_pre_relu[:, t, :]
                x_flat = x_pre_t.flatten()
                counts, edges = np.histogram(x_flat, bins=20)
                x_pre_stats = {
                    "mean": round(float(x_pre_t.mean()), 3),
                    "std": round(float(x_pre_t.std()), 3),
                    "max": round(float(x_pre_t.max()), 3),
                    "min": round(float(x_pre_t.min()), 3),
                    "positive_count": int((x_pre_t > 0).sum()),
                    "total": int(x_pre_t.size),
                    "histogram": [
                        {"start": round(float(edges[i]), 3), "end": round(float(edges[i+1]), 3), "count": int(counts[i])}
                        for i in range(len(counts))
                    ],
                }
            
            y_pre_stats = None
            if y_pre_relu is not None:
                y_pre_t = y_pre_relu[:, t, :]
                y_flat = y_pre_t.flatten()
                y_counts, y_edges = np.histogram(y_flat, bins=20)
                y_pre_stats = {
                    "mean": round(float(y_pre_t.mean()), 3),
                    "std": round(float(y_pre_t.std()), 3),
                    "max": round(float(y_pre_t.max()), 3),
                    "min": round(float(y_pre_t.min()), 3),
                    "positive_count": int((y_pre_t > 0).sum()),
                    "total": int(y_pre_t.size),
                    "histogram": [
                        {"start": round(float(y_edges[i]), 3), "end": round(float(y_edges[i+1]), 3), "count": int(y_counts[i])}
                        for i in range(len(y_counts))
                    ],
                }
            
            # Attention stats (with full weights for bar chart)
            attention_stats = None
            attention_weights_full = None
            if attn is not None and t > 0:
                attn_t = attn[:, t, :t+1]
                attn_avg = attn_t.mean(axis=0)
                top_attn_idx = np.argsort(attn_avg)[-3:][::-1]
                attention_stats = {
                    "top_attended": [
                        {"token_idx": int(idx), "char": input_chars[idx], "weight": round(float(attn_avg[idx]), 3)}
                        for idx in top_attn_idx if idx < len(input_chars)
                    ]
                }
                attention_weights_full = [
                    {"token_idx": int(idx), "char": input_chars[idx], "weight": round(float(attn_avg[idx]), 4)}
                    for idx in range(len(attn_avg)) if idx < len(input_chars)
                ]
            
            # Embedding info (layer 0, with downsampled vector for heatmap)
            embed_info = None
            if layer_idx == 0:
                embed_vec = embed_cpu[t].numpy()
                ds_size = 64
                group = max(1, len(embed_vec) // ds_size)
                vector_ds = [round(float(embed_vec[i*group:(i+1)*group].mean()), 4) for i in range(ds_size)]
                embed_info = {
                    "byte_value": token_bytes[t],
                    "norm": round(float(np.linalg.norm(embed_vec)), 3),
                    "mean": round(float(embed_vec.mean()), 3),
                    "std": round(float(embed_vec.std()), 3),
                    "vector_ds": vector_ds,
                }
            
            frame = {
                "token_idx": t,
                "token_byte": token_bytes[t],
                "token_char": input_chars[t],
                "layer": layer_idx,
                "x_active": x_active,
                "y_active": y_active,
                "x_sparsity": x_sparsity,
                "y_sparsity": y_sparsity,
                "x_active_count": x_active_count,
                "y_active_count": y_active_count,
                "x_top_neurons": sorted(x_top_neurons, key=lambda x: -x["value"])[:10],
                "y_top_neurons": sorted(y_top_neurons, key=lambda x: -x["value"])[:10],
                "gating": {
                    "x_only": x_only,
                    "y_only": y_only,
                    "both": both,
                    "survival_rate": round(both / max(x_active_count, 1), 3),
                },
                "x_activation_grid": x_grid,
                "y_activation_grid": y_grid,
                "hadamard_grid": hadamard_grid_data,
            }
            
            if x_pre_stats:
                frame["x_pre_relu"] = x_pre_stats
            if y_pre_stats:
                frame["y_pre_relu"] = y_pre_stats
            if attention_stats:
                frame["attention_stats"] = attention_stats
            if attention_weights_full:
                frame["attention_weights"] = attention_weights_full
            if embed_info:
                frame["embedding"] = embed_info
            
            # a* vector (attention output for current token, downsampled)
            if attn_out is not None:
                # attn_out shape: (nh, T, D) — head-average at position t
                a_star_vec = attn_out[:, t, :].mean(axis=0)  # (D,)
                ds_size = 64
                D = len(a_star_vec)
                group = max(1, D // ds_size)
                a_star_ds = [round(float(a_star_vec[i*group:(i+1)*group].mean()), 4) for i in range(ds_size)]
                a_star_norm = round(float(np.linalg.norm(a_star_vec)), 3)
                frame["a_star_ds"] = a_star_ds
                frame["a_star_norm"] = a_star_norm
            
            frames.append(frame)
    
    # Predictions
    probs = torch.softmax(logits_cpu, dim=-1).numpy()
    predictions = []
    for t in range(T):
        top_idx = np.argsort(probs[t])[-5:][::-1]
        token_preds = []
        for idx in top_idx:
            try:
                char = chr(idx) if 32 <= idx < 127 else f"\\x{idx:02x}"
            except:
                char = f"\\x{idx:02x}"
            token_preds.append({
                "byte": int(idx),
                "char": char,
                "prob": round(float(probs[t, idx]), 4),
            })
        predictions.append(token_preds)
    
    t3 = time.perf_counter()
    print(f"[PERF] setup={t1-t0:.3f}s inference={t2-t1:.3f}s frames={t3-t2:.3f}s total={t3-t0:.3f}s ({T}tok × {len(layer_data_cpu)}layers)")
    
    return ORJSONResponse({
        "input_text": request.text,
        "input_tokens": token_bytes,
        "input_chars": input_chars,
        "num_layers": config.n_layer,
        "num_heads": config.n_head,
        "neurons_per_head": config.n_neurons,
        "embedding_dim": config.n_embd,
        "total_neurons": config.n_head * config.n_neurons,
        "frames": frames,
        "predictions": predictions,
        "rho_matrices": rho_matrices,
    })


@router.post("/hebbian-track")
async def track_hebbian(request: HebbianTrackRequest, req: Request):
    """Track Hebbian learning dynamics token-by-token."""
    model_service = req.app.state.model_service
    
    try:
        model = model_service.get_or_load(request.model_name)
        config = model_service.get_config(request.model_name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model error: {e}")
    
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "training"))
    from bdh import ExtractionConfig
    
    tokens = torch.tensor(
        [list(request.text.encode('utf-8'))],
        dtype=torch.long,
        device=model_service.device
    )
    T = tokens.shape[1]
    
    extraction_config = ExtractionConfig(
        capture_sparse_activations=True,
        capture_attention_patterns=True,
    )
    
    hebbian_updates = []
    
    with torch.no_grad():
        with model.extraction_mode(extraction_config) as buffer:
            _, _ = model(tokens)
            
            for layer_idx in sorted(buffer.x_sparse.keys()):
                x = buffer.x_sparse[layer_idx][0]
                
                for t in range(1, T):
                    x_prev = x[:, t-1, :]
                    x_curr = x[:, t, :]
                    
                    for h in range(config.n_head):
                        prev_active = (x_prev[h] > 0.1).nonzero().squeeze(-1)
                        curr_active = (x_curr[h] > 0.1).nonzero().squeeze(-1)
                        
                        if prev_active.numel() > 0 and curr_active.numel() > 0:
                            pairs = []
                            for n1 in prev_active[:10].tolist():
                                for n2 in curr_active[:10].tolist():
                                    strength = x_prev[h, n1].item() * x_curr[h, n2].item()
                                    if strength > 0.01:
                                        pairs.append({
                                            "neuron_from": n1,
                                            "neuron_to": n2,
                                            "strength": round(strength, 4),
                                        })
                            
                            if pairs:
                                hebbian_updates.append({
                                    "token_idx": t,
                                    "layer": layer_idx,
                                    "head": h,
                                    "pairs": pairs[:20],
                                })
    
    return {"input_text": request.text, "num_tokens": T, "updates": hebbian_updates}


@router.get("/architecture-spec")
async def get_architecture_spec():
    """Get architecture specification for the interactive diagram."""
    return {
        "name": "BDH (Baby Dragon Hatchling)",
        "components": [
            {"id": "input", "type": "input", "label": "x_{l-1}"},
            {"id": "encoder_e", "type": "linear", "label": "Linear E", "formula": "x_latent = x @ E"},
            {"id": "relu_x", "type": "activation", "label": "ReLU", "formula": "x_sparse = ReLU(x_latent)"},
            {"id": "attention", "type": "attention", "label": "Linear Attention", "formula": "ρ += x^T v"},
            {"id": "encoder_v", "type": "linear", "label": "Linear E_v"},
            {"id": "relu_y", "type": "activation", "label": "ReLU"},
            {"id": "decoder", "type": "linear", "label": "Linear D"},
            {"id": "residual", "type": "operation", "label": "+"},
            {"id": "output", "type": "output", "label": "x_l"},
        ],
    }


@router.get("/color-scheme")
async def get_color_scheme():
    """Get color scheme for visualizations."""
    return {
        "encoder_path": {"fill": "#FFF3CD", "stroke": "#E6AC00"},
        "attention": {"fill": "#D4EDDA", "stroke": "#28A745"},
        "attention_state": {"fill": "#CCE5FF", "stroke": "#0066CC"},
        "activation_relu": {"fill": "#F8D7DA", "stroke": "#DC3545"},
    }
