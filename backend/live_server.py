#!/usr/bin/env python3
"""
BDH Live Inference Server

This is a simplified, single-file backend that:
1. Loads your trained .pt model
2. Accepts any text input via API
3. Returns real-time activation data for visualization

Run:
    python live_server.py --model checkpoints/french_specialist/checkpoint_best.pt

Then your frontend can POST to http://localhost:8000/api/inference
"""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json

# FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# PyTorch
import torch
import torch.nn.functional as F


# ============================================================================
# BDH MODEL (Minimal version for inference)
# ============================================================================

@dataclass
class BDHConfig:
    n_layer: int = 8
    n_embd: int = 256
    dropout: float = 0.0  # No dropout during inference
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256
    
    @property
    def n_neurons(self) -> int:
        return self.n_embd * self.mlp_internal_dim_multiplier // self.n_head
    
    @property
    def total_neurons(self) -> int:
        return self.n_neurons * self.n_head


def get_freqs(n: int, theta: float = 2**16) -> torch.Tensor:
    def quantize(t, q=2):
        return (t / q).floor() * q
    return (1.0 / (theta ** (quantize(torch.arange(0, n, 1, dtype=torch.float32)) / n)) / (2 * math.pi))


class Attention(torch.nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        N = config.n_neurons
        self.register_buffer("freqs", get_freqs(N).view(1, 1, 1, N))
    
    def forward(self, Q, K, V):
        _, _, T, _ = Q.size()
        r_phases = torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype).view(1, 1, -1, 1) * self.freqs
        phases = (r_phases % 1) * (2 * math.pi)
        v_rot = torch.stack((-Q[..., 1::2], Q[..., ::2]), dim=-1).view(*Q.size())
        QR = (Q * torch.cos(phases)).to(Q.dtype) + (v_rot * torch.sin(phases)).to(Q.dtype)
        scores = (QR @ QR.mT).tril(diagonal=-1)
        return scores @ V, scores


class BDH(torch.nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        nh, D, N = config.n_head, config.n_embd, config.n_neurons
        
        self.encoder = torch.nn.Parameter(torch.zeros((nh, D, N)))
        self.encoder_v = torch.nn.Parameter(torch.zeros((nh, D, N)))
        self.decoder = torch.nn.Parameter(torch.zeros((nh * N, D)))
        self.attn = Attention(config)
        self.ln = torch.nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = torch.nn.Embedding(config.vocab_size, D)
        self.lm_head = torch.nn.Parameter(torch.zeros((D, config.vocab_size)))
    
    def forward_with_extraction(self, idx: torch.Tensor) -> tuple:
        """Forward pass that captures all activations for visualization."""
        C = self.config
        B, T = idx.size()
        D, nh, N = C.n_embd, C.n_head, C.n_neurons
        
        extractions = {
            "x_sparse": [],
            "y_sparse": [],
            "attention_scores": [],
        }
        
        x = self.embed(idx).unsqueeze(1)  # (B, 1, T, D)
        x = self.ln(x)
        
        for layer_idx in range(C.n_layer):
            # Encode to neuron space
            x_latent = x @ self.encoder  # (B, 1, T, N) per head -> (B, nh, T, N)
            x_sparse = F.relu(x_latent)
            extractions["x_sparse"].append(x_sparse.detach().cpu())
            
            # Attention
            yKV, attn_scores = self.attn(Q=x_sparse, K=x_sparse, V=x)
            extractions["attention_scores"].append(attn_scores.detach().cpu())
            
            # Value path
            yKV = self.ln(yKV)
            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            extractions["y_sparse"].append(y_sparse.detach().cpu())
            
            # Gating and decode
            xy_sparse = x_sparse * y_sparse
            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            x = self.ln(x + self.ln(yMLP))
        
        logits = x.view(B, T, D) @ self.lm_head
        return logits, extractions
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = 5):
        for _ in range(max_new_tokens):
            logits, _ = self.forward_with_extraction(idx)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def load_model(checkpoint_path: str, device: str = "cpu") -> tuple:
    """Load a BDH checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "config" in checkpoint:
        config = BDHConfig(**checkpoint["config"])
        state_dict = checkpoint["model_state_dict"]
    else:
        # Infer from weights
        encoder_shape = checkpoint["encoder"].shape
        config = BDHConfig(
            n_layer=8,
            n_embd=encoder_shape[1],
            n_head=encoder_shape[0],
            mlp_internal_dim_multiplier=(encoder_shape[2] * encoder_shape[0] // encoder_shape[1]),
        )
        state_dict = checkpoint
    
    model = BDH(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Loaded: {config.n_layer}L, {config.n_embd}D, {config.n_head}H, N={config.n_neurons}")
    return model, config


# ============================================================================
# FASTAPI SERVER
# ============================================================================

app = FastAPI(
    title="BDH Live Inference",
    description="Real-time inference and visualization for BDH models"
)

# CORS - allow frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
MODEL = None
CONFIG = None
DEVICE = "cpu"


# Request/Response models
class InferenceRequest(BaseModel):
    text: str
    include_attention: bool = False


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 1.0


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    return {
        "service": "BDH Live Inference",
        "model_loaded": MODEL is not None,
        "device": DEVICE,
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "config": asdict(CONFIG) if CONFIG else None,
    }


@app.post("/api/inference/run")
def run_inference(request: InferenceRequest):
    """
    Main endpoint: Run any text through the model and get activations.
    
    This is what powers the live visualization!
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    text = request.text
    
    # Tokenize (byte-level)
    tokens = torch.tensor([list(text.encode('utf-8'))], dtype=torch.long, device=DEVICE)
    T = tokens.shape[1]
    
    if T > 1024:
        raise HTTPException(status_code=400, detail="Text too long (max 1024 bytes)")
    
    # Run inference with extraction
    with torch.no_grad():
        logits, extractions = MODEL.forward_with_extraction(tokens)
    
    # Build response
    frames = []
    sparsity_by_layer = []
    
    for layer_idx in range(CONFIG.n_layer):
        x_sparse = extractions["x_sparse"][layer_idx][0]  # (nh, T, N)
        y_sparse = extractions["y_sparse"][layer_idx][0]
        
        # Compute sparsity
        x_sparsity = 1 - ((x_sparse > 0).sum().item() / x_sparse.numel())
        y_sparsity = 1 - ((y_sparse > 0).sum().item() / y_sparse.numel())
        sparsity_by_layer.append((x_sparsity + y_sparsity) / 2)
        
        # Build frames for each token
        for t in range(T):
            token_byte = tokens[0, t].item()
            try:
                token_char = chr(token_byte) if 32 <= token_byte < 127 else f"\\x{token_byte:02x}"
            except:
                token_char = f"\\x{token_byte:02x}"
            
            x_t = x_sparse[:, t, :]  # (nh, N)
            y_t = y_sparse[:, t, :]
            
            # Only store non-zero activations (sparse!)
            x_active = []
            y_active = []
            
            for h in range(CONFIG.n_head):
                x_nz = (x_t[h] > 0).nonzero().squeeze(-1)
                y_nz = (y_t[h] > 0).nonzero().squeeze(-1)
                
                # Limit to top 100 for efficiency
                if x_nz.numel() > 100:
                    vals = x_t[h][x_nz]
                    top_idx = vals.argsort(descending=True)[:100]
                    x_nz = x_nz[top_idx]
                
                if y_nz.numel() > 100:
                    vals = y_t[h][y_nz]
                    top_idx = vals.argsort(descending=True)[:100]
                    y_nz = y_nz[top_idx]
                
                x_active.append({
                    "indices": x_nz.tolist() if x_nz.numel() > 0 else [],
                    "values": [round(v, 4) for v in x_t[h][x_nz].tolist()] if x_nz.numel() > 0 else [],
                })
                y_active.append({
                    "indices": y_nz.tolist() if y_nz.numel() > 0 else [],
                    "values": [round(v, 4) for v in y_t[h][y_nz].tolist()] if y_nz.numel() > 0 else [],
                })
            
            frames.append({
                "token_idx": t,
                "token_byte": token_byte,
                "token_char": token_char,
                "layer": layer_idx,
                "x_active": x_active,
                "y_active": y_active,
                "x_sparsity": round(1 - ((x_t > 0).sum().item() / x_t.numel()), 4),
                "y_sparsity": round(1 - ((y_t > 0).sum().item() / y_t.numel()), 4),
            })
    
    # Build character list
    input_chars = []
    for byte in tokens[0].cpu().tolist():
        try:
            char = chr(byte) if 32 <= byte < 127 else f"\\x{byte:02x}"
        except:
            char = f"\\x{byte:02x}"
        input_chars.append(char)
    
    return {
        "input_text": text,
        "input_tokens": tokens[0].cpu().tolist(),
        "input_chars": input_chars,
        "num_layers": CONFIG.n_layer,
        "num_heads": CONFIG.n_head,
        "neurons_per_head": CONFIG.n_neurons,
        "frames": frames,
        "overall_sparsity": round(sum(sparsity_by_layer) / len(sparsity_by_layer), 4),
        "sparsity_by_layer": [round(s, 4) for s in sparsity_by_layer],
    }


@app.post("/api/inference/generate")
def generate_text(request: GenerateRequest):
    """Generate text continuation from a prompt."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Tokenize
    tokens = torch.tensor([list(request.prompt.encode('utf-8'))], dtype=torch.long, device=DEVICE)
    
    # Generate
    with torch.no_grad():
        output = MODEL.generate(
            tokens,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
        )
    
    # Decode
    full_bytes = bytes(output[0].cpu().tolist())
    full_text = full_bytes.decode('utf-8', errors='backslashreplace')
    
    return {
        "prompt": request.prompt,
        "generated": full_text[len(request.prompt):],
        "full_text": full_text,
    }


@app.get("/api/model/info")
def model_info():
    """Get information about the loaded model."""
    if CONFIG is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "config": asdict(CONFIG),
        "total_neurons": CONFIG.total_neurons,
        "device": DEVICE,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    global MODEL, CONFIG, DEVICE
    
    parser = argparse.ArgumentParser(description="BDH Live Inference Server")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    
    args = parser.parse_args()
    
    DEVICE = args.device
    
    print("=" * 60)
    print("🐉 BDH Live Inference Server")
    print("=" * 60)
    
    # Load model
    MODEL, CONFIG = load_model(args.model, DEVICE)
    
    print(f"\nServer starting on http://{args.host}:{args.port}")
    print(f"API docs: http://{args.host}:{args.port}/docs")
    print("=" * 60)
    
    # Run server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
