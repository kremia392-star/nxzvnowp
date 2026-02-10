#!/usr/bin/env python3
"""
BDH Model Merging Utility

Merges two separately trained BDH models into a single polyglot model.
This is a unique capability of BDH's scale-free architecture that
transformers cannot achieve.

The merge works by concatenating the neuron spaces:
- French model: neurons 0 to N-1
- Portuguese model: neurons N to 2N-1
- Merged model: neurons 0 to 2N-1

This preserves the specialized knowledge in each model while creating
a unified model that handles both domains.

Usage:
    python merge.py --model1 checkpoints/french/best.pt \
                    --model2 checkpoints/portuguese/best.pt \
                    --output checkpoints/merged_polyglot.pt
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
from bdh import BDH, BDHConfig, load_model


@dataclass
class MergeConfig:
    """Configuration for model merging."""
    model1_path: str
    model2_path: str
    output_path: str
    model1_name: str = "french"
    model2_name: str = "portuguese"
    
    # Merge strategy
    merge_embeddings: str = "average"  # average, first, second
    merge_lm_head: str = "average"  # average, first, second


def load_checkpoint(path: str) -> Tuple[Dict[str, torch.Tensor], BDHConfig]:
    """Load model checkpoint and extract config."""
    checkpoint = torch.load(path, map_location="cpu")
    
    if "config" in checkpoint:
        config = BDHConfig(**checkpoint["config"])
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
        # Infer config from shapes
        encoder_shape = state_dict["encoder"].shape
        config = BDHConfig(
            n_layer=8,  # Default
            n_embd=encoder_shape[1],
            n_head=encoder_shape[0],
            mlp_internal_dim_multiplier=(
                encoder_shape[2] * encoder_shape[0] // encoder_shape[1]
            ),
        )
    
    return state_dict, config


def verify_compatible(config1: BDHConfig, config2: BDHConfig) -> bool:
    """Verify two models can be merged."""
    checks = [
        ("n_layer", config1.n_layer == config2.n_layer),
        ("n_embd", config1.n_embd == config2.n_embd),
        ("n_head", config1.n_head == config2.n_head),
        ("mlp_multiplier", config1.mlp_internal_dim_multiplier == config2.mlp_internal_dim_multiplier),
        ("vocab_size", config1.vocab_size == config2.vocab_size),
    ]
    
    all_pass = True
    for name, passed in checks:
        if not passed:
            print(f"❌ Incompatible {name}: {getattr(config1, name)} vs {getattr(config2, name)}")
            all_pass = False
        else:
            print(f"✓ {name} matches")
    
    return all_pass


def merge_models(
    state1: Dict[str, torch.Tensor],
    state2: Dict[str, torch.Tensor],
    config1: BDHConfig,
    merge_config: MergeConfig
) -> Tuple[Dict[str, torch.Tensor], BDHConfig]:
    """
    Merge two BDH models by concatenating neuron spaces.
    
    The key insight from the BDH paper (Section 7.1):
    - Encoder weights map D -> N (embedding to neuron space)
    - Decoder weights map N*nh -> D (neuron space to embedding)
    - We concatenate along the N dimension
    
    After merging:
    - New N = 2 * old N
    - French neurons occupy positions 0 to N-1
    - Portuguese neurons occupy positions N to 2N-1
    """
    
    print("\n🔀 Merging models...")
    
    merged_state = {}
    
    nh = config1.n_head
    D = config1.n_embd
    N = config1.n_neurons
    
    # === ENCODER ===
    # Shape: (nh, D, N)
    # Concatenate along N dimension -> (nh, D, 2N)
    encoder1 = state1["encoder"]  # (nh, D, N)
    encoder2 = state2["encoder"]  # (nh, D, N)
    merged_state["encoder"] = torch.cat([encoder1, encoder2], dim=2)
    print(f"  encoder: {encoder1.shape} + {encoder2.shape} -> {merged_state['encoder'].shape}")
    
    # === ENCODER_V ===
    # Same as encoder
    encoder_v1 = state1["encoder_v"]
    encoder_v2 = state2["encoder_v"]
    merged_state["encoder_v"] = torch.cat([encoder_v1, encoder_v2], dim=2)
    print(f"  encoder_v: {encoder_v1.shape} + {encoder_v2.shape} -> {merged_state['encoder_v'].shape}")
    
    # === DECODER ===
    # Shape: (nh*N, D)
    # Concatenate along first dimension -> (2*nh*N, D)
    decoder1 = state1["decoder"]  # (nh*N, D)
    decoder2 = state2["decoder"]  # (nh*N, D)
    merged_state["decoder"] = torch.cat([decoder1, decoder2], dim=0)
    print(f"  decoder: {decoder1.shape} + {decoder2.shape} -> {merged_state['decoder'].shape}")
    
    # === EMBEDDINGS ===
    # Shape: (vocab_size, D)
    # Options: average, first, second
    embed1 = state1["embed.weight"]
    embed2 = state2["embed.weight"]
    
    if merge_config.merge_embeddings == "average":
        merged_state["embed.weight"] = (embed1 + embed2) / 2
    elif merge_config.merge_embeddings == "first":
        merged_state["embed.weight"] = embed1
    else:
        merged_state["embed.weight"] = embed2
    print(f"  embed: {merge_config.merge_embeddings} -> {merged_state['embed.weight'].shape}")
    
    # === LM_HEAD ===
    # Shape: (D, vocab_size)
    lm1 = state1["lm_head"]
    lm2 = state2["lm_head"]
    
    if merge_config.merge_lm_head == "average":
        merged_state["lm_head"] = (lm1 + lm2) / 2
    elif merge_config.merge_lm_head == "first":
        merged_state["lm_head"] = lm1
    else:
        merged_state["lm_head"] = lm2
    print(f"  lm_head: {merge_config.merge_lm_head} -> {merged_state['lm_head'].shape}")
    
    # === ATTENTION FREQS ===
    # RoPE frequencies need to be expanded for larger N
    # Shape: (1, 1, 1, N) -> (1, 1, 1, 2N)
    freqs1 = state1["attn.freqs"]
    freqs2 = state2["attn.freqs"]
    merged_state["attn.freqs"] = torch.cat([freqs1, freqs2], dim=3)
    print(f"  attn.freqs: {freqs1.shape} + {freqs2.shape} -> {merged_state['attn.freqs'].shape}")
    
    # === Create merged config ===
    merged_config = BDHConfig(
        n_layer=config1.n_layer,
        n_embd=config1.n_embd,
        n_head=config1.n_head,
        mlp_internal_dim_multiplier=config1.mlp_internal_dim_multiplier * 2,  # Doubled!
        dropout=config1.dropout,
        vocab_size=config1.vocab_size,
    )
    
    print(f"\n📊 Merged model:")
    print(f"  Original neurons per head: {N}")
    print(f"  Merged neurons per head: {2 * N}")
    print(f"  Total original neurons: {nh * N}")
    print(f"  Total merged neurons: {2 * nh * N}")
    
    return merged_state, merged_config


def create_heritage_map(config1: BDHConfig, merge_config: MergeConfig) -> Dict[str, Any]:
    """
    Create a map tracking which neurons came from which model.
    
    This is crucial for interpretability - we can trace any synapse
    in the merged model back to its origin.
    """
    N = config1.n_neurons
    nh = config1.n_head
    
    heritage = {
        "model1_name": merge_config.model1_name,
        "model2_name": merge_config.model2_name,
        "neurons_per_head_per_model": N,
        "total_neurons_per_model": N * nh,
        "neuron_ranges": {
            merge_config.model1_name: {
                "per_head": [0, N - 1],
                "description": f"Neurons 0-{N-1} in each head"
            },
            merge_config.model2_name: {
                "per_head": [N, 2 * N - 1],
                "description": f"Neurons {N}-{2*N-1} in each head"
            }
        },
        "how_to_identify": {
            "given_neuron_idx": "If idx < N: from model1, else: from model2",
            "original_idx": "If from model2: original_idx = idx - N"
        }
    }
    
    return heritage


def validate_merged_model(
    merged_state: Dict[str, torch.Tensor],
    merged_config: BDHConfig,
    device: str = "cpu"
) -> bool:
    """Validate the merged model works correctly."""
    print("\n🔍 Validating merged model...")
    
    try:
        # Create model with merged config
        model = BDH(merged_config)
        
        # Load merged weights
        model.load_state_dict(merged_state)
        model.to(device)
        model.eval()
        
        # Test forward pass
        test_input = torch.randint(0, 256, (1, 64), device=device)
        with torch.no_grad():
            logits, _ = model(test_input)
        
        assert logits.shape == (1, 64, 256), f"Unexpected output shape: {logits.shape}"
        
        # Test generation
        prompt = torch.tensor([[ord('H'), ord('e'), ord('l'), ord('l'), ord('o')]], device=device)
        with torch.no_grad():
            generated = model.generate(prompt, max_new_tokens=10, top_k=5)
        
        assert generated.shape[1] == 15, f"Generation failed: {generated.shape}"
        
        print("✅ Validation passed!")
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Merge two BDH models")
    parser.add_argument("--model1", required=True, help="Path to first model checkpoint")
    parser.add_argument("--model2", required=True, help="Path to second model checkpoint")
    parser.add_argument("--output", required=True, help="Output path for merged model")
    parser.add_argument("--name1", default="french", help="Name for first model")
    parser.add_argument("--name2", default="portuguese", help="Name for second model")
    parser.add_argument("--merge-embeddings", choices=["average", "first", "second"], default="average")
    parser.add_argument("--merge-lm-head", choices=["average", "first", "second"], default="average")
    parser.add_argument("--device", default="cpu")
    
    args = parser.parse_args()
    
    merge_config = MergeConfig(
        model1_path=args.model1,
        model2_path=args.model2,
        output_path=args.output,
        model1_name=args.name1,
        model2_name=args.name2,
        merge_embeddings=args.merge_embeddings,
        merge_lm_head=args.merge_lm_head,
    )
    
    print("=" * 60)
    print("🐉 BDH Model Merger")
    print("=" * 60)
    print(f"Model 1: {args.model1} ({args.name1})")
    print(f"Model 2: {args.model2} ({args.name2})")
    print(f"Output: {args.output}")
    print("=" * 60)
    
    # Load models
    print("\n📂 Loading models...")
    state1, config1 = load_checkpoint(args.model1)
    state2, config2 = load_checkpoint(args.model2)
    
    print(f"Model 1: {config1.n_layer}L, {config1.n_embd}D, {config1.n_head}H, N={config1.n_neurons}")
    print(f"Model 2: {config2.n_layer}L, {config2.n_embd}D, {config2.n_head}H, N={config2.n_neurons}")
    
    # Verify compatibility
    print("\n🔍 Checking compatibility...")
    if not verify_compatible(config1, config2):
        print("\n❌ Models are incompatible for merging!")
        return 1
    
    # Merge
    merged_state, merged_config = merge_models(state1, state2, config1, merge_config)
    
    # Create heritage map
    heritage = create_heritage_map(config1, merge_config)
    
    # Validate
    if not validate_merged_model(merged_state, merged_config, args.device):
        print("\n❌ Merged model validation failed!")
        return 1
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model_state_dict": merged_state,
        "config": asdict(merged_config),
        "heritage": heritage,
        "merge_config": asdict(merge_config),
        "source_models": {
            args.name1: args.model1,
            args.name2: args.model2,
        }
    }
    
    torch.save(checkpoint, output_path)
    print(f"\n💾 Saved merged model to: {output_path}")
    
    # Save heritage map as JSON for frontend
    heritage_path = output_path.with_suffix(".heritage.json")
    with open(heritage_path, "w") as f:
        json.dump(heritage, f, indent=2)
    print(f"📋 Saved heritage map to: {heritage_path}")
    
    print("\n" + "=" * 60)
    print("✅ Merge complete!")
    print("=" * 60)
    print(f"\nMerged model: {merged_config.n_neurons * 2} neurons per head")
    print(f"Heritage tracking: Neurons 0-{config1.n_neurons-1} = {args.name1}")
    print(f"                   Neurons {config1.n_neurons}-{config1.n_neurons*2-1} = {args.name2}")
    
    return 0


if __name__ == "__main__":
    exit(main())
