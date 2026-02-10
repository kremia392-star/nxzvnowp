"""
Analysis API Routes

Endpoints for interpretability analysis:
- Sparsity measurement
- Monosemanticity probing
- Comparative analysis
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
import torch
import numpy as np

router = APIRouter()


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class SparsityRequest(BaseModel):
    """Request for sparsity analysis."""
    texts: List[str] = Field(..., description="Texts to analyze")
    model_name: str = Field(default="french_specialist")


class SparsityResult(BaseModel):
    """Sparsity result for one text."""
    text: str
    overall_sparsity: float
    x_sparsity: float
    y_sparsity: float
    sparsity_by_layer: List[float]
    active_neuron_percentage: float


class SparsityResponse(BaseModel):
    """Response with sparsity analysis."""
    model_name: str
    num_texts: int
    results: List[SparsityResult]
    aggregate: Dict[str, float]


class ConceptProbeRequest(BaseModel):
    """Request for concept probing."""
    concept_name: str = Field(..., description="Name of concept to probe")
    examples: List[str] = Field(..., description="Example words/phrases")
    model_name: str = Field(default="french_specialist")


class SynapseInfo(BaseModel):
    """Information about a discovered synapse."""
    layer: int
    head: int
    neuron_idx: int
    selectivity: float
    mean_activation: float
    activation_type: str


class ConceptProbeResponse(BaseModel):
    """Response from concept probing."""
    concept: str
    num_examples: int
    top_synapses: List[SynapseInfo]
    overall_activation_rate: float


class CompareRequest(BaseModel):
    """Request for model comparison."""
    text: str
    model_names: List[str] = Field(..., description="Models to compare")


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post("/sparsity", response_model=SparsityResponse)
async def analyze_sparsity(request: SparsityRequest, req: Request):
    """
    Analyze sparsity for a set of texts.
    
    Returns detailed sparsity metrics showing BDH's ~95% sparse activations.
    """
    model_service = req.app.state.model_service
    
    try:
        model = model_service.get_or_load(request.model_name)
        config = model_service.get_config(request.model_name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model error: {e}")
    
    results = []
    
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "training"))
    from bdh import ExtractionConfig
    
    for text in request.texts:
        tokens = torch.tensor(
            [list(text.encode('utf-8'))],
            dtype=torch.long,
            device=model_service.device
        )
        
        extraction_config = ExtractionConfig(
            capture_sparse_activations=True,
            capture_attention_patterns=False,
        )
        
        with torch.no_grad():
            with model.extraction_mode(extraction_config) as buffer:
                _, _ = model(tokens)
                
                x_total = 0
                x_active = 0
                y_total = 0
                y_active = 0
                sparsity_by_layer = []
                
                for layer_idx in sorted(buffer.x_sparse.keys()):
                    x = buffer.x_sparse[layer_idx]
                    y = buffer.y_sparse[layer_idx]
                    
                    x_total += x.numel()
                    x_active += (x > 0).sum().item()
                    y_total += y.numel()
                    y_active += (y > 0).sum().item()
                    
                    layer_x_sparsity = 1 - ((x > 0).sum().item() / x.numel())
                    layer_y_sparsity = 1 - ((y > 0).sum().item() / y.numel())
                    sparsity_by_layer.append((layer_x_sparsity + layer_y_sparsity) / 2)
        
        x_sparsity = 1 - (x_active / x_total)
        y_sparsity = 1 - (y_active / y_total)
        overall = (x_sparsity + y_sparsity) / 2
        
        results.append(SparsityResult(
            text=text,
            overall_sparsity=overall,
            x_sparsity=x_sparsity,
            y_sparsity=y_sparsity,
            sparsity_by_layer=sparsity_by_layer,
            active_neuron_percentage=(1 - overall) * 100,
        ))
    
    # Aggregate stats
    all_sparsities = [r.overall_sparsity for r in results]
    aggregate = {
        "mean_sparsity": np.mean(all_sparsities),
        "std_sparsity": np.std(all_sparsities),
        "min_sparsity": min(all_sparsities),
        "max_sparsity": max(all_sparsities),
    }
    
    return SparsityResponse(
        model_name=request.model_name,
        num_texts=len(request.texts),
        results=results,
        aggregate=aggregate,
    )


@router.post("/probe-concept", response_model=ConceptProbeResponse)
async def probe_concept(request: ConceptProbeRequest, req: Request):
    """
    Probe for concept-specific synapses.
    
    Runs concept examples through the model and identifies synapses
    that consistently activate for the given concept.
    """
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
    
    # Collect activations for all examples
    all_activations = {}  # key -> list of activation arrays
    
    for text in request.examples:
        tokens = torch.tensor(
            [list(text.encode('utf-8'))],
            dtype=torch.long,
            device=model_service.device
        )
        
        extraction_config = ExtractionConfig(
            capture_sparse_activations=True,
            capture_attention_patterns=False,
        )
        
        with torch.no_grad():
            with model.extraction_mode(extraction_config) as buffer:
                _, _ = model(tokens)
                
                for layer_idx in buffer.x_sparse.keys():
                    x = buffer.x_sparse[layer_idx][0]  # (nh, T, N)
                    y = buffer.y_sparse[layer_idx][0]
                    
                    # Average across tokens
                    x_mean = x.mean(dim=1).cpu().numpy()  # (nh, N)
                    y_mean = y.mean(dim=1).cpu().numpy()
                    
                    for head in range(config.n_head):
                        key_x = f"L{layer_idx}_H{head}_x"
                        key_y = f"L{layer_idx}_H{head}_y"
                        
                        if key_x not in all_activations:
                            all_activations[key_x] = []
                            all_activations[key_y] = []
                        
                        all_activations[key_x].append(x_mean[head])
                        all_activations[key_y].append(y_mean[head])
    
    # Find most selective synapses
    top_synapses = []
    
    for key, act_list in all_activations.items():
        # Parse key
        parts = key.split("_")
        layer = int(parts[0][1:])
        head = int(parts[1][1:])
        act_type = parts[2]
        
        # Stack activations
        acts = np.stack(act_list)  # (num_examples, N)
        
        # Compute mean and std
        mean_act = acts.mean(axis=0)
        std_act = acts.std(axis=0) + 1e-6
        
        # Selectivity = mean / std (high mean, low variance = consistent)
        selectivity = mean_act / std_act
        
        # Find top neurons
        top_k = 5
        top_indices = np.argsort(selectivity)[-top_k:][::-1]
        
        for idx in top_indices:
            if mean_act[idx] > 0.01:  # Only if actually activating
                top_synapses.append(SynapseInfo(
                    layer=layer,
                    head=head,
                    neuron_idx=int(idx),
                    selectivity=float(selectivity[idx]),
                    mean_activation=float(mean_act[idx]),
                    activation_type=f"{act_type}_sparse",
                ))
    
    # Sort by selectivity
    top_synapses.sort(key=lambda s: s.selectivity, reverse=True)
    
    # Compute overall activation rate
    all_acts = np.concatenate([np.stack(v).flatten() for v in all_activations.values()])
    activation_rate = (all_acts > 0).mean()
    
    return ConceptProbeResponse(
        concept=request.concept_name,
        num_examples=len(request.examples),
        top_synapses=top_synapses[:20],
        overall_activation_rate=float(activation_rate),
    )


@router.post("/compare")
async def compare_models(request: CompareRequest, req: Request):
    """
    Compare activation patterns across multiple models.
    
    Useful for comparing French vs Portuguese vs Merged models.
    """
    model_service = req.app.state.model_service
    
    results = {}
    
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "training"))
    from bdh import ExtractionConfig
    
    for model_name in request.model_names:
        try:
            model = model_service.get_or_load(model_name)
            config = model_service.get_config(model_name)
        except Exception as e:
            results[model_name] = {"error": str(e)}
            continue
        
        tokens = torch.tensor(
            [list(request.text.encode('utf-8'))],
            dtype=torch.long,
            device=model_service.device
        )
        
        extraction_config = ExtractionConfig(
            capture_sparse_activations=True,
        )
        
        with torch.no_grad():
            with model.extraction_mode(extraction_config) as buffer:
                _, _ = model(tokens)
                
                stats = buffer.get_sparsity_stats()
                
                # Get active neuron indices (first layer)
                if 0 in buffer.x_sparse:
                    x = buffer.x_sparse[0][0]  # (nh, T, N)
                    active_per_head = []
                    for h in range(config.n_head):
                        active = (x[h].sum(dim=0) > 0).sum().item()
                        active_per_head.append(active)
                else:
                    active_per_head = []
                
                results[model_name] = {
                    "sparsity": stats,
                    "n_neurons": config.n_neurons,
                    "n_heads": config.n_head,
                    "active_neurons_per_head": active_per_head,
                    "heritage": model_service.get_heritage(model_name),
                }
    
    return {
        "text": request.text,
        "models": results,
    }


@router.get("/concept-categories")
async def get_concept_categories():
    """
    Get available concept categories for probing.
    """
    # Import from monosemanticity module
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "analysis"))
    
    try:
        from monosemanticity import CONCEPT_CATEGORIES
        return {
            "categories": {
                name: {
                    "description": info["description"],
                    "num_examples": len(info["examples"]),
                    "sample_examples": info["examples"][:5],
                }
                for name, info in CONCEPT_CATEGORIES.items()
            }
        }
    except ImportError:
        return {
            "categories": {
                "currencies": {"description": "Monetary units", "num_examples": 20},
                "countries": {"description": "Nation names", "num_examples": 30},
                "languages": {"description": "Language names", "num_examples": 15},
            }
        }
