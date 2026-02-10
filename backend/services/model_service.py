"""
Model Service

Handles loading, caching, and managing BDH model checkpoints.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import threading

import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "training"))
from bdh import BDH, BDHConfig, ExtractionConfig, load_model


class ModelService:
    """
    Service for managing BDH model instances.
    
    Features:
    - Load/unload model checkpoints
    - Cache loaded models in memory
    - Thread-safe model access
    - Support for multiple models (French, Portuguese, Merged)
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_cached_models: int = 3
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self.max_cached_models = max_cached_models
        
        # Model cache
        self._models: Dict[str, BDH] = {}
        self._model_configs: Dict[str, BDHConfig] = {}
        self._model_heritage: Dict[str, Dict] = {}  # For merged models
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Track model load order for LRU eviction
        self._load_order: List[str] = []
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available model checkpoints."""
        available = []
        
        if not self.checkpoint_dir.exists():
            return available
        
        # Look for checkpoint directories and files
        for path in self.checkpoint_dir.iterdir():
            if path.is_dir():
                # Check for best.pt or latest.pt
                best = path / "checkpoint_best.pt"
                latest = path / "checkpoint_latest.pt"
                
                if best.exists() or latest.exists():
                    checkpoint = best if best.exists() else latest
                    available.append({
                        "name": path.name,
                        "path": str(checkpoint),
                        "type": "checkpoint_dir",
                    })
            elif path.suffix == ".pt":
                available.append({
                    "name": path.stem,
                    "path": str(path),
                    "type": "checkpoint_file",
                })
        
        return available
    
    def list_loaded_models(self) -> List[str]:
        """List currently loaded model names."""
        with self._lock:
            return list(self._models.keys())
    
    def is_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded."""
        with self._lock:
            return model_name in self._models
    
    def load_model(self, model_name: str, checkpoint_path: Optional[str] = None) -> BDH:
        """
        Load a model checkpoint.
        
        Args:
            model_name: Name to register the model under
            checkpoint_path: Path to checkpoint (auto-detected if not provided)
        
        Returns:
            Loaded BDH model
        """
        with self._lock:
            # Return cached if available
            if model_name in self._models:
                # Move to end of load order (LRU)
                self._load_order.remove(model_name)
                self._load_order.append(model_name)
                return self._models[model_name]
            
            # Find checkpoint path
            if checkpoint_path is None:
                # Try to find it
                model_dir = self.checkpoint_dir / model_name
                if model_dir.is_dir():
                    # Try various naming conventions
                    candidates = [
                        model_dir / "checkpoint_best.pt",
                        model_dir / "checkpoint_latest.pt",
                        model_dir / f"{model_name}_best.pt",
                        model_dir / f"{model_name}_model.pt",
                        model_dir / "best.pt",
                        model_dir / "model.pt",
                        model_dir / f"{model_name}.pt",
                    ]
                    checkpoint_path = None
                    for candidate in candidates:
                        if candidate.exists():
                            checkpoint_path = str(candidate)
                            break
                    
                    if checkpoint_path is None:
                        # Look for any .pt file
                        pt_files = list(model_dir.glob("*.pt"))
                        if pt_files:
                            # Prefer files with 'best' in name
                            best_files = [f for f in pt_files if 'best' in f.name.lower()]
                            if best_files:
                                checkpoint_path = str(best_files[0])
                            else:
                                checkpoint_path = str(pt_files[0])
                        else:
                            raise FileNotFoundError(f"No checkpoint found in {model_dir}")
                else:
                    checkpoint_path = str(self.checkpoint_dir / f"{model_name}.pt")
            
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            # Evict old models if needed
            while len(self._models) >= self.max_cached_models:
                self._evict_oldest()
            
            # Load checkpoint
            print(f"Loading model '{model_name}' from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Extract config
            if "config" in checkpoint:
                config = BDHConfig(**checkpoint["config"])
                state_dict = checkpoint["model_state_dict"]
            else:
                # Infer from state dict
                encoder_shape = checkpoint["encoder"].shape
                config = BDHConfig(
                    n_layer=8,
                    n_embd=encoder_shape[1],
                    n_head=encoder_shape[0],
                    mlp_internal_dim_multiplier=(
                        encoder_shape[2] * encoder_shape[0] // encoder_shape[1]
                    ),
                )
                state_dict = checkpoint
            
            # Handle state dict from torch.compile (has _orig_mod. prefix)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("_orig_mod."):
                    new_state_dict[k[len("_orig_mod."):]] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
            
            # Create model
            model = BDH(config)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            # Cache
            self._models[model_name] = model
            self._model_configs[model_name] = config
            self._load_order.append(model_name)
            
            # Store heritage info for merged models
            if "heritage" in checkpoint:
                self._model_heritage[model_name] = checkpoint["heritage"]
            
            print(f"Loaded: {config.n_layer}L, {config.n_embd}D, {config.n_head}H, "
                  f"N={config.n_neurons}")
            
            return model
    
    def _evict_oldest(self):
        """Evict the least recently used model."""
        if self._load_order:
            oldest = self._load_order.pop(0)
            if oldest in self._models:
                del self._models[oldest]
                del self._model_configs[oldest]
                if oldest in self._model_heritage:
                    del self._model_heritage[oldest]
                print(f"Evicted model: {oldest}")
    
    def unload_model(self, model_name: str):
        """Unload a specific model."""
        with self._lock:
            if model_name in self._models:
                del self._models[model_name]
                del self._model_configs[model_name]
                if model_name in self._model_heritage:
                    del self._model_heritage[model_name]
                self._load_order.remove(model_name)
                torch.cuda.empty_cache()
                print(f"Unloaded: {model_name}")
    
    def unload_all(self):
        """Unload all models."""
        with self._lock:
            self._models.clear()
            self._model_configs.clear()
            self._model_heritage.clear()
            self._load_order.clear()
            torch.cuda.empty_cache()
            print("Unloaded all models")
    
    def get_model(self, model_name: str) -> BDH:
        """Get a loaded model by name."""
        with self._lock:
            if model_name not in self._models:
                raise KeyError(f"Model not loaded: {model_name}")
            return self._models[model_name]
    
    def get_config(self, model_name: str) -> BDHConfig:
        """Get model configuration."""
        with self._lock:
            if model_name not in self._model_configs:
                raise KeyError(f"Model not loaded: {model_name}")
            return self._model_configs[model_name]
    
    def get_heritage(self, model_name: str) -> Optional[Dict]:
        """Get heritage info for merged models."""
        with self._lock:
            return self._model_heritage.get(model_name)
    
    def get_or_load(self, model_name: str) -> BDH:
        """Get model, loading if necessary."""
        if not self.is_loaded(model_name):
            self.load_model(model_name)
        return self.get_model(model_name)
