"""
Path Cache - Save and load precomputed paths for deterministic 
algorithms

For algorithms like boustrophedon that produce the same path 
for the same map, we can cache results to avoid recomputation.
"""
import hashlib
import pickle
import json
from pathlib import Path
from typing import List, Tuple, Optional


class PathCache:
    """
    Cache system for deterministic algorithm paths
    
    Paths are stored in cache/ directory, keyed by:
    - Map hash (MD5 of grid content)
    - Algorithm name
    """
    
    def __init__(self, cache_dir: str = "cache"):
        """Initialize cache with specified directory"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Load cache metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def _save_metadata(self):
        """Save cache metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _compute_map_hash(self, map_file: str) -> str:
        """Compute hash of map file content"""
        with open(map_file, 'rb') as f:
            content = f.read()
            return hashlib.md5(content).hexdigest()
    
    def _get_cache_key(self, map_file: str, 
                       algorithm: str) -> str:
        """Generate cache key from map and algorithm"""
        map_hash = self._compute_map_hash(map_file)
        return f"{algorithm}_{map_hash}"
    
    def get_path(self, map_file: str, 
                 algorithm: str) -> Optional[List[Tuple[int, int]]]:
        """
        Retrieve cached path if available
        
        Args:
            map_file: Path to map file
            algorithm: Algorithm name
        
        Returns:
            Cached path or None if not found
        """
        key = self._get_cache_key(map_file, algorithm)
        cache_file = self.cache_dir / f"{key}.pkl"
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                path = pickle.load(f)
            print(f"✓ Loaded cached path for {algorithm} "
                  f"({len(path)} steps)")
            return path
        
        return None
    
    def save_path(self, map_file: str, algorithm: str,
                  path: List[Tuple[int, int]], 
                  metrics: dict = None):
        """
        Save path to cache
        
        Args:
            map_file: Path to map file
            algorithm: Algorithm name
            path: List of (x, y) positions
            metrics: Optional performance metrics
        """
        key = self._get_cache_key(map_file, algorithm)
        cache_file = self.cache_dir / f"{key}.pkl"
        
        # Save path
        with open(cache_file, 'wb') as f:
            pickle.dump(path, f)
        
        # Update metadata
        self.metadata[key] = {
            'map_file': map_file,
            'algorithm': algorithm,
            'steps': len(path),
            'metrics': metrics or {}
        }
        self._save_metadata()
        
        print(f"✓ Cached path for {algorithm} "
              f"({len(path)} steps)")
    
    def clear_cache(self, algorithm: str = None):
        """
        Clear cached paths
        
        Args:
            algorithm: If specified, only clear this algorithm.
                      Otherwise, clear all.
        """
        if algorithm:
            # Clear specific algorithm
            keys_to_remove = [k for k in self.metadata.keys() 
                             if k.startswith(f"{algorithm}_")]
            for key in keys_to_remove:
                cache_file = self.cache_dir / f"{key}.pkl"
                if cache_file.exists():
                    cache_file.unlink()
                del self.metadata[key]
            print(f"✓ Cleared cache for {algorithm}")
        else:
            # Clear all
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            self.metadata = {}
            print("✓ Cleared all cache")
        
        self._save_metadata()
    
    def list_cached(self):
        """List all cached paths"""
        if not self.metadata:
            print("Cache is empty")
            return
        
        print("Cached paths:")
        print("-" * 60)
        for key, info in self.metadata.items():
            print(f"  {info['algorithm']:15} "
                  f"{Path(info['map_file']).name:20} "
                  f"{info['steps']:5} steps")
