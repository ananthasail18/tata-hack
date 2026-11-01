"""
Model Registry and Versioning Module.
Manages versioned model artifacts, metadata catalogs, champion/challenger designation,
and rollback capabilities for production deployments.
"""

from datetime import datetime, timezone
import json
import os
import shutil
from typing import Any, Dict, List, Optional
import joblib

class ModelRegistry:
    """
    Local filesystem-backed Model Registry with champion-challenger lifecycle tracking.
    """
    def __init__(self, registry_dir: str = "models", catalog_path: str = "models/model_catalog.json"):
        self.registry_dir = registry_dir
        self.catalog_path = catalog_path
        os.makedirs(self.registry_dir, exist_ok=True)
        self._init_catalog()

    def _init_catalog(self) -> None:
        if not os.path.exists(self.catalog_path):
            initial_catalog = {
                "champion_version": None,
                "models": []
            }
            with open(self.catalog_path, "w", encoding="utf-8") as f:
                json.dump(initial_catalog, f, indent=2)

    def _read_catalog(self) -> Dict[str, Any]:
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_catalog(self, catalog: Dict[str, Any]) -> None:
        with open(self.catalog_path, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2)

    def register_model(
        self,
        version: str,
        model_type: str,
        artifact_path: str,
        metrics: Dict[str, float],
        hyperparameters: Dict[str, Any],
        is_champion: bool = False
    ) -> Dict[str, Any]:
        """
        Register a new model version into the catalog and optionally designate it as Champion.
        """
        catalog = self._read_catalog()
        now_ts = datetime.now(timezone.utc).isoformat()
        
        # Target versioned path
        ext = os.path.splitext(artifact_path)[1]
        versioned_filename = f"{model_type}_{version}{ext}"
        versioned_path = os.path.join(self.registry_dir, versioned_filename)
        
        # Copy artifact to versioned registry location (case-insensitive check for Windows)
        if os.path.abspath(artifact_path) != os.path.abspath(versioned_path) and os.path.exists(artifact_path):
            shutil.copyfile(artifact_path, versioned_path)

        entry = {
            "version": version,
            "model_type": model_type,
            "artifact_path": versioned_path,
            "metrics": metrics,
            "hyperparameters": hyperparameters,
            "registered_at": now_ts,
            "status": "CHAMPION" if is_champion else "CANDIDATE"
        }

        # Demote previous champion if this one is designated champion
        if is_champion:
            for m in catalog["models"]:
                if m.get("status") == "CHAMPION":
                    m["status"] = "ARCHIVED"
            catalog["champion_version"] = version

            # Copy to champion_model.joblib for production runtime
            champ_target = os.path.join(self.registry_dir, f"champion_model{ext}")
            shutil.copyfile(versioned_path, champ_target)

        catalog["models"].append(entry)
        self._write_catalog(catalog)
        return entry

    def get_champion(self) -> Optional[Dict[str, Any]]:
        """Retrieve current production Champion model metadata."""
        catalog = self._read_catalog()
        champ_ver = catalog.get("champion_version")
        if not champ_ver:
            return None
        for m in catalog["models"]:
            if m["version"] == champ_ver:
                return m
        return None

    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        return self._read_catalog().get("models", [])

    def rollback_to_version(self, target_version: str) -> bool:
        """Rollback active production champion to a previously registered model version."""
        catalog = self._read_catalog()
        found = None
        for m in catalog["models"]:
            if m["version"] == target_version:
                found = m
                break
                
        if not found:
            raise ValueError(f"Version '{target_version}' not found in registry.")

        for m in catalog["models"]:
            if m.get("status") == "CHAMPION":
                m["status"] = "ARCHIVED"
            if m["version"] == target_version:
                m["status"] = "CHAMPION"

        catalog["champion_version"] = target_version
        
        # Re-link champion artifact
        ext = os.path.splitext(found["artifact_path"])[1]
        champ_target = os.path.join(self.registry_dir, f"champion_model{ext}")
        shutil.copyfile(found["artifact_path"], champ_target)
        
        self._write_catalog(catalog)
        return True
