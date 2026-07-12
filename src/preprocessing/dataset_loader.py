import logging
from src.preprocessing.compas_adapter import CompasAdapter
from src.preprocessing.adult_adapter import AdultAdapter

logger = logging.getLogger(__name__)

def get_dataset_adapter(dataset_name: str = None, filepath: str = None):
    """Factory function to load the appropriate dataset adapter based on name."""
    import configs.config as cfg
    name = (dataset_name or cfg.DATASET).lower().strip()
    path = filepath or cfg.DATASET_PATH
    
    logger.info(f"Retrieving dataset adapter for: {name} (path: {path})")
    
    if name == "compas":
        return CompasAdapter(filepath=path)
    elif name == "adult":
        return AdultAdapter(filepath=path)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
