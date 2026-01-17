from models.load_model import _load_model_instance

_model_instance = None

def set_model(model_name:str): # gemma/gemma-3-270m -
    global _model_instance
    _model_instance = _load_model_instance(model_name)
    
def get_model(): 
    if _model_instance is None:
        raise RuntimeError(f"Model instance is not loaded. Call set_model(model_name) first.")
    return _model_instance
