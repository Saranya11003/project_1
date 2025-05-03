from .model1 import YOLOModel

def load_model(model_name: str):
    if model_name == "model1":
        return YOLOModel()
    else:
        raise ValueError(f"Model {model_name} not found.")    