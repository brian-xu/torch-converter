import io

import torch
from torch import nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def byte_convert(model: dict) -> dict:
    new_model = dict()
    for key in model:
        if type(key) == bytes:
            new_key = key.decode("utf-8")
        else:
            new_key = key
        new_model[new_key] = dict()
        if isinstance(model[key], dict):
            new_model[new_key] = byte_convert(model[key])
        else:
            new_model[new_key] = model[key]
    return new_model


def load_model(model_path: str) -> dict:
    return torch.load(model_path, encoding='bytes', map_location=device)


def load_model2(model_path: str) -> nn.Module:
    with open(model_path, 'rb') as f:
        buffer = io.BytesIO(f.read())
        return torch.load(buffer, map_location=device)


def save_model(architecture: nn.Module, state_dict: dict) -> None:
    model = architecture().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    torch.save(model, architecture.__name__ + '.pth')
