import torch
from models.UnetModel import UNet3D

def LoadModel(model_config, modelPath=None):
    model = UNet3D(**model_config)
    if not modelPath is None:
        ckpt = torch.load(modelPath, map_location='cuda:0')
        model.load_state_dict(ckpt['state_dict'])
    return model
