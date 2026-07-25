import torch.nn as nn

def _is_bn_param(n: str) -> bool:
    """Helper to identify Batch Normalization parameters."""
    return (n.startswith('bn1.') or
            '.bn1.' in n or
            '.bn2.' in n or
            '.downsample.1.' in n)

def mark_only_scale_as_trainable(model: nn.Module) -> None:
    """
    Stage II (Feature Recalibration):
    Trains personalized parameters (phi) and BN.
    Freezes the shared backbone.
    """
    for n, p in model.named_parameters():
        if 'phi' in n or _is_bn_param(n):
            p.requires_grad = True
        else:
            p.requires_grad = False

def mark_only_backbone_as_trainable(model: nn.Module) -> None:
    """
    Stage I (Structural Distillation):
    Trains the backbone (theta) and BN.
    Freezes personalized parameters (phi).
    """
    for n, p in model.named_parameters():
        if 'phi' in n:
            p.requires_grad = False
        else:
            p.requires_grad = True