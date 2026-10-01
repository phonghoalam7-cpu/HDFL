import torch.nn as nn

def _is_bn_param(n: str) -> bool:
    return (n.startswith('bn1.') or
            '.bn1.' in n or
            '.bn2.' in n or
            '.downsample.1.' in n)

def mark_only_scale_as_trainable(model: nn.Module) -> None:

    for n, p in model.named_parameters():

        if 'phi_feat' in n or 'phi_logit' in n or _is_bn_param(n):
            p.requires_grad = True
        else:
            p.requires_grad = False

def mark_only_backbone_as_trainable(model: nn.Module) -> None:

    for n, p in model.named_parameters():

        if 'phi_feat' in n or 'phi_logit' in n:
            p.requires_grad = False
        else:

            p.requires_grad = True
