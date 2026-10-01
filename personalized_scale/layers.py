import torch
import torch.nn as nn
import torch.nn.functional as F

class PersonalizedScaleLayer:
    def __init__(self):
        pass

class ChannelScale(nn.Module, PersonalizedScaleLayer):
    def __init__(self, num_channels: int):
        super(ChannelScale, self).__init__()
        PersonalizedScaleLayer.__init__(self)

        self.phi = nn.Parameter(torch.ones(num_channels))

    def reset_parameters(self):
        if hasattr(self, 'phi'):
            nn.init.ones_(self.phi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x * self.phi
        elif x.dim() == 4:
            return x * self.phi.view(1, -1, 1, 1)
        elif x.dim() == 3:
            return x * self.phi.view(1, -1, 1)
        else:
            try:
                return x * self.phi
            except RuntimeError as e:
                print(f"Error in ChannelScale: Input shape {x.shape} "
                      f"is not compatible with phi shape {self.phi.shape}.")
                raise e


class AffineScale(nn.Module, PersonalizedScaleLayer):

    def __init__(self, num_channels: int):
        super(AffineScale, self).__init__()
        PersonalizedScaleLayer.__init__(self)
        self.phi = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels)) 

    def reset_parameters(self):
        if hasattr(self, 'phi'):
            nn.init.ones_(self.phi)
        if hasattr(self, 'beta'):
            nn.init.zeros_(self.beta) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x * self.phi + self.beta
        elif x.dim() == 4:
            return x * self.phi.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
        elif x.dim() == 3:
            return x * self.phi.view(1, -1, 1) + self.beta.view(1, -1, 1)
        else:
            try:
                return x * self.phi + self.beta
            except RuntimeError as e:
                print(f"Error in AffineScale: Input shape {x.shape} "
                      f"is not compatible with parameter shape.")
                raise e
