import torch
import torch.nn as nn
import torch.nn.functional as F

class PersonalizedScaleLayer:
    def __init__(self):
        pass

class ChannelScale(nn.Module, PersonalizedScaleLayer):
    """
    Channel-wise scaling layer for personalized feature recalibration
    and logit boundary adjustment (phi parameters in HDFL).
    """
    def __init__(self, num_channels: int):
        super(ChannelScale, self).__init__()
        PersonalizedScaleLayer.__init__(self)
        # phi: The learnable personalized scaling vector
        self.phi = nn.Parameter(torch.ones(num_channels))

    def reset_parameters(self):
        if hasattr(self, 'phi'):
            nn.init.ones_(self.phi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies channel-wise scaling based on input dimensionality."""
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