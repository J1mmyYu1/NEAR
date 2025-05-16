import torch.nn as nn
import torch.nn.functional as F

class INRLayer(nn.Module):
    """
    A basic FC layer followed by ReLU activation.

    Parameters
    ----------
    d_in : int
        Input feature dimension.
    d_out : int
        Output feature dimension (a.k.a. hidden width).
    bias : bool, default=True
        Whether to add a bias term in the linear layer.
    """
    def __init__(self, d_in, d_out, bias=True):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out, bias=bias)
    
    def forward(self, x):
        return F.relu(self.fc(x))

class INR(nn.Module):
    """
    Implicit Neural Representation (INR) network — an MLP that maps a
    coordinate vector to a signal value.
    
    Parameters
    ----------
    in_features : int
        Dimension of input coordinate vector.
    hidden : int, default=256
        Width of hidden layers.
    layers : int, default=3
        Total number of hidden layers (not counting the final linear layer).
    out_features : int, default=1
        Dimension of the network output.
    outermost_linear : bool, default=True
        If True, the last layer is purely linear (no ReLU).
    """
    def __init__(self, in_features, hidden=256, layers=3, out_features=1, outermost_linear=True):
        super().__init__()
        net = [INRLayer(in_features, hidden)]
        for _ in range(layers - 1):
            net.append(INRLayer(hidden, hidden))
        if outermost_linear:
            net.append(nn.Linear(hidden, out_features))
        else:
            net.append(INRLayer(hidden, out_features))
        self.net = nn.Sequential(*net)
    
    def forward(self, x):
        return self.net(x)
