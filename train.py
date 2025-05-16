"""
Example:
    python -m train --load_data data.mat --save_data results.mat
"""
import argparse, torch
import math
import numpy as np
from model import INR
from scipy.io import loadmat, savemat


def pos_embed(u1, u2, L, device=None, dtype=torch.float32):
    '''
    Generate 2-D sinusoidal positional encodings.
    Each spatial position (x, y) is mapped to a 4 * L-dimensional vector.
    
    Parameters
    ----------
    u1 : int
        Number of samples along the x-axis (columns).
    u2 : int
        Number of samples along the y-axis (rows).
    L : int
        Number of frequency bands.
    device : torch.device
        default: 'cuda'
    dtype : torch.dtype
        default: torch.float32
        
    Returns
    -------
    torch.Tensor
        Positional-encoding tensor of shape (u2, u1, 4 * L).
    '''
    
    # Normalization
    x = torch.linspace(0.0, 1.0, steps=u1, device=device, dtype=dtype)  # (u1,)
    y = torch.linspace(0.0, 1.0, steps=u2, device=device, dtype=dtype)  # (u2,)

    # Meshgrid, xx contains x-coordinates, yy contains y-coordinates
    yy, xx = torch.meshgrid(y, x, indexing='ij')                        # (u2, u1)
    
    # Frequency vector
    freq = (2.0 ** torch.arange(L, device=device, dtype=dtype)) * math.pi  # (L,)

    # Broadcast
    xx = xx.unsqueeze(-1) * freq      # (u2, u1, L)
    yy = yy.unsqueeze(-1) * freq      # (u2, u1, L)

    # Sinusoidal functions
    sin_x = torch.sin(xx)
    sin_y = torch.sin(yy)
    cos_x = torch.cos(xx)
    cos_y = torch.cos(yy)

    # Stack & flatten
    pe = torch.stack([sin_x, sin_y, cos_x, cos_y], dim=-1)              # (u2, u1, L, 4)
    pe = pe.flatten(-2)                                                # (u2, u1, 4*L)

    return pe

def low_rank_regularizer(output, coeff1, coeff2, u1, u2, k_max):
    '''
    Low-rank Hankel regularizer.
    Compute the residual loss of the predicted responses using the harmonic structure (coeff1 and coeff2).
    
    Parameters
    ----------
    output : torch.Tensor, shape (u2, u1)
        Network prediction.
    coeff1 : torch.Tensor, shape (k_max, 1)
        Coefficient vector for row-wise Hankel structure.
    coeff2 : torch.Tensor, shape (k_max, 1)
        Coefficient vector for column-wise Hankel structure.
    u1 : int
        Number of samples along the x-axis (columns).
    u2 : int
        Number of samples along the y-axis (rows).
    k_max : int
        Number of targets assumed.

    Returns
    -------
    float
        Residual loss.
    '''
    reg = 0
    
    # row-wise residual loss
    for i in range(u2):
        H = torch.cat([output[i, j:j+u1-k_max].unsqueeze(1) for j in range(k_max)], dim=1)  # (u1 - k_max, k_max)
        pred = output[i, k_max:u1].unsqueeze(1)  # (u1 - k_max,)
        reg += torch.abs(pred - H @ coeff1).sum()

    # column-wise residual loss
    for i in range(u1):
        H = torch.cat([output.T[i, j:j+u2-k_max].unsqueeze(1) for j in range(k_max)], dim=1)  # (u2 - k_max, k_max)
        pred = output.T[i, k_max:u2].unsqueeze(1)  # (u2 - k_max,)
        reg += torch.abs(pred - H @ coeff2).sum()
    
    return reg


def main():
    parser = argparse.ArgumentParser()
    # Path
    parser.add_argument('--load_data', required=True, help='MAT file containing Observation matrix')
    parser.add_argument('--save_data', required=True, help='Output MAT file path')
    # Sparse array index set
    parser.add_argument('--S1', type=list, default=[0,1,2,3,4,9,14,19], help='Sparse array geometry S1')
    parser.add_argument('--S2', type=list, default=[0,1,2,3,4,9,14,19], help='Sparse array geometry S2')
    # Training hyper-parameters
    parser.add_argument('--warmup_iter', type=int, default=5000)
    parser.add_argument('--adaption_iter', type=int, default=25000)
    parser.add_argument('--change_lr_iter', type=int, default=20000)
    parser.add_argument('--steps_til_summary', type=int, default=1000)
    parser.add_argument('--L', type=int, default=10)
    parser.add_argument('--k_max', type=int, default=2, help='Number of targets assumed')
    parser.add_argument('--lr_init', type=float, default=1e-4)
    parser.add_argument('--lr_adapt_net', type=float, default=1e-3)
    parser.add_argument('--lr_adapt_coeff', type=float, default=3e-3)
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--hidden', type=int, default=256, help='Hidden layer width')
    parser.add_argument('--layers', type=int, default=3, help='Number of hidden layers')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Load data
    X = torch.tensor(loadmat(args.load_data)['Observation']).to(args.device)  # (u2, u1)
    u2, u1 = X.shape
    
    # Positional Encoding
    model_input = pos_embed(u2, u1, args.L, args.device)  # (u2, u1, 4*L)
    
    # Network initialization
    # Two networks representing the real and imaginary part of the array response
    INR_init_real = INR(in_features=model_input.shape[-1],
                hidden=args.hidden,
                layers=args.layers).to(args.device)
    INR_init_imag = INR(in_features=model_input.shape[-1],
                hidden=args.hidden,
                layers=args.layers).to(args.device)
    
    # Warm-up stage 
    optimizer_init = torch.optim.Adam(
    [{'params': filter(lambda p: p.requires_grad, INR_init_real.parameters()), 'lr': args.lr_init},
    {'params': filter(lambda p: p.requires_grad, INR_init_imag.parameters()), 'lr': args.lr_init}],
    betas=(0.9, 0.999), weight_decay=1e-4)
    
    for iter in range(1, args.warmup_iter + 1):
        model_output = torch.complex(INR_init_real(model_input), INR_init_imag(model_input)).reshape(u2, u1)
        loss_init = torch.abs(model_output[args.S2, :][:, args.S1]-X[args.S2, :][:, args.S1]).sum()  # reconstruction loss
        optimizer_init.zero_grad()
        loss_init.backward()
        optimizer_init.step()
        if not iter % args.steps_til_summary:
            print(f'Epoch {iter}/{args.warmup_iter} | Loss: {loss_init.item()}')
    
    # Results using pure INR
    pred_INR = model_output.detach().numpy()
    
    # adaption stage
    coeff_init_h = torch.zeros((args.k_max,1), dtype=torch.complex64)  # coeff1 initialization
    coeff_init_v = torch.zeros((args.k_max,1), dtype=torch.complex64)  # coeff2 initialization

    coeff_1 = coeff_init_h.to(args.device)
    coeff_1 = coeff_1.clone().detach().requires_grad_()

    coeff_2 = coeff_init_v.to(args.device)
    coeff_2 = coeff_2.clone().detach().requires_grad_()
    
    # Use the warm-up network weights as initialization for adaption stage
    INR_adapt_real = INR(in_features=model_input.shape[-1],
                hidden=args.hidden,
                layers=args.layers).to(args.device)
    INR_adapt_imag = INR(in_features=model_input.shape[-1],
                hidden=args.hidden,
                layers=args.layers).to(args.device)
    
    INR_adapt_real.load_state_dict(INR_init_real.state_dict())
    INR_adapt_imag.load_state_dict(INR_init_imag.state_dict())
    
    # Optimizer for network weights and coeff1 & coeff2
    optimizer_ls = torch.optim.Adam(
        [{'params': filter(lambda p: p.requires_grad, INR_adapt_real.parameters()), 'lr': args.lr_adapt_net},
        {'params': filter(lambda p: p.requires_grad, INR_adapt_imag.parameters()), 'lr': args.lr_adapt_net},
        {'params': [coeff_1], 'lr': args.lr_adapt_coeff},
        {'params': [coeff_2], 'lr': args.lr_adapt_coeff}],
        betas=(0.9, 0.999), weight_decay=1e-4)
    
    # Change the learning rate after a few steps
    def lr_lambda(step):
        if step < args.change_lr_iter:
            return 1.0
        else:
            return 0.1
    
    scheduler_ls = torch.optim.lr_scheduler.LambdaLR(optimizer_ls, lr_lambda, last_epoch=-1, verbose=True)
    
    for iter in range(1, args.adaption_iter + 1):
        model_output = torch.complex(INR_adapt_real(model_input), INR_adapt_imag(model_input)).reshape(u2, u1)
        loss = torch.abs(model_output[args.S2, :][:, args.S1]-X[args.S2, :][:, args.S1]).sum()  # reconstruction loss
        loss += args.lam * low_rank_regularizer(model_output, coeff_1, coeff_2, u1, u2, args.k_max)  # total loss
        
        optimizer_ls.zero_grad()
        loss.backward()
        optimizer_ls.step()
        scheduler_ls.step()
        
        if not iter % args.steps_til_summary:
            print(f'Epoch {iter}/{args.adaption_iter} | Loss: {loss.item()}')
    
    # Results using INR + Regularizer 
    pred_INR_reg = model_output.detach().numpy()
    
    prediction = {
        'INR_with_reg': pred_INR_reg
    }
    
    savemat(args.save_data,prediction)
    
if __name__ == '__main__':
    main()
