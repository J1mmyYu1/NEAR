# NEAR: Neural Electromagnetic Array Response

## Running the model
The example command below runs with the default simulation configuration; replace it with the appropriate configuration file when testing other scenarios—such as different SNR regimes or target counts.
### Simulations
```cmd
python -m train --load_data data.mat --save_data results.mat
```
### Real World Experiments
The example below launches the script with a parameter set tuned for the commercial MIMO radar platform **IMAGEVK-74**. When working with a different radar, replace these values to match your own system’s SNR range, antenna geometry, carrier frequency, and target density.
```cmd
python -m train --load_data data.mat --save_data results.mat --warmup_iter 10000 --adaption_iter 50000 --change_lr_iter 45000 --k_max 4 --lam 1
```
