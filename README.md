# NEAR: Neural Electromagnetic Array Response

## Running the model
### Simulations
```cmd
python -m train --load_data data.mat --save_data results.mat
```
### Real World Experiments
```cmd
python -m train --load_data data.mat --save_data results.mat --warmup_iter 10000 --adaption_iter 50000 --change_lr_iter 45000 --k_max 4 --lam 1
```
