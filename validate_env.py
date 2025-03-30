import torch
print(torch.rand(5,3))
print(f'CUDA available:{torch.cuda.is_available()}')