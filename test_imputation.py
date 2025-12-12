import numpy as np
from prepareData import mask_op
B, N, F, T = 1, 16, 3, 10
np.random.seed(42)
data = np.random.rand(B, N, F, T).astype(np.float32)

mask_matrix = np.ones(N)
mask_matrix[:3] = 0

adj_matrix = np.eye(N)

methods = [
    'neighbor_mean', 'zero_fill', 'stgcn', 'gnn', 'chebyshev',
    'graphsage', 'attention', 'residual', 'adaptive', 'multiscale'
]

for method in methods:
    try:
        print(f"\nTesting imputation method: {method}")
        filled = mask_op(data.copy(), mask_matrix, adj_matrix, method=method)
        print(f"{method} imputed data shape: {filled.shape}")
    except Exception as e:
        print(f"{method} imputation failed: {e}") 