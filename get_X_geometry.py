import os
import argparse

import numpy as np
import torch
import pickle
from tqdm import tqdm


def get_X_geometry(N: int=10_000, device: str='cpu', batch_size: int=20):
    """Get X coordinates and corresponding geodesic time matrix."""

    # load X transportation geometry
    X_geometry_filename = f"data2/X_transportation_geometry_{N}.pkl"
    if os.path.isfile(X_geometry_filename):
        with open(X_geometry_filename, "rb") as f:
            X_geometry = pickle.load(f)
    else:
        raise FileNotFoundError(f"File not found: {X_geometry_filename}")

    X = X_geometry['X']
    t = X_geometry['t']
    geodesic_times_car_only = X_geometry['geodesic_times_car_only']
    geodesic_times_ferry_only = X_geometry['geodesic_times_ferry_only']
    geodesics = X_geometry['geodesics']
    airport_matching = X_geometry['airport_matching']

    # compute geodesic times between all points of X
    t_gpu = torch.tensor(t, device=device, dtype=torch.float16)
    g_gpu = torch.tensor(geodesics, device=device, dtype=torch.float16)

    times = np.zeros((N, N), dtype=np.float16)

    for i_start in tqdm(range(0, N, batch_size), desc='i'):
        i_end = min(i_start + batch_size, N)
        chunk_size = i_end - i_start
        chunk_result = torch.zeros((chunk_size, N), device=device, dtype=torch.float16)
        
        for j_start in tqdm(range(i_start, N, batch_size), desc='j', leave=False):
            j_end = min(j_start + batch_size, N)

            ti, tj = t_gpu[i_start:i_end], t_gpu[j_start:j_end]
            big_tij = ti.unsqueeze(0).unsqueeze(2) + tj.unsqueeze(1).unsqueeze(-1)

            chunk_result[:, j_start:j_end] = torch.amin(big_tij + g_gpu.unsqueeze(0).unsqueeze(0), axis=(2,3))

            ## older, less vectorized version

            # for local_i, global_i in enumerate(range(i_start, i_end)):
            #     ti = t_gpu[global_i]
                
            #     for global_j in range(j_start, j_end):
            #         if global_i != global_j:
            #             tj = t_gpu[global_j]
                        
            #             big_tij = ti.unsqueeze(1) + tj.unsqueeze(0)
            #             chunk_result[local_i, global_j] = (big_tij + g_gpu).min()
        
        times[i_start:i_end] = chunk_result.cpu().numpy()
        
        del chunk_result
        torch.cuda.empty_cache()

    # symmetrize times
    i_lower = np.tril_indices(N, -1)
    times[i_lower] = times.T[i_lower]

    # final geodesic times - is necessarily the shortest time
    geodesic_times = np.minimum(times, geodesic_times_car_only, geodesic_times_ferry_only)
    np.fill_diagonal(geodesic_times, 0)

    X_geometry = {
        'X': X,
        'airport_matching': airport_matching,
        'geodesic_times': geodesic_times
    }

    return X_geometry
    

if __name__ == '__main__':
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser(description="Compute geodesic times on X with GPU acceleration.")
    parser.add_argument('--N', type=int, default=10000)
    parser.add_argument('--device', type=str, default=default_device)
    parser.add_argument('--batch_size', type=int, default=2)

    args = parser.parse_args()

    N = args.N
    device = args.device
    batch_size = args.batch_size

    X_geometry = get_X_geometry(N, device, batch_size)

    with open(f"data2/X_geometry_{N}.pkl", "wb") as f:
        pickle.dump(X_geometry, f)
