import os
import argparse
from itertools import product

import numpy as np
import torch
import pickle
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from tqdm import tqdm

from surface_spring_system import SpringSystem
from map_utils import make_iterable, restrict_knn_distance_matrix
from get_X_transportation_geometry import get_airport_data


def reconstruct_path(predecessors: np.ndarray, start: int, end: int) -> list:
    """From a predecessors matrix, reconstruct path from start to end."""
    path = [end]
    current = end

    while current != start:
        current = predecessors[start, current]
        if current == -9999:
            return None  # No path
        path.append(current)

    path.reverse()
    return path


def run_spring_simulations(N: int, 
                          ks: list[int], 
                          dts: list[float], 
                          dampings: list[float], 
                          max_stepss: list[int], 
                          alphas: list[float], 
                          superspring_constants: list[float],
                          surface_strengths: list[float]=None, 
                          alignment_strengths: list[float]=None, 
                          neighbors_smoothings: list[int]=None):
    """
    Deform X so that Euclidean-induced distances on X reflect best geodesic times, through spring simulations, for a grid of parameters.
    N: number of points on X
    k: neighbor parameter for geodesic time - for all direct neighbors, geodesic distance is Euclidean distance
    damping: prevent too many oscillations
    max_steps: number of steps for each spring simulation
    alpha: magnitude of radial force
    superspring_constant: spring constant of each superspring
    surface_strength: magnitude of surface constraint forces
    alignment_strength: magnitude of normal alignment forces
    k_neighbors: number of neighbors for surface constraints
    """

    # make everything iterable, just in case
    ks = make_iterable(ks)
    dts = make_iterable(dts)
    dampings = make_iterable(dampings)
    max_stepss = make_iterable(max_stepss)
    alphas = make_iterable(alphas)
    surface_strengths = make_iterable(surface_strengths)
    alignment_strengths = make_iterable(alignment_strengths)
    neighbors_smoothings = make_iterable(neighbors_smoothings)

    # load the geometry of X
    X_geometry_filename = f"data2/X_geometry_{N}.pkl"
    if os.path.isfile(X_geometry_filename):
        with open(X_geometry_filename, "rb") as f:
            X_geometry = pickle.load(f) 
    else:
        raise FileNotFoundError(f"File not found: {X_geometry_filename}")

    X = X_geometry['X']
    D = X_geometry['geodesic_times']
    airport_matching = X_geometry['airport_matching']

    # load airport data
    airport_name_to_idx, airport_times = get_airport_data()

    # main loop
    for k in tqdm(ks, desc='k'):

        print('Creating neighbor distance matrix...')

        # neighbor matrix
        D_restrict = restrict_knn_distance_matrix(D, k)
        D_restrict_sym = np.minimum(D_restrict, D_restrict.T) # because not neighbour -> np.inf distance
        D_correct = np.where(np.isinf(D_restrict_sym), 0, D_restrict_sym)
        np.fill_diagonal(D_correct, 0)
        sparse_graph = csr_matrix(D_correct.astype(np.float32))

        print('Getting predecessors...')

        # get predecessor matrix
        _, predecessors = shortest_path(csgraph=sparse_graph, directed=False, return_predecessors=True)

        print('Constructing springs')

        ## CONSTRUCT SPRINGS
        # super springs
        major_airports = list(airport_matching.keys())
        
        super_springs = []
        for airport_a, airport_b in product(major_airports, major_airports):
            idx_a, idx_b = airport_matching[airport_a], airport_matching[airport_b]
            flight_time = airport_times[airport_name_to_idx[airport_a], airport_name_to_idx[airport_b]]
            if np.isfinite(flight_time) and airport_a != airport_b:
                super_springs.append((reconstruct_path(predecessors, idx_a, idx_b), flight_time))
        
        # 2-springs
        two_spring_mask = (D_restrict != 0) & np.isfinite(D_restrict)
        two_spring_indices = np.argwhere(two_spring_mask)
        two_springs = [([int(i), int(j)], float(D_restrict[i, j])) for (i, j) in two_spring_indices]
        
        # springs
        springs = two_springs + super_springs

        mean_rest_length = np.mean([spring[1] for spring in springs])

        for damping, dt, max_steps, alpha, superspring_constant, surface_strength, alignment_strength, neighbors_smoothing in tqdm(
            product(dampings, dts, max_stepss, alphas, superspring_constants, surface_strengths, alignment_strengths, neighbors_smoothings), 
            desc='Inner', leave=False):
            
            spring_kwargs = {
                'spring_constant': 1.0,
                'damping': damping,
                'mass': 1.0,
                'radial_force_strength': alpha * mean_rest_length ** 2,
                'superspring_constant': superspring_constant,
                'surface_constraint_strength': surface_strength,
                'alignment_strength': alignment_strength,
                'k_neighbors': neighbors_smoothing
            }
            
            prop = mean_rest_length * 2 / np.pi
            
            X_rad = np.radians(X)
            x_start = np.cos(X_rad[:,0]) * np.cos(X_rad[:,1])
            y_start = np.cos(X_rad[:,0]) * np.sin(X_rad[:,1])
            z_start = np.sin(X_rad[:,0])
            
            starting_pos = prop * np.vstack((x_start, y_start, z_start)).T

            spring_system = SpringSystem(points=starting_pos, springs=springs, **spring_kwargs, device=device)

            try:
                print('Running...')
                historic = spring_system.simulate_to_equilibrium(dt=dt, max_steps=max_steps, show_progress=True)
                X_transfo = spring_system.positions.cpu().numpy()

                parameters_str = f"N{N}_k{k}_damping{damping}_dt{dt}_max_steps{max_steps}_alpha{alpha}_superspring_constant{superspring_constant}_surface{surface_strength}_alignment{alignment_strength}_smoothing{neighbors_smoothing}"

                with open(f"data2/historics_{parameters_str}.pkl", "wb") as f:
                    pickle.dump(historic, f)

                with open(f"data2/X_transfo_{parameters_str}.pkl", "wb") as f:
                    pickle.dump(X_transfo, f)

            except Exception as e:
                print(f"Error for k {k}: {e}")


if __name__ == '__main__':
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser(description="Run spring simulation")
    parser.add_argument('--N', type=int, default=10_000, help='Number of points')
    parser.add_argument('--ks', nargs='+', type=int, default=[3], help='Number of neighbors')
    parser.add_argument('--dts', nargs='+', type=float, default=[1e-4], help='Timestep')
    parser.add_argument('--dampings', nargs='+', type=float, default=[5.0], help='Damping')
    parser.add_argument('--max_stepss', nargs='+', type=int, default=[50_000], help='Maximal number of steps')
    parser.add_argument('--alphas', nargs='+', type=float, default=[10.], help='Radial force magnitude')
    parser.add_argument('--superspring_constants', nargs='+', type=float, default=[5.], help='Superspring constants')
    parser.add_argument('--surface_strengths', nargs='+', type=float, default=[2.], help='Surface constraint strength')
    parser.add_argument('--alignment_strengths', nargs='+', type=float, default=[5.], help='Alignment strength')
    parser.add_argument('--neighbors_smoothings', nargs='+', type=int, default=[10], help='Number of neighbors for smoothing')
    parser.add_argument('--device', type=str, default=default_device)

    args = parser.parse_args()

    N = args.N
    ks = args.ks
    dts = args.dts
    dampings = args.dampings
    max_stepss = args.max_stepss
    alphas = args.alphas
    superspring_constants = args.superspring_constants
    surface_strengths = args.surface_strengths
    alignment_strengths = args.alignment_strengths
    neighbors_smoothings = args.neighbors_smoothings
    device = args.device

    run_spring_simulations(N, ks, dts, dampings, max_stepss, alphas, superspring_constants, surface_strengths, alignment_strengths, neighbors_smoothings)
