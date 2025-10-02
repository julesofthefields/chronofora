import numpy as np
import torch
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


class SpringSystem:
    def __init__(self, points, springs, spring_constant=1.0, damping=0.1, mass=1.0, 
                 radial_force_strength=0.0, superspring_constant=5.0, 
                 surface_constraint_strength=None, alignment_strength=None, k_neighbors=None, 
                 device='cpu'):
        """
        Initialize a 3D spring system with surface constraints.

        Args:
            points: np.array of shape (n_points, 3) - initial positions
            springs: list of tuples ([list of indices], rest_length) - connections between points
            spring_constant: float - spring stiffness
            damping: float - damping coefficient
            mass: float - mass of each point
            radial_force_strength: float - strength of outward radial force from center of mass
            superspring_constant: stiffness (importance) of supersprings (they refer to long-scale flights)
            surface_constraint_strength: float - strength of surface constraint forces
            alignment_strength: float - strength of normal alignment forces
            k_neighbors: int - number of neighbors for surface constraints
            device: str - 'cpu' or 'cuda'
        """
        self.device = device
        self.positions = torch.tensor(points, dtype=torch.float32, device=device)
        self.velocities = torch.zeros_like(self.positions)
        self.springs = springs
        self.k = spring_constant
        self.super_k = superspring_constant
        self.damping = damping
        self.mass = mass
        self.n_points = len(points)
        self.radial_force_strength = radial_force_strength
        self.surface_constraint_strength = surface_constraint_strength
        self.alignment_strength = alignment_strength
        self.k_neighbors = k_neighbors
        self.dim = self.positions.shape[1]

        self.add_constraints = all(x is not None for x in [self.surface_constraint_strength, self.alignment_strength, self.k_neighbors])

        # build surface connectivity for constraints
        self.build_surface_connectivity(points)

        # process supersprings (original code)
        i_list, j_list, rest_lengths, spring_sizes = [], [], [], []
        
        for (index_sequence, rest_length) in self.springs:
            i_list.extend(index_sequence[:-1])
            j_list.extend(index_sequence[1:])
            rest_lengths.append(rest_length)
            spring_sizes.append(len(index_sequence) - 1)

        self.i = torch.tensor(i_list, dtype=torch.long, device=device)
        self.j = torch.tensor(j_list, dtype=torch.long, device=device)
        self.rest_lengths = torch.tensor(rest_lengths, dtype=torch.float32, device=device)
        spring_sizes = torch.tensor(spring_sizes, dtype=torch.long, device=device)

        self.k_per_spring = torch.full((len(self.springs),), self.k, dtype=torch.long, device=device)
        self.k_per_spring[spring_sizes > 1] = self.super_k

        # create mapping from individual spring segments to supersprings (r2d matrix)
        total_links = spring_sizes.sum().item()
        num_supersprings = spring_sizes.shape[0]
        
        # build r2d sparse matrix and spring constants
        r2d_indices = []
        r2d_values = []
        start = 0
        for idx, size in enumerate(spring_sizes):
            # add indices for this superspring
            for j in range(start, start + size):
                r2d_indices.append([idx, j])
                r2d_values.append(1.0)
            start += size
        
        r2d_indices = torch.tensor(r2d_indices, device=device).T
        r2d_values = torch.tensor(r2d_values, dtype=torch.float32, device=device)
        self.r2d = torch.sparse_coo_tensor(r2d_indices, r2d_values, 
                                          size=(num_supersprings, total_links), 
                                          device=device).coalesce()

        # create mapping from spring forces to point forces
        force_indices = []
        force_values = []
        
        for k, (i_idx, j_idx) in enumerate(zip(self.i.cpu().numpy(), self.j.cpu().numpy())):
            force_indices.extend([[i_idx, k], [j_idx, k]])
            force_values.extend([-1.0, 1.0])
        
        force_indices = torch.tensor(force_indices, device=device).T
        force_values = torch.tensor(force_values, dtype=torch.float32, device=device)
        self.force_to_pos = torch.sparse_coo_tensor(force_indices, force_values,
                                                   size=(self.n_points, total_links),
                                                   device=device).coalesce()

    def build_surface_connectivity(self, points: np.ndarray):
        """Build connectivity graph for surface constraints"""
        if self.add_constraints:
            # sklearn is more efficient for neighbor finding here...
            nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1).fit(points)
            distances, indices = nbrs.kneighbors(points)
            
            # convert to tensors and move to device
            self.neighbors = torch.tensor(indices[:, 1:], dtype=torch.long, device=self.device)  # Exclude self
            self.neighbor_distances = torch.tensor(distances[:, 1:], dtype=torch.float32, device=self.device)

    def compute_center_of_mass(self) -> np.ndarray:
        """Compute the center of mass of the system."""
        return self.positions.mean(dim=0)

    def compute_radial_forces(self, tol: float=1e-9) -> np.ndarray:
        """Compute radial forces from center of mass."""
        if self.radial_force_strength == 0:
            return torch.zeros_like(self.positions)
        
        center = self.compute_center_of_mass()
        radial_vectors = self.positions - center
        distances = torch.norm(radial_vectors, dim=1, keepdim=True)
        
        # more robust handling of near-zero distances
        safe_distances = torch.clamp(distances, min=tol)
        radial_forces = self.radial_force_strength * radial_vectors / safe_distances.pow(3)
        return radial_forces

    def compute_spring_forces(self, tol: float=1e-9) -> np.ndarray:
        """Compute spring forces acting on each point."""
        # vector between connected points
        r = self.positions[self.i] - self.positions[self.j]
        distances = torch.norm(r, dim=1)
        directions = r / (distances.unsqueeze(1) + tol)
        
        # calculate force magnitudes using superspring logic
        superspring_lengths = torch.sparse.mm(self.r2d, distances.unsqueeze(1)).squeeze(1)
        extensions = superspring_lengths - self.rest_lengths
        force_magnitudes = torch.sparse.mm(self.r2d.T, (self.k_per_spring * extensions).unsqueeze(1)).squeeze(1)
        
        # apply forces along directions
        forces = force_magnitudes.unsqueeze(1) * directions
        
        # map spring forces to point forces
        spring_forces = torch.sparse.mm(self.force_to_pos, forces)
        return spring_forces

    def compute_current_normals(self) -> np.ndarray:
        """Compute current normal vectors for all points (vectorized)"""
        if self.add_constraints:
            # get all neighbor points at once: [n_points, k_neighbors, 3]
            neighbor_points = self.positions[self.neighbors]  # Broadcasting magic!
            
            # center around each point: [n_points, k_neighbors, 3]
            centered_neighbors = neighbor_points - self.positions.unsqueeze(1)
            
            # compute covariance matrices for all points at once: [n_points, 3, 3]
            cov_matrices = torch.bmm(centered_neighbors.transpose(1, 2), centered_neighbors) / self.k_neighbors
            
            # batch eigendecomposition: [n_points, 3], [n_points, 3, 3]
            eigenvals, eigenvecs = torch.linalg.eigh(cov_matrices)
            
            # get normal (smallest eigenvalue): [n_points, 3]
            min_eigenval_indices = torch.argmin(eigenvals, dim=1)
            normals = eigenvecs[torch.arange(self.n_points), :, min_eigenval_indices]
            
            return normals

    def compute_surface_constraint_forces(self) -> np.ndarray:
        """Forces to keep points on their local tangent planes (vectorized)"""
        if self.add_constraints:
            if self.surface_constraint_strength == 0:
                return torch.zeros_like(self.positions)
            
            # get all neighbor points: [n_points, k_neighbors, 3]
            neighbor_points = self.positions[self.neighbors]
            
            # center around each point: [n_points, k_neighbors, 3]
            centered_neighbors = neighbor_points - self.positions.unsqueeze(1)
            
            # compute covariance matrices: [n_points, 3, 3]
            cov_matrices = torch.bmm(centered_neighbors.transpose(1, 2), centered_neighbors) / self.k_neighbors
            
            # batch eigendecomposition: [n_points, 3], [n_points, 3, 3]
            eigenvals, eigenvecs = torch.linalg.eigh(cov_matrices)
            
            # get normals (smallest eigenvalue): [n_points, 3]
            min_eigenval_indices = torch.argmin(eigenvals, dim=1)
            current_normals = eigenvecs[torch.arange(self.n_points), :, min_eigenval_indices]
            
            # displacement from neighbor centroids: [n_points, 3]
            neighbor_centroids = torch.mean(neighbor_points, dim=1)
            displacement_from_neighbors = self.positions - neighbor_centroids
            
            # out-of-plane components: [n_points]
            out_of_plane_magnitudes = torch.sum(displacement_from_neighbors * current_normals, dim=1)
            
            # out-of-plane vectors: [n_points, 3]
            out_of_plane_components = out_of_plane_magnitudes.unsqueeze(1) * current_normals
            
            # restoring forces
            forces = -self.surface_constraint_strength * out_of_plane_components
            
            return forces
        
        else:
            return torch.tensor(0.0, device=self.device)

    def compute_normal_alignment_forces(self, tol: float=1e-9) -> np.ndarray:
        """Forces to keep local normals smooth across the surface (vectorized)"""
        if self.add_constraints:
            if self.alignment_strength == 0:
                return torch.zeros_like(self.positions)
            
            # get current normals for all points: [n_points, 3]
            current_normals = self.compute_current_normals()
            
            # get neighbor normals: [n_points, k_neighbors, 3]
            neighbor_normals = current_normals[self.neighbors]
            
            # average neighbor normals: [n_points, 3]
            avg_neighbor_normals = torch.mean(neighbor_normals, dim=1)
            
            # normalize
            norms = torch.norm(avg_neighbor_normals, dim=1, keepdim=True)
            avg_neighbor_normals = avg_neighbor_normals / torch.clamp(norms, min=tol)
            
            # normal differences: [n_points, 3]
            normal_differences = avg_neighbor_normals - current_normals
            
            # convert to position forces
            forces = self.alignment_strength * normal_differences * 0.1
            
            return forces
        
        else:
            return torch.tensor(0.0, device=self.device)

    def compute_neighbor_spring_forces(self, tol: float=1e-9) -> np.ndarray:
        """Additional spring forces between surface neighbors (vectorized)"""
        if self.add_constraints:
            # get all neighbor positions: [n_points, k_neighbors, 3]
            neighbor_positions = self.positions[self.neighbors]
            
            # displacement vectors: [n_points, k_neighbors, 3]
            displacements = neighbor_positions - self.positions.unsqueeze(1)
            
            # current distances: [n_points, k_neighbors]
            current_distances = torch.norm(displacements, dim=2)
            
            # force magnitudes: [n_points, k_neighbors]
            force_magnitudes = 0.5 * (current_distances - self.neighbor_distances)
            
            # force directions (normalized displacements): [n_points, k_neighbors, 3]
            safe_distances = torch.clamp(current_distances, min=tol)
            force_directions = displacements / safe_distances.unsqueeze(2)
            
            # force vectors: [n_points, k_neighbors, 3]
            force_vectors = force_magnitudes.unsqueeze(2) * force_directions
            
            # sum forces from all neighbors: [n_points, 3]
            forces = torch.sum(force_vectors, dim=1)
            
            return forces
        
        else:
            return torch.tensor(0.0, device=self.device)

    def compute_forces(self, tol: float=1e-9) -> np.ndarray:
        """Compute total forces acting on each point."""
        # sum all forces
        spring_forces = self.compute_spring_forces(tol)
        radial_forces = self.compute_radial_forces(tol)
        surface_forces = self.compute_surface_constraint_forces()
        alignment_forces = self.compute_normal_alignment_forces(tol)
        neighbor_spring_forces = self.compute_neighbor_spring_forces(tol)
        damping_forces = -self.damping * self.velocities
        
        return (spring_forces + radial_forces + surface_forces + 
                alignment_forces + neighbor_spring_forces + damping_forces)

    def step(self, dt: float):
        """Perform one simulation step using Verlet integration."""
        positions_cpu = self.positions.cpu().numpy()
        self.build_surface_connectivity(positions_cpu)
        
        forces = self.compute_forces()
        accelerations = forces / self.mass

        # update positions and velocities
        self.positions += self.velocities * dt + 0.5 * accelerations * dt ** 2
        self.velocities += accelerations * dt

    def get_spring_energy(self) -> float:
        """Calculate total spring potential energy."""
        r = self.positions[self.i] - self.positions[self.j]
        distances = torch.norm(r, dim=1)
        superspring_lengths = torch.sparse.mm(self.r2d, distances.unsqueeze(1)).squeeze(1)
        extensions = superspring_lengths - self.rest_lengths
        energy = 0.5 * (self.k_per_spring * extensions ** 2).sum()
        return energy

    def get_radial_energy(self, tol: float=1e-9) -> float:
        """Calculate radial potential energy."""
        if self.radial_force_strength == 0:
            return torch.tensor(0.0, device=self.device)
        
        center = self.compute_center_of_mass()
        distances = torch.norm(self.positions - center, dim=1)
        safe_distances = torch.clamp(distances, min=tol)
        energy = -0.5 * self.radial_force_strength * (1 / safe_distances).sum()
        return energy

    def get_total_energy(self) -> float:
        """Calculate total system energy."""
        kinetic = 0.5 * self.mass * (self.velocities ** 2).sum()
        spring = self.get_spring_energy()
        radial = self.get_radial_energy()
        return kinetic + spring + radial

    def simulate_to_equilibrium(self, dt: float=0.01, max_steps: int=10000, tolerance: float=1e-6, show_progress: bool=True) -> list:
        """Run simulation until equilibrium is reached."""
        spring_energy_history = []

        for step in tqdm(range(max_steps), disable=not show_progress):
            self.step(dt)

            spring_energy = self.get_spring_energy()
            spring_energy_history.append(spring_energy.item())

            if spring_energy < tolerance:
                print(f"Equilibrium reached at step {step}")
                break
        else:
            print(f"Max steps ({max_steps}) reached")

        return spring_energy_history


if __name__ == '__main__':
    points = np.array([
        [0, 0, 0],
        [0, 4, 0],
        [0, 0, 5],
        [1, 1, 3]
    ])

    springs = [
        ([0, 1], 3),
        ([1, 2], 5),
        ([0, 2], 4),
        ([3, 1], 3),
        ([3, 0], 1),
        ([1, 0, 2], 7)
    ]

    spring_system = SpringSystem(points=points, springs=springs)
    print(spring_system.compute_spring_forces())
    print(spring_system.get_spring_energy())
