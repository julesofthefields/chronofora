import itertools

import numpy as np
from scipy.sparse.csgraph import shortest_path
import networkx as nx
import plotly.graph_objects as go
import open3d as o3d
from sklearn.neighbors import NearestNeighbors


def make_iterable(obj):
    if isinstance(obj, (str, bytes)):
        return [obj]
    try:
        iter(obj)
        return obj
    except TypeError:
        return [obj]


def is_distance_matrix(D, tol=1e-6, triangle_check=False):
    n = len(D)
    assert (D >= -tol).all(), "distance matrix must have non-negative values"
    assert (np.abs(np.diag(D)) < tol).all(), "distance matrix must have zero diagonal values"
    assert (np.abs(D - D.T) < tol).all(), "distance matrix must be symmetric"
    if triangle_check:
        for (i, j, k) in itertools.product(range(n), repeat=3):
            assert (D[i, k] <= D[i, j] + D[j, k] + tol), f"distance matrix must verify triangle inequality at triplet {(i, j, k)}"
    return True


def restrict_knn_distance_matrix(D, k, sym=False):
    assert is_distance_matrix(D)
    assert (isinstance(k, int) and k >= 0), "k must be a non-negative integer"
    knn = np.argsort(np.where(D==0, np.inf, D), axis=1)[:, :k]
    is_knn = np.zeros(D.shape).astype(bool)
    for i, neighbors in enumerate(knn):
        for neighbor in neighbors:
            is_knn[i, neighbor] = True
        is_knn[i, i] = True
    D_knn = np.where(is_knn, D, np.inf)
    return np.minimum(D_knn, D_knn.T) if sym else D_knn


def restrict_cutoff_distance_matrix(D, cutoff):
    assert is_distance_matrix(D)
    assert (isinstance(cutoff, float) or isinstance(cutoff, int) and cutoff >= 0), "cutoff value must be non-negative"
    return np.where(D <= cutoff, D, np.inf)


def adjacency_matrix_from_distance_matrix(D):
    return (D < np.inf).astype(int)


def laplacian_from_adjacency_matrix(A):
    return np.diag(A.sum(axis=0)) - A


def is_fully_connected_graph(laplacian, tol=1e-8):
    eigenvals = np.linalg.eigvals(laplacian)
    return (np.abs(eigenvals) < tol).sum() == 1.


def floyd_warshall_completion(D):
    return shortest_path(D, method='FW')


def multidimensional_scaling(D, dim=3, tol=1e-9):
    centering = np.eye(D.shape[0]) - np.ones(D.shape[0]) / D.shape[0]
    B = -0.5 * centering @ (D ** 2) @ centering
    eigvals, eigvecs = np.linalg.eigh(B)
    eigvals = np.where(np.abs(eigvals) < tol, 0, eigvals)
    X = (np.sqrt(eigvals[-dim:]).reshape(-1, 1) * eigvecs[-dim:]).T
    return X


R_EARTH = 6371

def haversine_distance(lat1_in_radians, lon1_in_radians, lat2_in_radians, lon2_in_radians):
    dlat = lat1_in_radians.reshape(-1, 1) - lat2_in_radians.reshape(1, -1)
    dlon = lon1_in_radians.reshape(-1, 1) - lon2_in_radians.reshape(1, -1)
    temp = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_in_radians).reshape(-1, 1) * np.cos(lat2_in_radians).reshape(1, -1) * np.sin(dlon / 2.0) ** 2
    geodesic = R_EARTH * 2 * np.arcsin(np.sqrt(temp))
    return geodesic


def pairwise_haversine_distance(latitude_in_radians, longitude_in_radians):
    return haversine_distance(latitude_in_radians, longitude_in_radians, latitude_in_radians, longitude_in_radians)


def generate_random_points_on_surface_of_earth(size, mode='degrees'):
    function = np.degrees if mode == 'degrees' else lambda x: x
    longitudes = function(np.random.uniform(-np.pi, np.pi, size=size))
    latitudes = function(np.arcsin(np.random.uniform(-1, 1, size=size)))
    return np.column_stack((latitudes, longitudes))


def generate_regular_sample_on_surface_of_earth(size, mode='degrees'):
    function = np.degrees if mode == 'degrees' else lambda x: x

    indices = np.arange(0, size, dtype=np.float64) + 0.5

    phi = np.arccos(1 - 2 * indices / size)
    lat = np.pi / 2 - phi

    golden_angle = np.pi * (3 - np.sqrt(5))
    lon = (golden_angle * indices) % (2 * np.pi)
    lon = lon - np.pi

    lat_deg, lon_deg = function(lat), function(lon)

    return np.stack([lat_deg, lon_deg], axis=1)


def is_probably_globally_rigid(D, k, trials=1, assert_mode=True, easy=False):
    G = nx.from_numpy_array(np.where(np.isinf(D), 0, D))
    m = G.number_of_edges()
    n = G.number_of_nodes()

    # is dense enough
    has_enough_edges = (m >= k * n - (k * (k + 1)) // 2)
    if assert_mode:
        assert has_enough_edges, f"{m} > {k * n - (k * (k + 1)) // 2}"

    if easy: # poor heuristics because too expensive computationally otherwise
        is_connected = (min(dict(G.degree()).values()) >= (k+1))
        if assert_mode:
            assert is_connected, f"probably not connected"
        return is_connected and has_enough_edges
    
    else:
        # is connected
        conn = nx.node_connectivity(G)
        is_connected = conn >= (k + 1)# poor heuristic
        if assert_mode:
            assert is_connected, f"{conn} < ({k + 1})"


        # is probably generically rigid
        positions = np.random.randn(n, k, trials)
        R = np.zeros((m, k * n, trials))
        edges = np.array(G.edges(), dtype=int)
        
        pos_i = positions[edges[:, 0]]
        pos_j = positions[edges[:, 1]]
        
        diffs = pos_i - pos_j
        
        R = np.zeros((m, k * n, trials))
        
        for dim in range(k):
            R[np.arange(m), k * edges[:, 0] + dim] = diffs[:, dim]
            R[np.arange(m), k * edges[:, 1] + dim] = -diffs[:, dim]
        
        rigidity_ranks = [np.linalg.matrix_rank(rigidity) for rigidity in R.T]
        genericity = all([rank == k * n - (k * (k + 1)) // 2 for rank in rigidity_ranks])
        if assert_mode:
            assert genericity, rigidity_ranks

        return (is_connected and 
                has_enough_edges and 
                genericity)


def plot_on_sphere(X):
    latitudes = X[:,0]
    longitudes = X[:,1] 
    
    lat_rad = np.radians(latitudes)
    lon_rad = np.radians(longitudes)
    
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    
    fig = go.Figure(data=go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=5, color='red'),
        text=[f"({lat}, {lon})" for lat, lon in zip(latitudes, longitudes)],
        hoverinfo='text'
    ))
    
    u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
    xs = np.cos(u)*np.sin(v)
    ys = np.sin(u)*np.sin(v)
    zs = np.cos(v)
    
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=zs,
        opacity=0.2,
        showscale=False,
        colorscale='Blues'
    ))
    
    fig.update_layout(
        title='Points on Unit Sphere (lat/lon)',
        scene=dict(
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            zaxis=dict(showgrid=False)
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    fig.show()


def run_poisson_reconstruction(points, density_threshold=0.15, outlier_std_ratio=1.5, 
                              smoothing_neighbors=6, normal_radius=0.2, normal_max_nn=50,
                              poisson_depth=9, poisson_scale=1.2, laplacian_iterations=5):
    
    # create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # remove statistical outliers to reduce noise
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=outlier_std_ratio)
    
    # apply local smoothing to remaining points
    points_array = np.asarray(pcd.points)
    nbrs = NearestNeighbors(n_neighbors=smoothing_neighbors).fit(points_array)
    _, indices = nbrs.kneighbors(points_array)
    
    smoothed_points = np.zeros_like(points_array)
    for i in range(len(points_array)):
        neighbor_indices = indices[i][1:]  # exclude the point itself
        smoothed_points[i] = np.mean(points_array[neighbor_indices], axis=0)
    
    pcd.points = o3d.utility.Vector3dVector(smoothed_points)
    
    # estimate normals with larger radius for stability
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,    # adjustable radius
            max_nn=normal_max_nn     # adjustable max neighbors
        )
    )
    
    # orient normals consistently outward from centroid
    centroid = np.mean(smoothed_points, axis=0)
    normals = np.asarray(pcd.normals)
    
    for i in range(len(normals)):
        to_point = smoothed_points[i] - centroid
        if np.dot(normals[i], to_point) < 0:
            normals[i] = -normals[i]
    
    pcd.normals = o3d.utility.Vector3dVector(normals)
    
    # run Poisson reconstruction with parameters optimized for smooth surfaces
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, 
        depth=poisson_depth,   # adjustable octree depth
        width=0,         
        scale=poisson_scale,   # adjustable scale factor
        linear_fit=False       # better for curved surfaces
    )
    
    # remove low-density vertices more aggressively
    if density_threshold > 0:
        vertices_to_remove = densities < np.quantile(densities, density_threshold)
        mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # keep only the largest connected component
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    
    if len(cluster_n_triangles) > 0:
        largest_cluster_idx = cluster_n_triangles.argmax()
        triangles_to_remove = triangle_clusters != largest_cluster_idx
        mesh.remove_triangles_by_mask(triangles_to_remove)
    
    # apply smoothing to get cleaner surface
    mesh = mesh.filter_smooth_laplacian(number_of_iterations=laplacian_iterations)
    
    # clean up mesh
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    
    print(f"Reconstructed mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
    
    return mesh


if __name__ == '__main__':

    D = np.array([
        [0, 1, 2, 3, 4, 5, 1, 3, 5, 4, 5, 6],
        [1, 0, 1, 2, 3, 4, 2, 2, 4, 3, 4, 5],
        [2, 1, 0, 1, 2, 3, 3, 1, 3, 2, 3, 4],
        [3, 2, 1, 0, 1, 2, 4, 2, 2, 3, 4, 5],
        [4, 3, 2, 1, 0, 1, 5, 3, 1, 4, 5, 6],
        [5, 4, 3, 2, 1, 0, 6, 4, 2, 5, 6, 7],
        [1, 2, 3, 4, 5, 6, 0, 4, 6, 5, 6, 7],
        [3, 2, 1, 2, 3, 4, 4, 0, 4, 1, 2, 3],
        [5, 4, 3, 2, 1, 2, 6, 4, 0, 5, 6, 7],
        [4, 3, 2, 3, 4, 5, 5, 1, 5, 0, 1, 2],
        [5, 4, 3, 4, 5, 6, 6, 2, 6, 1, 0, 1],
        [6, 5, 4, 5, 6, 7, 7, 3, 7, 2, 1, 0],
    ])

    cutoff_D = restrict_cutoff_distance_matrix(D, 2)
    L = laplacian_from_adjacency_matrix(adjacency_matrix_from_distance_matrix(cutoff_D))
    hopla = is_fully_connected_graph(L)
