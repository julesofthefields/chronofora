import copy
import itertools

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
import open3d as o3d
import pickle
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

from map_utils import haversine_distance


### UTILS
#######################################################

def format_population(pop: float) -> str:
    """Show population figures in easy-to-read formats."""
    if pop >= 1e6:
        return f"{pop/1e6:.1f}M"
    elif pop >= 1e3:
        return f"{pop/1e3:.1f}K"
    else:
        return str(pop)


# dataframe with information on a large number of cities on Earth
df = pd.read_csv("data2/worldcities.csv")
df = df[['city', 'lat', 'lng', 'country', 'population']]

def get_city_coordinates(query: str, df: pd.DataFrame=df, max_suggestions: int=5) -> np.ndarray:
    """Get latitude and longitude from the name of a city on Earth."""
    # get city names which are most similar to query
    similarities = df["city"].apply(lambda x: fuzz.token_sort_ratio(query.title(), x))
    good_matches = df[similarities == similarities.max()].sort_values(by='population', ascending=False).iloc[:max_suggestions].reset_index()

    # if there is only possible name, return the coordinates associated to that name
    if len(good_matches) == 1:
        return good_matches.iloc[0][['lat', 'lng']].to_numpy().astype(float)
    
    # otherwise, ask the user, listing options by decreasing order of population
    else:
        question = "Do you mean:\n"
        for index, row in good_matches.iterrows():
            question += '\n' + rf'{index}: {row["city"]}, {row["country"]} (population ≈ {format_population(row["population"])})'
        question += "?"
        index = input(question)
        return good_matches.iloc[0][['lat', 'lng']].to_numpy().astype(float)

############################################################

@dataclass 
class PointShape:
    """Data structure of points on the mesh."""
    name: str
    coordinates: np.ndarray
    marker: o3d.cpu.pybind.geometry.TriangleMesh
    color: list = field(default_factory=lambda: [0, 0, 0])
    sphere_radius: float=0.2
    

class TransfoShape:
    """Manipulate the mesh."""

    def __init__(self, mesh, X_init, X_transfo, is_it_land, add_color=True):
        self.mesh = mesh
        self.vertices = np.array(self.mesh.vertices)
        self.X_init = X_init
        self.X_transfo = X_transfo
        self.diff = self.vertices[:, np.newaxis, :].astype(np.float16) - X_transfo[np.newaxis, :, :].astype(np.float16)
        self.dists = np.linalg.norm(self.diff, axis=2)
        self.is_it_land = is_it_land
        
        self.colors = itertools.cycle(plt.cm.tab10.colors)

        self.add_color = add_color
        if self.add_color:
            self.mesh = self.color_mesh()

        self.points = {}
        self.count_points = 0

    def color_mesh(self, primary_colors: dict=None, knn: int=10):
        """Color mesh according to land/sea through a neighbor approach."""
        # choose colors for land and sea
        if primary_colors is None:
            primary_colors = {}
        land_green = primary_colors.get('land_green', (85/255, 139/255, 47/255))
        sea_blue = primary_colors.get('sea_blue', (70/255, 130/255, 180/255))

        # neighbor approach to coloring the mesh
        mask = np.zeros_like(self.dists, dtype=int)
        knn_indices = np.argpartition(self.dists, kth=knn-1, axis=1)[:, :knn]
        rows = np.arange(self.dists.shape[0])[:, None]
        mask[rows, knn_indices] = 1

        # effectively, this keeps ~70% of Earth as sea
        is_it_land_transfo = (mask @ self.is_it_land.to_numpy() / knn) > .25 

        colors = np.full((len(self.vertices), 3), sea_blue)
        colors[is_it_land_transfo] = land_green

        color_mesh = copy.deepcopy(self.mesh)
        color_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        color_mesh.compute_vertex_normals()
        return color_mesh

    def show(self):
        """Show current geometries."""
        fig, ax = plt.subplots(figsize=(2, self.count_points))
        to_show = [self.mesh]
        for x in list(self.points.values()):
            to_show.append(x.marker)
            ax.scatter([], [], color=x.color, label=x.name)
        ax.legend(frameon=False)
        ax.axis("off")
        plt.show(block=False)
        o3d.visualization.draw_geometries(to_show)

    def new_coordinates(self, points_in_radians):
        """For a set of points on Earth, map them to images of their closest neighbors."""
        X_in_radians = np.radians(self.X_init)
        new_indices = np.argmin(haversine_distance(*points_in_radians.T, *X_in_radians.T), axis=1)
        closests = np.argmin(self.dists[:, new_indices], axis=0)
        return np.squeeze(self.vertices[closests])
    
    def mark_vertex(self, vertex_to_show, color=None, sphere_radius=0.2, name=None):
        """Make a vertex apparent."""
        # create a marker as a colored sphere
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        if color is None:
            color = next(self.colors)
        marker.paint_uniform_color(color)

        # set it at the chosen coordinates
        marker.translate(vertex_to_show)
        point_name = f'Point{self.count_points}' if name is None else name
        self.count_points += 1

        # cache it with the appropriate data structure
        self.points[point_name] = PointShape(point_name, vertex_to_show, marker, color, sphere_radius)

    def mark_city(self, city_name, color=None, sphere_radius=0.2):
        """Create the PointShape structure associated to a city and plot it on the mesh."""
        if city_name:
            city_coordinates = get_city_coordinates(city_name)
            new_city_coordinates = self.new_coordinates(np.radians(city_coordinates))
            self.mark_vertex(new_city_coordinates, color=color, sphere_radius=sphere_radius, name=city_name)

    def clean_mesh(self):
        """Remove all geometric objects."""
        self.geometries = []
    

if __name__ == '__main__':
    N = 10000
    k = 3
    dt = 1e-4
    damping = 5.0
    max_steps = 100000
    alpha = 100.0
    superspring_constant = 5.0
    surface_strength = 2.0
    alignment_strength = 5.0
    neighbors_smoothing = 10

    filename = f"data2/results/X_transfo_N{N}_k{k}_damping{damping}_dt{dt}_max_steps{max_steps}_alpha{alpha}_superspring_constant{superspring_constant}_surface{surface_strength}_alignment{alignment_strength}_smoothing{neighbors_smoothing}.pkl"

    with open(filename, 'rb') as f:
        X_transfo = pickle.load(f)

    # mesh = run_poisson_reconstruction(X_transfo, 
    #                               density_threshold=0., 
    #                               outlier_std_ratio=2,
    #                               smoothing_neighbors=20,
    #                               normal_radius=2,
    #                               poisson_depth=6,
    #                               poisson_scale=2,
    #                               laplacian_iterations=8)
    
    # o3d.io.write_triangle_mesh("data2/mesh2.ply", mesh)

    mesh = o3d.io.read_triangle_mesh("data2/mesh.ply")

    with open('data2/X_geometry_10000.pkl', 'rb') as f:
        X_init = pickle.load(f)['X']

    with open('data2/is_it_land_10000.pkl', 'rb') as f:
        is_it_land = pickle.load(f)

    chronofora = TransfoShape(mesh, X_init, X_transfo, is_it_land)

    cities = ['New York', 'Paris', 'London', 'Tokyo', 'Cape Town', 'Rio de Janeiro', 'Auckland', 'Sydney']
    for city in cities:
        chronofora.mark_city(city, sphere_radius=1)

    chronofora.show()
