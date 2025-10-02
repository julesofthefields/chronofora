import os
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import pickle

from map_utils import make_iterable, pairwise_haversine_distance, restrict_knn_distance_matrix, adjacency_matrix_from_distance_matrix, haversine_distance, floyd_warshall_completion, generate_regular_sample_on_surface_of_earth


LAND = gpd.read_file('data2/ne_50m_land.shp')
os.environ['SHAPE_RESTORE_SHX'] = 'YES'


def is_land_from_degrees(lats: list[float], lons: list[float], land=LAND):
    """Determine if Earth coordinates correspond to land or water."""
    lats, lons = make_iterable(lats), make_iterable(lons)
    points = gpd.GeoSeries([Point(lon, lat) for lat, lon in zip(lats, lons)])
    return points.within(land.union_all())


def write_full_data(plane_speed: float=750., 
                    train_speed: float=100., 
                    ferry_speed: float=40., 
                    car_speed: float=50.,
                    neighbour_ports: int=4,
                    neighbour_stations: int=5,
                    show_routes=False):
    ## AIRPORT DATA (useful later so we load it everytime)
    # clean raw data
    airport_df_raw = pd.read_csv('data2/airports.csv')
    airport_filter = (airport_df_raw['scheduled_service'] == 'yes') & (airport_df_raw['type'].isin(['large_airport', 'medium_airport']))
    airport_keys = ['latitude_deg', 'longitude_deg', 'iata_code']
    airport_df = airport_df_raw[airport_filter][airport_keys].dropna()
    airport_df.rename(columns={'iata_code': 'name'}, inplace=True)
    airport_df = airport_df.reset_index().drop(columns=['index'])

    # airport locations
    airport_latitude, airport_longitude = np.radians(airport_df['latitude_deg'].to_numpy()), np.radians(airport_df['longitude_deg'].to_numpy())

    # flight routes
    airport_routes_raw = pd.read_csv('data2/flights.txt', header=None)
    openflights_routes_columns = ['airline_id', 'openflights_airline_id', 'airport_source', 'airport_source_id', 'airport_destination', 'airport_destination_id', 'codeshare', 'stops', 'equipment']
    airport_routes_raw.columns = openflights_routes_columns
    airport_routes = airport_routes_raw[airport_routes_raw['stops'] == 0][['airport_source', 'airport_destination']]
    airport_routes.columns = ['source', 'destination']
    airport_names = airport_df['name'].tolist()
    airport_routes = airport_routes[airport_routes['source'].isin(airport_names) & airport_routes['destination'].isin(airport_names)]

    airport_name_to_idx = {name: i for i, name in enumerate(airport_names)}
    airport_adj = np.zeros((len(airport_names), len(airport_names)))

    for src, dst in airport_routes[['source', 'destination']].values:
        airport_adj[airport_name_to_idx[src], airport_name_to_idx[dst]] = 1

    airport_adj = np.where(airport_adj == 0, np.inf, airport_adj)
    np.fill_diagonal(airport_adj, 0)

    airport_D = pairwise_haversine_distance(airport_latitude, airport_longitude)
    airport_D_restrict = (airport_D * airport_adj)

    # flight route times
    airport_times = airport_D_restrict / plane_speed

    ## PORT DATA
    # clean raw data
    port_df_raw = pd.read_excel('data2/WPI2019.xls')
    port_filter = ((port_df_raw['HARBORSIZE'].isin(['L', 'M'])) & (port_df_raw['PORTOFENTR'] == 'Y') & (port_df_raw['SHELTER'].isin(['G', 'F'])))
    port_keys = ['PORT_NAME', 'LATITUDE', 'LONGITUDE']
    port_df = port_df_raw[port_filter][port_keys].dropna().reset_index().drop(columns=['index'])
    port_df.columns = ['name', 'latitude_deg', 'longitude_deg']
    port_df.loc[port_df['name'] == 'WILMINGTON', 'name'] = ['WILMINGTON NC', 'WILMINGTON DE']

    # port locations
    port_latitude, port_longitude = np.radians(port_df['latitude_deg'].to_numpy()), np.radians(port_df['longitude_deg'].to_numpy())

    # ferry routes (assuming given number of neighbouring ports)
    port_D = pairwise_haversine_distance(port_latitude, port_longitude)
    port_D_restrict = restrict_knn_distance_matrix(port_D, neighbour_ports)

    if show_routes:
        port_adj = adjacency_matrix_from_distance_matrix(port_D_restrict)
        np.fill_diagonal(port_adj, 0)
        
        port_src_idx, port_dst_idx = np.where(port_adj)
        port_names = port_df['name'].tolist()
        
        port_routes = pd.DataFrame({
            'source': [port_names[i] for i in port_src_idx],
            'destination': [port_names[j] for j in port_dst_idx]
        })

        port_routes.to_parquet('data2/port_routes.pqt')

    # ferry route times
    port_times = port_D_restrict / ferry_speed

    ## TRAIN STATION DATA
    # clean raw data
    transportation_raw = pd.read_csv('data2/transportation.txt', header=None)
    openflights_columns = ['id', 'name', 'city', 'country', 'iata_code', 'icao_code', 'latitude_deg', 'longitude_deg', 'elevation_deg', 'timezone', 'DST', 'timezone_db', 'type', 'source']
    transportation_raw.columns = openflights_columns
    station_df = transportation_raw[(transportation_raw['type'] == 'station')].drop_duplicates(subset=['name', 'city', 'country'], keep='last')
    station_df['name'] = station_df.apply(lambda row: f"{row['name']}, {row['city']}, {row['country']}", axis=1)
    station_df = station_df[['latitude_deg', 'longitude_deg', 'name']].reset_index().drop(columns=['index'])
    station_df = station_df.replace(r'\\N', np.nan, regex=True).dropna()

    # train station locations
    station_latitude, station_longitude = np.radians(station_df['latitude_deg'].to_numpy()), np.radians(station_df['longitude_deg'].to_numpy())

    # train routes (assuming given number of neighbouring train stations)
    station_D = pairwise_haversine_distance(station_latitude, station_longitude)
    station_D_restrict = restrict_knn_distance_matrix(station_D, neighbour_stations)

    if show_routes:
        station_adj = adjacency_matrix_from_distance_matrix(station_D_restrict)
        np.fill_diagonal(station_adj, 0)
        
        station_src_idx, station_dst_idx = np.where(station_adj)
        station_names = station_df['name'].tolist()
        
        station_routes = pd.DataFrame({
            'source': [station_names[i] for i in station_src_idx],
            'destination': [station_names[j] for j in station_dst_idx]
        })

        station_routes.to_parquet('data2/station_routes.pqt')

    # train route times
    station_times = station_D_restrict / train_speed

    ### COMPLETE TRANSPORTATION DATA
    
    transportation = pd.concat({
        'plane': airport_df.set_index('name'),
        'ferry': port_df.set_index('name'),
        'train': station_df.set_index('name')
    })

    transportation.to_parquet('data2/transportation.pqt')

    ## complete transportation times between distinct means of transportation
    # between airports and train stations
    airport_to_station_times = haversine_distance(airport_latitude, airport_longitude, station_latitude, station_longitude) / car_speed
    station_to_airport_times = airport_to_station_times.T

    # between train stations and ports
    station_to_port_times = haversine_distance(station_latitude, station_longitude, port_latitude, port_longitude) / car_speed
    port_to_station_times = station_to_port_times.T

    # between ports and airports
    port_to_airport_times = haversine_distance(port_latitude, port_longitude, airport_latitude, airport_longitude) / car_speed
    airport_to_port_times = port_to_airport_times.T

    ## create transportation time matrix
    transportation_times = np.block([
        [airport_times, airport_to_port_times, airport_to_station_times],
        [port_to_airport_times, port_times, port_to_station_times],
        [station_to_airport_times, station_to_port_times, station_times]
    ])

    ## compute transportation times between all means of transportation
    geodesics = floyd_warshall_completion(transportation_times)
    geodesics = 0.5 * (geodesics + geodesics.T) # does it need to be symmetric?
    
    geodesics_df = pd.DataFrame(geodesics, 
                                index=transportation.index.get_level_values(1).tolist(), 
                                columns=transportation.index.get_level_values(1).tolist())
    geodesics_df.to_parquet('data2/geodesics.pqt')


def get_airport_data(plane_speed: float=750.):
    # clean raw data
    airport_df_raw = pd.read_csv('data2/airports.csv')
    airport_filter = (airport_df_raw['scheduled_service'] == 'yes') & (airport_df_raw['type'].isin(['large_airport', 'medium_airport']))
    airport_keys = ['latitude_deg', 'longitude_deg', 'iata_code']
    airport_df = airport_df_raw[airport_filter][airport_keys].dropna()
    airport_df.rename(columns={'iata_code': 'name'}, inplace=True)
    airport_df = airport_df.reset_index().drop(columns=['index'])

    # airport locations
    airport_latitude, airport_longitude = np.radians(airport_df['latitude_deg'].to_numpy()), np.radians(airport_df['longitude_deg'].to_numpy())

    # flight routes
    airport_routes_raw = pd.read_csv('data2/flights.txt', header=None)
    openflights_routes_columns = ['airline_id', 'openflights_airline_id', 'airport_source', 'airport_source_id', 'airport_destination', 'airport_destination_id', 'codeshare', 'stops', 'equipment']
    airport_routes_raw.columns = openflights_routes_columns
    airport_routes = airport_routes_raw[airport_routes_raw['stops'] == 0][['airport_source', 'airport_destination']]
    airport_routes.columns = ['source', 'destination']
    airport_names = airport_df['name'].tolist()
    airport_routes = airport_routes[airport_routes['source'].isin(airport_names) & airport_routes['destination'].isin(airport_names)]

    airport_name_to_idx = {name: i for i, name in enumerate(airport_names)}
    airport_adj = np.zeros((len(airport_names), len(airport_names)))

    for src, dst in airport_routes[['source', 'destination']].values:
        airport_adj[airport_name_to_idx[src], airport_name_to_idx[dst]] = 1

    airport_adj = np.where(airport_adj == 0, np.inf, airport_adj)
    np.fill_diagonal(airport_adj, 0)

    airport_D = pairwise_haversine_distance(airport_latitude, airport_longitude)
    airport_D_restrict = (airport_D * airport_adj)

    # flight route times
    airport_times = airport_D_restrict / plane_speed

    return airport_name_to_idx, airport_times


def get_X_transportation_geometry(N: int=10_000, flight_time_threshold: float=3., ferry_speed: float=40., car_speed: float=50.):
    """
    Get X coordinates, geodesic times between X and transportation stations, and geodesic times of easy routes (direct by car or ferry).
    3h flight time threshold covers 75% of all major flights, meaning only 25% of all major flights last longer than 3h.
    """

    # pre-setting up
    airport_name_to_idx, airport_times = get_airport_data()
    airport_names = list(airport_name.keys())

    # loading coordinates of transportation stations
    transportation = pd.read_parquet('data2/transportation.pqt')
    type_to_idx = {'plane': [], 'ferry': [], 'train': []}
    for i, (idx, _) in enumerate(transportation.iterrows()):
        type_to_idx[idx[0]].append(i)

    ferry_is_here = np.zeros(len(transportation))
    ferry_is_here[type_to_idx['ferry']] = 1

    # loading durations of geodesics between transportation stations
    geodesics_df = pd.read_parquet('data2/geodesics.pqt')
    geodesics = geodesics_df.to_numpy()

    ## CREATE POINT SAMPLE
    # add to X all major airports because airport bend space the most
    longest_flights = pd.Series(np.sum(np.where(~np.isinf(airport_times), (airport_times > flight_time_threshold), 0), axis=1), index=airport_names)
    major_airports = longest_flights[longest_flights != 0].index.tolist()
    major_airports_locations = transportation.loc['plane'].loc[major_airports].to_numpy()

    # generate regular sample on unit sphere, with major airports because they bend space the most
    X = generate_regular_sample_on_surface_of_earth(N)
    
    # compute distances between X and all major airports
    haversines = haversine_distance(*np.radians(X).T, *np.radians(major_airports_locations).T)

    # for each major airport, find the point on X which is closest
    representatives = np.argmin(haversines, axis=0)
    representatives_dic = {representative: [] for representative in representatives.tolist()}
    for airport, representative in zip(major_airports, representatives.tolist()):
        representatives_dic[representative].append(airport)

    # for each point that has been identified as close to a major airport, associate the most important major airport, i.e. the one with the largest number of long flights
    correct_representatives_dic = {representative: [] for representative in representatives.tolist()}
    for representative in np.unique(representatives.tolist()):
        correct_representatives_dic[representative] = longest_flights[representatives_dic[representative]].idxmax()

    # find a unique point on X to which we can associate every eligible major airport
    matching = pd.Series(correct_representatives_dic).reset_index()
    matching.columns = ['closest_point', 'airport']
    closest_points_list_per_airport = matching.groupby('airport')['closest_point'].apply(list)
    major_airport_idx = {airport_name: i for i, airport_name in enumerate(major_airports)}

    # then replace the original point on X with the location of the matching major airport
    final_matching = {}
    for major_airport, closest_points_list in closest_points_list_per_airport.items():
        final_matching[major_airport] = closest_points_list[np.argmin(haversines[closest_points_list, major_airport_idx[major_airport]])]

    for major_airport, X_index in final_matching.items():
        X[X_index] = major_airports_locations[major_airport_idx[major_airport]]

    ## COMPUTE GEODESIC TIMES
    # Haversine distances, by car
    d_X = haversine_distance(*np.radians(X.T), np.radians(transportation['latitude_deg'].to_numpy()), np.radians(transportation['longitude_deg'].to_numpy()))
    t_X = d_X / car_speed

    # replace t_X with 0 transportation time to airport when point is associated to an airport
    for airport_name, idx in final_matching.items():
        t_X[idx, airport_name_to_idx[airport_name]] = 0

    # major airports are supposedly on land
    is_land_array = is_land_from_degrees(*X.T).to_numpy()
    is_land_array[list(final_matching.values())] = True
    is_land = is_land_array.tolist()
    is_ferry_array = np.array([np.ones(len(transportation)) if is_it_land else ferry_is_here for is_it_land in is_land])

    t = np.where(is_ferry_array, t_X, np.inf)
    t = t.astype(np.float16)

    # geodesic times using car or ferry only
    geodesic_times_car_only = np.where(np.array(is_land).reshape(-1, 1) * np.array(is_land).reshape(1, -1), pairwise_haversine_distance(*np.radians(X.T)) / car_speed, np.inf)
    geodesic_times_ferry_only = np.where(~(np.array(is_land).reshape(-1, 1) * np.array(is_land).reshape(1, -1)), pairwise_haversine_distance(*np.radians(X.T)) / ferry_speed, np.inf)

    X_transportation_geometry = {
        'X': X,
        't': t,
        'geodesic_times_car_only': geodesic_times_car_only,
        'geodesic_times_ferry_only': geodesic_times_ferry_only,
        'geodesics': geodesics,
        'airport_matching': final_matching
    }

    return X_transportation_geometry


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Construct dictionary of X geometric structure.")
    parser.add_argument('--N', type=int, default=1000)
    args = parser.parse_args()
    N = args.N

    X_transportation_geometry = get_X_transportation_geometry(N)

    with open(f"data2/X_transportation_geometry_{N}.pkl", "wb") as f:
        pickle.dump(X_transportation_geometry, f)
