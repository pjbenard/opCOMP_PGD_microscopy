import numpy as np
from scipy.spatial import distance_matrix

def load_positions(plafrim_path=True):
    if plafrim_path:
        positions = np.loadtxt('/beegfs/pbenard/these/code/solversuperres_v2/positions.csv', delimiter=',', skiprows=1)
    else:
        positions = np.loadtxt('positions.csv', delimiter=',', skiprows=1)
    return positions

def normalize_positions(positions):
    lbounds_old = np.array([0, 0, -670])
    ubounds_old = np.array([6400, 6400, 640])
    normalized_0_1 = (positions - lbounds_old) / (ubounds_old - lbounds_old)

    lbounds_new = np.array([0, 0, 0])
    ubounds_new = np.array([6.4, 6.4, 0.8])

    return (normalized_0_1 + lbounds_new) * (ubounds_new - lbounds_new)

def randomize_positions(positions):
    positions_idx = np.arange(positions.shape[0])
    np.random.shuffle(positions_idx)

    return positions[positions_idx]

def create_batches(nbatch, plafrim_path=True):
    positions = load_positions(plafrim_path=plafrim_path)
    positions = normalize_positions(positions)
    positions = randomize_positions(positions)
    nb_t = positions.shape[0]
    batches = [positions[nbatch * i: nbatch * (i + 1)] for i in range(nb_t // nbatch)]
    batches.append(positions[nbatch * (nb_t // nbatch):])

    return batches

def in_ellipse(point, center, radii):
    return np.sum((point - center)**2 / radii**2) <= 1

def semi_gridded_init(
    density=np.array([7, 7, 2]), 
    size=np.array([6.4, 6.4, 0.8]), 
    nrep=5_000
):
    dim_size = density.size
    
    dist_points = size / density
    range_dims = [np.linspace(
        dist_points[dim] / 2, 
        size[dim] - dist_points[dim] / 2, 
        num=density[dim]
    ) for dim in range(dim_size)]
    
    positions = np.meshgrid(*range_dims)
    grid_points = np.stack(list(map(np.ravel, positions)), axis=-1)

    
    radii = np.array(dist_points) / 2
    N = grid_points.shape[0]

    min_dist = 0
    max_min_dist = 0

    for it in range(nrep):
        t0 = []
        for grid_point in grid_points:
            point = grid_point + (2 * np.random.rand(dim_size) - 1) * radii
            while not in_ellipse(point, grid_point, radii):
                point = grid_point + (2 * np.random.rand(dim_size) - 1) * radii
            t0.append(point)
        t0 = np.array(t0)

        min_dist = np.min(distance_matrix(t0, t0) + np.eye(N))

        if min_dist > max_min_dist:
            max_min_dist = min_dist
            best_t0 = np.copy(t0)
            
    return best_t0

def random_init(
    N, 
    size=np.array([6.4, 6.4, 0.8]), 
    nrep=5_000
):
    dim_size = size.size
    min_dist = 0
    max_min_dist = 0

    for it in range(nrep):
        t0 = np.random.rand(N, dim_size) * np.array(size)

        min_dist = np.min(distance_matrix(t0, t0) + np.eye(N))

        if min_dist > max_min_dist:
            max_min_dist = min_dist
            best_t0 = np.copy(t0)
            
    return best_t0