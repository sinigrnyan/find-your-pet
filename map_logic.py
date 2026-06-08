import math
import random
from collections import Counter, defaultdict
import numpy as np
import networkx as nx
import osmnx as ox
import folium
import json
from folium import Element, PolyLine, CircleMarker
from folium.plugins import HeatMap
from sklearn.cluster import KMeans
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
from shapely.prepared import prep
from functools import lru_cache
from shapely.vectorized import contains
from scipy.spatial.distance import cdist
import time

env_cache = {}
node_cache = {}
cveta = ['red', 'blue', 'green', 'purple', 'orange']
def euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def greedy_tsp(points):
    if len(points) <= 1:
        return list(range(len(points)))
    unvisited = set(range(1, len(points)))
    route = [0]
    cur = 0
    while unvisited:
        nxt = min(
            unvisited,
            key=lambda i: euclid(points[cur], points[i])
        )
        route.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return route


def route_length(route, points):
    total = 0
    for i in range(len(route)-1):
        total += euclid(
            points[route[i]],
            points[route[i+1]]
        )
    return total


def two_opt(route, points):
    improved = True
    while improved:
        improved = False
        best_len = route_length(route, points)
        for i in range(1, len(route)-2):
            for j in range(i+1, len(route)):
                candidate = (
                    route[:i]
                    + route[i:j+1][::-1]
                    + route[j+1:]
                )
                cand_len = route_length(
                    candidate,
                    points
                )
                if cand_len < best_len:
                    route = candidate
                    best_len = cand_len
                    improved = True
        if not improved:
            break
    return route
pov = {
    ("koshka", "truslivaya"): {"P_STAY": 0.18, "P_RETURN": 0.06, "MAX_STEP": 50.0, "EDGE_SCALE": 30.0, "AVOID_MAJOR_ROADS": True},
    ("koshka", "uverennaya"): {"P_STAY": 0.08, "P_RETURN": 0.03, "MAX_STEP": 80.0, "EDGE_SCALE": 60.0, "AVOID_MAJOR_ROADS": True},
    ("sobaka", "truslivaya"): {"P_STAY": 0.06, "P_RETURN": 0.03, "MAX_STEP": 90.0, "EDGE_SCALE": 55.0, "AVOID_MAJOR_ROADS": True},
    ("sobaka", "uverennaya"): {"P_STAY": 0.02, "P_RETURN": 0.01, "MAX_STEP": 150.0, "EDGE_SCALE": 120.0, "AVOID_MAJOR_ROADS": False}
}
ox.settings.use_cache = True
ox.settings.log_console = True
ox.settings.headers = {
    "User-Agent": "SinigrNyan (sinigrchannel@gmail.com)",
    "Referer": "https://2072-34-80-242-236.ngrok-free.app"
}
@lru_cache(maxsize=16)
def load_osm_cached(lat_key, lon_key, rad_key):

    G = ox.graph_from_point(
        (lat_key, lon_key),
        dist=rad_key,
        network_type="all",
        simplify=True
    )

    G = ox.distance.add_edge_lengths(G)

    tags = {
        "landuse": True,
        "natural": True,
        "leisure": True,
        "highway": True,
        "amenity": True,
        "building": True
    }

    gdf = ox.geometries_from_point(
        (lat_key, lon_key),
        tags=tags,
        dist=rad_key
    )

    for col in [
        "landuse",
        "natural",
        "leisure",
        "highway",
        "amenity",
        "building"
    ]:
        if col not in gdf.columns:
            gdf[col] = None

    return G, gdf

def generate_map(zhiv, khar, shir, dolg, rad, vol):
    global env_cache, node_cache
    full_t0 = time.perf_counter()
    profile = {
        "is_valid": 0,
        "is_path_valid": 0
    }
    env_cache = {}
    node_cache = {}
    conf = pov[(zhiv, khar)]
    rng = random.Random(228)
    def to_xy(lat, lon):
        x = (lon - dolg) * 111320.0 * math.cos(math.radians(shir))
        y = (lat - shir) * 111320.0
        return x, y

    def to_latlon(x, y):
        lon = dolg + x / (111320.0 * math.cos(math.radians(shir)))
        lat = shir + y / 111320.0
        return lat, lon

    def is_valid(x, y):
        profile["is_valid"] += 1
        dist_from_center = math.hypot(x, y)

        if dist_from_center > rad:
            return False

        idx = grid_index(x, y)

        if idx is None:
            return False

        gx, gy = idx

        return free_mask[gy, gx] == 1

    def is_path_valid(x1, y1, x2, y2):
        profile["is_path_valid"] += 1
        dist = math.hypot(x2 - x1, y2 - y1)

        steps = max(10, int(dist / 5))

        for i in range(steps + 1):

            t = i / steps

            x = x1 + (x2 - x1) * t
            y = y1 + (y2 - y1) * t

            if not is_valid(x, y):
                return False

        return True

    def mestnost(x, y, golod, stress):

        idx = grid_index(x, y)

        if idx is None:
            return 0.0

        gx, gy = idx

        key = (
            gx,
            gy,
            int(golod * 10),
            int(stress * 10)
        )

        if key in env_cache:
            return env_cache[key]

        score = 1.0
        if walkable_mask[gy, gx]:
            score *= 1.4 + 0.3 * stress
        if res_mask[gy, gx]:
            score *= 1.6
        if parking_mask[gy, gx]:
            score *= 1.2 + golod
        if roads_mask[gy, gx]:
            score *= max(0.3, 1.0 - stress * 1.5)
        env_cache[key] = score

        return score


    def shag(pos, predugl, golod, stress, zhiv1, khar1):

        x, y = pos

        best = None
        best_score = -1

        ATTEMPTS = 4

        for _ in range(ATTEMPTS):

            if zhiv1 == "koshka":
                angle_spread = 0.4
            elif khar1 == "uverennaya":
                angle_spread = 0.25
            else:
                angle_spread = 0.6

            ugl = predugl + rng.uniform(-angle_spread, angle_spread)

            max_step = conf["MAX_STEP"]


            if zhiv1 == "sobaka" and khar1 == "uverennaya":
                dist = rng.triangular(
                    20,
                    max_step,
                    max_step * 0.75
                )
            else:
                dist = rng.triangular(
                    10,
                    max_step,
                    max_step * 0.45
                )

            nx = x + dist * math.cos(ugl)
            ny = y + dist * math.sin(ugl)

            if not is_valid(nx, ny):
                continue

            if not is_path_valid(x, y, nx, ny):
                continue

            d = math.hypot(
                nx - start_xy[0],
                ny - start_xy[1]
            )

            if zhiv1 == "sobaka" and khar1 == "uverennaya":
                home_scale = 3.5
            elif zhiv1 == "sobaka":
                home_scale = 1.8
            elif khar1 == "uverennaya":
                home_scale = 1.3
            else:
                home_scale = 0.9

            env = mestnost(nx, ny, golod, stress)


            if zhiv1 == "sobaka" and khar1 == "uverennaya":
                explore_bonus = 2.5
            elif zhiv1 == "sobaka":
                explore_bonus = 1.7
            elif khar1 == "uverennaya":
                explore_bonus = 1.5
            else:
                explore_bonus = 0.7

            r = d / rad

            if r < 0.5:
                issled = 1.0 + r * explore_bonus

            elif r < 0.8:
                issled = 1.0 + 0.5 * explore_bonus

            else:
                edge_penalty = 1.0 - (r - 0.8) / 0.2
                edge_penalty = max(0.3, edge_penalty)

                issled = (
                    1.0 +
                    0.5 * explore_bonus
                ) * edge_penalty

            golod_drive = 1.0 + golod

            score = (
                env
                * issled
                * golod_drive
            )

            if score > best_score:
                best = (nx, ny, ugl)
                best_score = score
        if best is None:

            for _ in range(20):

                dist = rng.uniform(5, 20)
                ugl = rng.uniform(0, 2 * math.pi)

                nx = x + dist * math.cos(ugl)
                ny = y + dist * math.sin(ugl)

                if is_valid(nx, ny):
                    return (nx, ny), ugl

            return pos, predugl

        return (best[0], best[1]), best[2]
    start_xy = to_xy(shir, dolg)
    lat_key = round(shir, 3)
    lon_key = round(dolg, 3)
    rad_key = int(rad)
    t0 = time.perf_counter()
    G, gdf = load_osm_cached(
        lat_key,
        lon_key,
        rad_key
    )
    print(
        f"[PROFILE] OSM load: "
        f"{time.perf_counter()-t0:.3f}s "
        f"(nodes={len(G.nodes)}, "
        f"edges={len(G.edges)}, "
        f"objects={len(gdf)})"
    )
    start = ox.distance.nearest_nodes(G, dolg, shir)
    for col in ["landuse", "natural", "leisure", "highway", "amenity"]:
        if col not in gdf.columns:
            gdf[col] = None
    walkable = gdf[
        gdf["landuse"].isin(["forest", "grass", "meadow"]) |
        gdf["natural"].isin(["wood", "grassland", "scrub"]) |
        gdf["leisure"].isin(["park"])
    ]
    blocked = gdf[
        gdf["natural"].isin(["water"]) |
        gdf["landuse"].isin(["industrial", "commercial"]) |
        (gdf["building"].notna())
    ]
    t0 = time.perf_counter()
    walkable_geom = unary_union(walkable.geometry) if not walkable.empty else None
    blocked_geom = unary_union(blocked.geometry) if not blocked.empty else None
    walkable_union = prep(walkable_geom) if walkable_geom else None
    blocked_union = prep(blocked_geom) if blocked_geom else None
    residential = gdf[gdf["landuse"].isin(["residential"])]
    parking = gdf[gdf["amenity"].isin(["parking"])]
    roads = gdf[gdf["highway"].notna()]
    res_geom = unary_union(residential.geometry) if not residential.empty else None
    parking_geom = unary_union(parking.geometry) if not parking.empty else None
    roads_geom = unary_union(roads.geometry) if not roads.empty else None
    res_union = prep(res_geom) if res_geom else None
    parking_union = prep(parking_geom) if parking_geom else None
    roads_union = prep(roads_geom) if roads_geom else None
    print(
        f"[PROFILE] unary_union: "
        f"{time.perf_counter()-t0:.3f}s"
    )
    GRID_RES = 8.0

    xmin = -rad
    xmax = rad
    ymin = -rad
    ymax = rad
    t0 = time.perf_counter()
    xs = np.arange(xmin, xmax, GRID_RES)
    ys = np.arange(ymin, ymax, GRID_RES)

    xx, yy = np.meshgrid(xs, ys)
    print(
        f"[PROFILE] grid build: "
        f"{time.perf_counter()-t0:.3f}s "
        f"shape={xx.shape}"
    )

    lon_grid = dolg + xx / (111320.0 * math.cos(math.radians(shir)))
    lat_grid = shir + yy / 111320.0

    free_mask = np.ones(xx.shape, dtype=np.uint8)
    walkable_mask = np.zeros(xx.shape, dtype=np.uint8)
    roads_mask = np.zeros(xx.shape, dtype=np.uint8)
    res_mask = np.zeros(xx.shape, dtype=np.uint8)
    parking_mask = np.zeros(xx.shape, dtype=np.uint8)
    t0 = time.perf_counter()
    if walkable_geom:
        walkable_mask[contains(walkable_geom, lon_grid, lat_grid)] = 1

    if roads_geom:
        roads_mask[contains(roads_geom, lon_grid, lat_grid)] = 1

    if res_geom:
        res_mask[contains(res_geom, lon_grid, lat_grid)] = 1

    if parking_geom:
        parking_mask[contains(parking_geom, lon_grid, lat_grid)] = 1

    if blocked_geom:
        blocked_mask = contains(
            blocked_geom,
            lon_grid,
            lat_grid
        )

        free_mask[blocked_mask] = 0
    print(
        f"[PROFILE] masks build: "
        f"{time.perf_counter()-t0:.3f}s"
    )
    def grid_index(x, y):

        gx = int((x - xmin) / GRID_RES)
        gy = int((y - ymin) / GRID_RES)

        if gx < 0 or gy < 0:
            return None

        if gx >= free_mask.shape[1]:
            return None

        if gy >= free_mask.shape[0]:
            return None

        return gx, gy
    sim_t0 = time.perf_counter()
    vis = []
    t_global = 0
    GOLOD_MAX = 1.0
    STRESS_MAX = 1.0
    GOLOD_GROWTH = 0.004
    STRESS_UPAD = 0.01
    SHUM_STRESS = 0.05

    for _ in range(750):
        cur = start_xy
        ugl = rng.uniform(0, 2*math.pi)
        golod = rng.uniform(0.2, 0.4)
        stress = rng.uniform(0.1, 0.3)
        for _ in range(180):
            env_here = mestnost(cur[0], cur[1], golod, stress)
            stay_prob = min(0.95, conf["P_STAY"] * env_here * (1 + stress))
            if rng.random() < stay_prob:
                vis.append((cur[0], cur[1], t_global))
                golod = min(GOLOD_MAX, golod + GOLOD_GROWTH)
                t_global = t_global + 1
                continue
            if zhiv == "sobaka" and khar == "uverennaya":
                return_prob = conf["P_RETURN"] * 0.15
            elif zhiv == "sobaka":
                return_prob = conf["P_RETURN"] * 0.5
            else:
                return_prob = conf["P_RETURN"] * (1 + stress * 2)
            if rng.random() < return_prob:
                dx = start_xy[0] - cur[0]
                dy = start_xy[1] - cur[1]
                norm = math.hypot(dx, dy) + 1e-9
                step_size = conf["MAX_STEP"] * 0.5
                cand = (cur[0] + dx/norm*step_size, cur[1] + dy/norm*step_size)
                if is_path_valid(cur[0], cur[1], cand[0], cand[1]):
                    cur = cand
                vis.append((cur[0], cur[1], t_global))
                golod = min(GOLOD_MAX, golod + GOLOD_GROWTH)
                t_global = t_global + 1
                continue
            cur, ugl = shag(cur, ugl, golod, stress, zhiv, khar)
            vis.append((cur[0], cur[1], t_global))
            golod = min(GOLOD_MAX, golod + GOLOD_GROWTH)
            t_global = t_global + 1
    print(
        f"[PROFILE] simulation: "
        f"{time.perf_counter()-sim_t0:.3f}s "
        f"vis={len(vis)}"
    )
    grid = defaultdict(float)
    cell = 1
    T = vis[-1][2] + 1 if vis else 1
    for x, y, t in vis:
        if not is_valid(x, y):
            continue
        dist_from_home = math.hypot(
            x - start_xy[0],
            y - start_xy[1]
        )

        if zhiv == "sobaka" and khar == "uverennaya":

            dist_bonus = 1.0 + (dist_from_home / rad) * 2.0

        elif zhiv == "sobaka" and khar == "truslivaya":

            dist_bonus = max(
                0.5,
                1.0 - (dist_from_home / rad) * 0.4
            )

        elif zhiv == "koshka" and khar == "uverennaya":

            dist_bonus = 1.0 + (dist_from_home / rad) * 1.2

        else:

            dist_bonus = max(
                0.2,
                1.0 - (dist_from_home / rad)
            )

        w = (
            ((t + 1) / T) ** 1.5
            * dist_bonus
            * rng.uniform(0.9, 1.1)
        )
        gx = int(x // cell)
        gy = int(y // cell)
        grid[(gx, gy)] = max(grid[(gx, gy)], w)
    values = np.array(list(grid.values()))
    grid = {k: v for k, v in grid.items() if v >= np.percentile(values, 15)}
    max_w = max(grid.values()) if grid else 1
    heat = []
    srt = sorted(grid.items(), key=lambda x: -x[1])
    TOP_POINTS = []
    for i, ((gx, gy), w) in enumerate(srt):
        x = gx * cell
        y = gy * cell
        lat, lon = to_latlon(x, y)
        w_norm = (w / max_w) ** 2.5
        heat.append([lat, lon, w_norm])
        if i < 10:
            TOP_POINTS.append((lat, lon, w_norm))
    weights = np.array([w for _, _, w in heat])

    hot_threshold = np.percentile(weights, 85)
    och = [
        (lat, lon, 1.0)
        for lat, lon, w in heat
        if w >= hot_threshold
    ]
    if len(och) > 0:

        # Чем больше радиус поиска,
        # тем крупнее ячейка объединения

        MERGE_CELL = max(
            10,          # минимум 40 м
            rad / 6     # растёт вместе с радиусом
        )

        clusters = defaultdict(list)

        for lat, lon, w in och:

            x, y = to_xy(lat, lon)

            gx = int(x // MERGE_CELL)
            gy = int(y // MERGE_CELL)

            clusters[(gx, gy)].append(
                (lat, lon, w)
            )

        och_new = []

        for pts in clusters.values():

            mean_lat = np.mean([p[0] for p in pts])
            mean_lon = np.mean([p[1] for p in pts])

            mean_w = np.mean([p[2] for p in pts])

            och_new.append(
                (
                    mean_lat,
                    mean_lon,
                    mean_w
                )
            )

        print(
            f"[PROFILE] hotpoints reduce: "
            f"{len(och)} -> {len(och_new)}"
        )

        och = och_new
    t0 = time.perf_counter()
    if len(och) == 0:

        clust = [[] for _ in range(vol)]

    else:

        coords = np.array([
            to_xy(lat, lon)
            for lat, lon, _ in och
        ])

        k = min(vol, len(coords))
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=30
        )

        labels = kmeans.fit_predict(coords)

        centers = kmeans.cluster_centers_

        clust = [[] for _ in range(k)]

        assigned_routes = [[] for _ in range(k)]

        for idx, point in enumerate(coords):

            best_cluster = None
            best_score = 1e18

            for c in range(k):

                center_dist = np.linalg.norm(
                    point - centers[c]
                )

                overlap_penalty = 0.0

                for oc in range(k):

                    if oc == c:
                        continue

                    if len(assigned_routes[oc]) == 0:
                        continue

                    d = np.min(
                        cdist(
                            [point],
                            assigned_routes[oc]
                        )
                    )

                    overlap_penalty += 300.0 / (d + 1.0)

                score = center_dist + overlap_penalty

                if score < best_score:
                    best_score = score
                    best_cluster = c

            clust[best_cluster].append(och[idx])

            assigned_routes[best_cluster].append(point)
    print(
        f"[PROFILE] clustering: "
        f"{time.perf_counter()-t0:.3f}s "
        f"hot_points={len(och)}"
    )
    marsh = [[] for _ in range(vol)]
    nearest_time = 0
    routing_time = 0
    routing_calls = 0
    for agent_id, points in enumerate(clust):
        nodes = []
        for lat, lon, _ in points:
            try:
                key = (round(lat, 5), round(lon, 5))
                if key in node_cache:
                    node = node_cache[key]
                else:
                    t1 = time.perf_counter()
                    node = ox.distance.nearest_nodes(G, lon, lat)
                    nearest_time += (
                        time.perf_counter() - t1
                    )
                    node_cache[key] = node
                nodes.append(node)
            except:
                continue
        nodes = list(set(nodes))
        if not nodes:
            continue
        node_coords = []

        for node in nodes:

            node_coords.append(
                (
                    G.nodes[node]["x"],
                    G.nodes[node]["y"]
                )
            )

        if len(node_coords) <= 1:

            order = list(range(len(node_coords)))

        else:
            t1 = time.perf_counter()
            order = greedy_tsp(node_coords)

            order = two_opt(
                order,
                node_coords
            )
            print(
                f"[PROFILE] tsp: "
                f"{time.perf_counter()-t1:.3f}s "
                f"nodes={len(node_coords)}"
            )

        ordered_nodes = [
            nodes[i]
            for i in order
        ]
        filtered_nodes = []

        MIN_NODE_DIST = 120

        for n in ordered_nodes:

            if not filtered_nodes:
                filtered_nodes.append(n)
                continue

            prev = filtered_nodes[-1]

            d = ox.distance.great_circle_vec(
                G.nodes[prev]["y"],
                G.nodes[prev]["x"],
                G.nodes[n]["y"],
                G.nodes[n]["x"]
            )

            if d >= MIN_NODE_DIST:
                filtered_nodes.append(n)

        ordered_nodes = filtered_nodes
        if len(node_coords) > 0:

            center_x = np.mean([p[0] for p in node_coords])
            center_y = np.mean([p[1] for p in node_coords])

            start_node = ox.distance.nearest_nodes(
                G,
                center_x,
                center_y
            )

            cur = start_node

        else:

            cur = start
        for target in ordered_nodes:

            try:
                t1 = time.perf_counter()
                path = nx.shortest_path(
                    G,
                    cur,
                    target,
                    weight="length"
                )
                routing_time += (
                    time.perf_counter() - t1
                )
                routing_calls += 1
                marsh[agent_id].extend(path)

                cur = target

            except:
                continue
    print(
        f"[PROFILE] nearest_nodes: "
        f"{nearest_time:.3f}s"
    )
    print(
        f"[PROFILE] shortest_path: "
        f"{routing_time:.3f}s "
        f"calls={routing_calls}"
    )
    m = folium.Map(location=[shir, dolg], zoom_start=15)
    kartan = m.get_name()
    HeatMap(heat,radius=12,blur=10,min_opacity=0.2, gradient={0.2: "blue",0.4: "lime",0.6: "yellow",0.8: "orange",1.0: "red"}).add_to(m)
    for i, (lat, lon, w) in enumerate(TOP_POINTS):
        folium.Marker(
            [lat, lon],
            popup=f"Priority #{i+1}, p={w:.2f}",
            icon=folium.Icon(color="black", icon="star")
        ).add_to(m)
    # serv = "https://40b0-35-203-149-68.ngrok-free.app"
    rex = []

    for put in marsh:
        if not put:
            rex.append([])
            continue
        coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in put]
        rex.append(coords)
    print(
        "[PROFILE] "
        f"is_valid={profile['is_valid']:,} "
        f"is_path_valid={profile['is_path_valid']:,}"
    )
    print(
        f"[PROFILE] TOTAL: "
        f"{time.perf_counter()-full_t0:.3f}s"
    )
    return {
        "heat": heat,
        "routes": rex,
        "start": [shir, dolg],
        "top_points": TOP_POINTS
    }

