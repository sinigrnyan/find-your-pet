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

env_cache = {}
node_cache = {}
cveta = ['red', 'blue', 'green', 'purple', 'orange']
pov = {
    ("koshka", "truslivaya"): {"P_STAY": 0.18, "P_RETURN": 0.06, "MAX_STEP": 50.0, "EDGE_SCALE": 30.0, "AVOID_MAJOR_ROADS": True},
    ("koshka", "uverennaya"): {"P_STAY": 0.08, "P_RETURN": 0.03, "MAX_STEP": 80.0, "EDGE_SCALE": 55.0, "AVOID_MAJOR_ROADS": True},
    ("sobaka", "truslivaya"): {"P_STAY": 0.06, "P_RETURN": 0.03, "MAX_STEP": 90.0, "EDGE_SCALE": 70.0, "AVOID_MAJOR_ROADS": True},
    ("sobaka", "uverennaya"): {"P_STAY": 0.02, "P_RETURN": 0.01, "MAX_STEP": 150.0, "EDGE_SCALE": 120.0, "AVOID_MAJOR_ROADS": False}
}
ox.settings.use_cache = True
ox.settings.log_console = True
ox.settings.headers = {
    "User-Agent": "SinigrNyan (sinigrchannel@gmail.com)",
    "Referer": "https://2072-34-80-242-236.ngrok-free.app"
}

def generate_map(zhiv, khar, shir, dolg, rad, vol):
    global env_cache, node_cache
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
        lat, lon = to_latlon(x, y)
        p = Point(lon, lat)
        if blocked_union and blocked_union.contains(p):
            return False
        return True

    def is_path_valid(x1, y1, x2, y2):
        lat1, lon1 = to_latlon(x1, y1)
        lat2, lon2 = to_latlon(x2, y2)
        line = LineString([(lon1, lat1), (lon2, lat2)])
        if blocked_union and blocked_union.intersects(line):
            return False
        return True

    def mestnost(x, y, golod, stress):
        key = (int(x//10), int(y//10), int(golod*10), int(stress*10))
        if key in env_cache:
            return env_cache[key]
        lat, lon = to_latlon(x, y)
        p = Point(lon, lat)
        score = 1.0
        if walkable_union and walkable_union.contains(p):
            score *= 1.4 + 0.3 * stress
        if res_union and res_union.contains(p):
            score *= 1.6
        if parking_union and parking_union.contains(p):
            score *= 1.2 + golod
        if roads_union and roads_union.contains(p):
            score *= max(0.3, 1.0 - stress * 1.5)
        if blocked_geom:
            dist = blocked_geom.distance(p)
            if dist < 20:
                score *= 1.5 + 0.5 * stress
        env_cache[key] = score
        return score


    def shag(pos, predugl, golod, stress, zhiv1, khar1):
        x, y = pos
        best = None
        best_score = -1
        for _ in range(5):
            if zhiv1 == "koshka":
                ugl = predugl + rng.uniform(-0.4, 0.4)
                max_step = conf["MAX_STEP"]
            else:
                ugl = predugl + rng.uniform(-0.6, 0.6) if khar1 == "truslivaya" else predugl + rng.uniform(-0.3, 0.3)
                max_step = conf["MAX_STEP"]
            dist = rng.uniform(0.3 * max_step, max_step)
            nx = x + dist * math.cos(ugl)
            ny = y + dist * math.sin(ugl)
            if not is_valid(nx, ny) or not is_path_valid(x, y, nx, ny):
                continue
            d = math.hypot(nx - start_xy[0], ny - start_xy[1])
            kdomu = math.exp(-d / (rad * (0.6 if zhiv1 == "koshka" else 0.8)))
            env = mestnost(nx, ny, golod, stress)
            issled = 1.0 + (d / rad) * (0.3 if zhiv1 == "koshka" else 0.4)
            golod_drive = 1.0 + golod * (0.8 if zhiv1 == "koshka" else 1.0)
            stress_dom = 1.0 + stress * (1.2 if zhiv1 == "koshka" else 1.0)
            score = env * kdomu * issled * golod_drive * (1 / stress_dom)
            if score > best_score:
                best = (nx, ny, ugl)
                best_score = score
        if best:
            return (best[0], best[1]), best[2]
        return pos, predugl
    start_xy = to_xy(shir, dolg)
    G = ox.graph_from_point((shir, dolg), dist=rad, network_type="all", simplify=True)
    G = ox.distance.add_edge_lengths(G)
    start = ox.distance.nearest_nodes(G, dolg, shir)
    tags = {"landuse": True, "natural": True, "leisure": True, "highway": True, "amenity": True}
    gdf = ox.geometries_from_point((shir, dolg), tags=tags, dist=rad)
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
    vis = []
    t_global = 0
    GOLOD_MAX = 1.0
    STRESS_MAX = 1.0
    GOLOD_GROWTH = 0.004
    STRESS_UPAD = 0.01
    SHUM_STRESS = 0.05

    for _ in range(1500):
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
            return_prob = min(0.85, conf["P_RETURN"] * (1 + stress*2))
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
    xy = np.array([(x, y) for x, y, _ in vis])
    if len(xy) > 0:
        k = min(vol, len(xy))
        labels = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(xy)
    else:
        labels = []

    grid = defaultdict(float)
    cell = 1
    T = vis[-1][2] + 1 if vis else 1
    for x, y, t in vis:
        if not is_valid(x, y):
            continue
        w = (((t + 1) / T) ** 3) * (1 / (1 + ((math.hypot(x - start_xy[0], y - start_xy[1]))/ (rad * 0.4)) ** 2)) * rng.uniform(0.9, 1.1)
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
    och = []
    for (lat, lon, w) in heat:
        if w >= 0.03:
            och.append((lat, lon, w))
    if len(och) == 0:
        clust = [[] for _ in range(vol)]
    else:
        coords = np.array([to_xy(lat, lon) for lat, lon, _ in och])
        vesaw = np.ones(len(coords))
        k = min(vol, len(coords))
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = kmeans.fit_predict(coords, sample_weight=vesaw)
        clust = [[] for _ in range(k)]
        for idx, lb in enumerate(labels):
            clust[lb].append(och[idx])
        def cluster_cost(cluster):
            if len(cluster) < 2:
                return 0
            pts = np.array([to_xy(lat, lon) for lat, lon, _ in cluster])
            center = pts.mean(axis=0)
            return np.sum(np.linalg.norm(pts - center, axis=1))
        for _ in range(10):
            costs = [cluster_cost(c) for c in clust]
            hi = int(np.argmax(costs))
            lo = int(np.argmin(costs))
            if costs[hi] - costs[lo] < 50:
                break
            if len(clust[hi]) <= 1:
                break
            hi_pts = clust[hi]
            center = np.mean([to_xy(lat, lon) for lat, lon, _ in hi_pts], axis=0)
            dists = [
                (i, np.linalg.norm(np.array(to_xy(lat, lon)) - center))
                for i, (lat, lon, _) in enumerate(hi_pts)
            ]
            idx_move = max(dists, key=lambda x: x[1])[0]
            clust[lo].append(clust[hi].pop(idx_move))

    marsh = [[] for _ in range(vol)]
    for agent_id, points in enumerate(clust):
        nodes = []
        for lat, lon, _ in points:
            try:
                key = (round(lat, 5), round(lon, 5))
                if key in node_cache:
                    node = node_cache[key]
                else:
                    node = ox.distance.nearest_nodes(G, lon, lat)
                    node_cache[key] = node
                nodes.append(node)
            except:
                continue
        nodes = list(set(nodes))
        if not nodes:
            continue
        cluster_center = np.mean([[lat, lon] for lat, lon, _ in points], axis=0)
        komm_nodes = [ox.distance.nearest_nodes(G, cluster_center[1], cluster_center[0])] + nodes
        n = len(komm_nodes)
        dist = [[float('inf')] * n for _ in range(n)]
        all_lengths = {}
        for u in komm_nodes:
            all_lengths[u] = nx.single_source_dijkstra_path_length(G, u, weight="length")
        for i, u in enumerate(komm_nodes):
            for j, v in enumerate(komm_nodes):
                if v in all_lengths[u]:
                    dist[i][j] = all_lengths[u][v]
        nepose = set(range(1, n))
        tour = [0]
        cur = 0
        while nepose:
            nxt = min(nepose, key=lambda j: dist[cur][j])
            tour.append(nxt)
            nepose.remove(nxt)
            cur = nxt
        def dlin(t):
            return sum(dist[t[i]][t[i+1]] for i in range(len(t)-1))
        luchs = True
        while luchs:
            luchs = False
            for i in range(len(tour) - 2):
                for j in range(i + 2, len(tour)):
                    new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
                    if dlin(new_tour) < dlin(tour):
                        tour = new_tour
                        luchs = True
                        break
                if luchs:
                    break
        cur = start
        for idx in tour[1:]:
            target = komm_nodes[idx]
            try:
                path = nx.shortest_path(G, cur, target, weight="length")
                marsh[agent_id].extend(path)
                cur = target
            except:
                continue
        if not marsh[agent_id]:
            marsh[agent_id] = [start]

    m = folium.Map(location=[shir, dolg], zoom_start=15)
    kartan = m.get_name()
    HeatMap(heat,radius=12,blur=10,min_opacity=0.2, gradient={0.2: "blue",0.4: "lime",0.6: "yellow",0.8: "orange",1.0: "red"}).add_to(m)
    for i, (lat, lon, w) in enumerate(TOP_POINTS):
        folium.Marker(
            [lat, lon],
            popup=f"Priority #{i+1}, p={w:.2f}",
            icon=folium.Icon(color="black", icon="star")
        ).add_to(m)
    serv = "https://40b0-35-203-149-68.ngrok-free.app"
    rex = []

    for put in marsh:
        if not put:
            rex.append([])
            continue
        coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in put]
        rex.append(coords)

    return {
        "heat": heat,
        "routes": rex,
        "start": [shir, dolg],
        "top_points": TOP_POINTS
    }

