import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

BASE_URL = "https://opendataapi.dmi.dk/v1/forecastedr"
COLLECTION = "harmonie_ig_sf"

LOCATION_NAME = "TASIILAQ"
DATA_FILE = Path("data.json")
HISTORY_FILE = Path("history.json")

REQUEST_TIMEOUT = 20
REQUEST_SLEEP = 0.5
REQUEST_RETRIES = 4
TIME_TOLERANCE_HOURS = 2.5

ICE_PRESSURE_NORMAL_HPA = 1013.25

# Punkter vi vil lese ut fra bbox-gridet
ICE_TARGETS = {
    "core": (-39.5, 69.3),
    "corridor": (-41.5, 68.1),
}

SEA_TARGETS = {
    "SC": (-29.0, 65.0),
    "C": (-27.0, 66.0),
    "N": (-24.8, 67.0),
}

# Litt romslige bokser rundt punktene
ICE_BBOX = (-42.2, 67.6, -38.6, 69.8)   # minLon, minLat, maxLon, maxLat
SEA_BBOX = (-30.5, 64.4, -24.0, 67.4)

ICE_PARAMS = ["pressure-sealevel", "temperature-2m", "wind-speed-100m", "latitude", "longitude"]
SEA_PARAMS = ["pressure-sealevel", "temperature-2m", "latitude", "longitude"]


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def norm(x, lo, hi):
    if hi == lo:
        return 0.0
    return clamp((x - lo) / (hi - lo), 0.0, 1.0)


def avg(values):
    return sum(values) / len(values) if values else 0.0


def kelvin_to_celsius(k):
    return k - 273.15


def parse_iso(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def iso_z(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def get_json(url, params=None, retries=REQUEST_RETRIES):
    last_err = None
    for attempt in range(retries):
        try:
            time.sleep(REQUEST_SLEEP)
            r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code == 429:
                wait = 5 + attempt * 3
                print(f"429 from DMI, waiting {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(1.0 + attempt)
    raise last_err


def list_instances():
    url = f"{BASE_URL}/collections/{COLLECTION}/instances"
    data = get_json(url)
    ids = []

    if isinstance(data, dict):
        if isinstance(data.get("instances"), list):
            for item in data["instances"]:
                if isinstance(item, dict):
                    iid = item.get("id") or item.get("instanceId")
                    if iid:
                        ids.append(iid)

        if not ids and isinstance(data.get("features"), list):
            for feat in data["features"]:
                iid = feat.get("id")
                if iid:
                    ids.append(iid)

        if not ids and isinstance(data.get("links"), list):
            for link in data["links"]:
                href = link.get("href", "")
                if "/instances/" in href:
                    iid = href.rstrip("/").split("/instances/")[-1].split("/")[0]
                    if iid:
                        ids.append(iid)

    ids = sorted(set(ids))
    if not ids:
        raise RuntimeError("Fant ingen DMI instanceId-er.")
    return ids[-3:]


def fetch_bbox(instance_id, bbox, params):
    min_lon, min_lat, max_lon, max_lat = bbox
    url = f"{BASE_URL}/collections/{COLLECTION}/instances/{instance_id}/bbox"
    qp = {
        "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "parameter-name": ",".join(params),
        "crs": "crs84",
        "f": "CoverageJSON",
    }
    return get_json(url, params=qp)


def parse_covjson(data, param_names):
    domain = data.get("domain", {})
    axes = domain.get("axes", {})
    times = axes.get("t", {}).get("values", [])
    xs = axes.get("x", {}).get("values", [])
    ys = axes.get("y", {}).get("values", [])
    ranges = data.get("ranges", {})
    values = {p: ranges.get(p, {}).get("values", []) for p in param_names}
    return times, xs, ys, values


def nearest_time_index(times, target_dt):
    if not times:
        return None
    parsed = [parse_iso(t) for t in times]
    best_i = None
    best_diff = None
    for i, dt in enumerate(parsed):
        diff_h = abs((dt - target_dt).total_seconds()) / 3600.0
        if best_diff is None or diff_h < best_diff:
            best_i = i
            best_diff = diff_h
    if best_diff is None or best_diff > TIME_TOLERANCE_HOURS:
        return None
    return best_i


def pick_target_indices(lat_vals, lon_vals, target_map):
    """
    lat_vals / lon_vals: flat lister for bbox-grid.
    target_map: {"name": (lon, lat)}
    """
    out = {}
    n = min(len(lat_vals), len(lon_vals))
    for name, (t_lon, t_lat) in target_map.items():
        best_i = None
        best_d2 = None
        for i in range(n):
            lat = lat_vals[i]
            lon = lon_vals[i]
            if lat is None or lon is None:
                continue
            d2 = (lat - t_lat) ** 2 + (lon - t_lon) ** 2
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_i = i
        out[name] = best_i
    return out


def value_at(flat_values, point_index, time_index, n_points):
    if point_index is None or time_index is None:
        return None
    idx = time_index * n_points + point_index
    if idx < 0 or idx >= len(flat_values):
        return None
    return flat_values[idx]


def extract_fields_at(dataset, target_indices, time_index, point_type):
    times, xs, ys, values = dataset
    n_points = len(values.get("latitude", []))
    if n_points == 0:
        return None

    if point_type == "ice":
        pressures = []
        temps = []
        winds = []

        for name in ["core", "corridor"]:
            pi = target_indices.get(name)
            p = value_at(values.get("pressure-sealevel", []), pi, time_index, n_points)
            t = value_at(values.get("temperature-2m", []), pi, time_index, n_points)
            w = value_at(values.get("wind-speed-100m", []), pi, time_index, n_points)

            if p is not None:
                pressures.append(p / 100.0)
            if t is not None:
                temps.append(kelvin_to_celsius(t))
            if w is not None:
                winds.append(w)

        if not pressures:
            return None

        ice_pressure = 0.55 * pressures[0] + 0.45 * pressures[-1] if len(pressures) == 2 else pressures[0]
        ice_wind = 0.55 * winds[0] + 0.45 * winds[-1] if len(winds) == 2 else (winds[0] if winds else 0.0)

        return {
            "pressure": ice_pressure,
            "tempC": avg(temps),
            "wind": ice_wind,
            "count": len(pressures),
        }

    if point_type == "sea":
        candidates = []
        temps = []

        for name in ["SC", "C", "N"]:
            pi = target_indices.get(name)
            p = value_at(values.get("pressure-sealevel", []), pi, time_index, n_points)
            t = value_at(values.get("temperature-2m", []), pi, time_index, n_points)

            if p is not None:
                candidates.append((p / 100.0, name))
            if t is not None:
                temps.append(kelvin_to_celsius(t))

        if not candidates:
            return None

        sea_pressure, sector = min(candidates, key=lambda x: x[0])

        return {
            "pressure": sea_pressure,
            "tempC": avg(temps),
            "sector": sector,
            "count": len(candidates),
        }

    return None


def collect_all_bbox_data(instances):
    cache = {}
    for iid in instances:
        print(f"Fetching bbox data for {iid}")
        ice_data = fetch_bbox(iid, ICE_BBOX, ICE_PARAMS)
        sea_data = fetch_bbox(iid, SEA_BBOX, SEA_PARAMS)
        cache[iid] = {
            "ice": parse_covjson(ice_data, ICE_PARAMS),
            "sea": parse_covjson(sea_data, SEA_PARAMS),
        }
    return cache


def build_time_pool(cache):
    pool = []
    for iid, block in cache.items():
        times = block["ice"][0]
        for t in times:
            pool.append(
                {
                    "instanceId": iid,
                    "validTime": t,
                    "dt": parse_iso(t),
                }
            )
    uniq = {}
    for item in pool:
        uniq[(item["instanceId"], item["validTime"])] = item
    return sorted(uniq.values(), key=lambda x: x["dt"])


def choose_trend_targets(pool):
    now_dt = datetime.now(timezone.utc)
    targets = {
        "now": now_dt,
        "m6": now_dt - timedelta(hours=6),
        "m12": now_dt - timedelta(hours=12),
    }

    chosen = {}
    used = set()

    for key in ["now", "m6", "m12"]:
        best = None
        best_diff = None
        for c in pool:
            k = (c["instanceId"], c["validTime"])
            if k in used:
                continue
            diff_h = abs((c["dt"] - targets[key]).total_seconds()) / 3600.0
            if diff_h > TIME_TOLERANCE_HOURS:
                continue
            if best is None or diff_h < best_diff:
                best = c
                best_diff = diff_h
        chosen[key] = best
        if best is not None:
            used.add((best["instanceId"], best["validTime"]))

    count = sum(1 for v in chosen.values() if v is not None)
    if count == 3:
        status = "ok"
    elif count == 2:
        status = "partial"
    else:
        status = "insufficient_distinct_steps"

    return chosen, status


def history_prev(history, hours_back):
    if len(history) >= hours_back:
        return history[-hours_back]
    return None


def classify_risk(score):
    if score >= 75:
        return "RED", "PITERAQ RELEASE", "0-6T"
    if score >= 55:
        return "ORG", "PITERAQ LIKELY", "6-12T"
    if score >= 35:
        return "YEL", "PITERAQ BUILDING", "12-24T"
    return "GRN", "PITERAQ LOW", "12-24T"


def gradient_boost(gradient_hpa):
    return norm(gradient_hpa, 20, 40)


def potential_index(reservoir, coupling, gradient, d6, ice_wind_trend_6h):
    return 100 * (
        0.35 * (reservoir / 100.0)
        + 0.20 * (coupling / 100.0)
        + 0.25 * norm(gradient, 15, 40)
        + 0.10 * norm(d6, 0, 8)
        + 0.10 * norm(ice_wind_trend_6h, 0, 6)
    )


def main():
    now_dt = datetime.now(timezone.utc)
    now_str = now_dt.strftime("%Y-%m-%d %H:%M UTC")
    history = load_json(HISTORY_FILE, [])

    instances = list_instances()
    latest = instances[-1]

    cache = collect_all_bbox_data(instances)

    # Finn grid-indekser én gang per bbox
    ice_lat = cache[latest]["ice"][3]["latitude"]
    ice_lon = cache[latest]["ice"][3]["longitude"]
    sea_lat = cache[latest]["sea"][3]["latitude"]
    sea_lon = cache[latest]["sea"][3]["longitude"]

    ice_target_indices = pick_target_indices(ice_lat, ice_lon, ICE_TARGETS)
    sea_target_indices = pick_target_indices(sea_lat, sea_lon, SEA_TARGETS)

    pool = build_time_pool(cache)
    chosen, trend_status = choose_trend_targets(pool)

    if chosen["now"] is None:
        raise RuntimeError("Fant ikke gyldig now-tid i bbox-data.")

    def fields_for(choice):
        if choice is None:
            return None
        iid = choice["instanceId"]
        vt = choice["validTime"]

        ice_times = cache[iid]["ice"][0]
        sea_times = cache[iid]["sea"][0]

        ti_ice = nearest_time_index(ice_times, parse_iso(vt))
        ti_sea = nearest_time_index(sea_times, parse_iso(vt))

        ice_fields = extract_fields_at(cache[iid]["ice"], ice_target_indices, ti_ice, "ice")
        sea_fields = extract_fields_at(cache[iid]["sea"], sea_target_indices, ti_sea, "sea")

        if ice_fields is None or sea_fields is None:
            return None

        return {
            "icePressure": ice_fields["pressure"],
            "iceTempC": ice_fields["tempC"],
            "iceWind": ice_fields["wind"],
            "seaPressure": sea_fields["pressure"],
            "seaTempC": sea_fields["tempC"],
            "sector": sea_fields["sector"],
            "usedReservoirPoints": ice_fields["count"],
            "usedIcePoints": ice_fields["count"],
            "usedSeaPoints": sea_fields["count"],
        }

    now_fields = fields_for(chosen["now"])
    m6_fields = fields_for(chosen["m6"])
    m12_fields = fields_for(chosen["m12"])

    if now_fields is None:
        raise RuntimeError("Kunne ikke lese now-felter fra bbox-data.")

    ice_pressure = now_fields["icePressure"]
    ice_temp_c = now_fields["iceTempC"]
    ice_wind = now_fields["iceWind"]
    sea_pressure = now_fields["seaPressure"]
    sea_temp_c = now_fields["seaTempC"]
    sector = now_fields["sector"]

    gradient = ice_pressure - sea_pressure
    gradient_m6 = (m6_fields["icePressure"] - m6_fields["seaPressure"]) if m6_fields else None
    gradient_m12 = (m12_fields["icePressure"] - m12_fields["seaPressure"]) if m12_fields else None

    d6 = (gradient - gradient_m6) if gradient_m6 is not None else 0.0
    d12 = (gradient - gradient_m12) if gradient_m12 is not None else 0.0
    sf6 = (m6_fields["seaPressure"] - sea_pressure) if m6_fields else 0.0
    sf12 = (m12_fields["seaPressure"] - sea_pressure) if m12_fields else 0.0

    acc_g = d6 - (d12 - d6) if m6_fields and m12_fields else 0.0
    acc_s = sf6 - (sf12 - sf6) if m6_fields and m12_fields else 0.0
    ice_wind_trend_6h = (ice_wind - m6_fields["iceWind"]) if m6_fields else 0.0

    ice_pressure_anom_now = ice_pressure - ICE_PRESSURE_NORMAL_HPA
    dT_coast_ice = max(0.0, sea_temp_c - ice_temp_c)
    cold_support_now = max(0.0, -ice_temp_c)
    sea_low_depth = max(0.0, 1000.0 - sea_pressure)

    snapshot = {
        "t": now_dt.isoformat(),
        "icePressure": round(ice_pressure, 1),
        "seaPressure": round(sea_pressure, 1),
        "gradient": round(gradient, 1),
        "iceWind": round(ice_wind, 1),
        "icePressureAnomNow": round(ice_pressure_anom_now, 1),
        "coldSupportNow": round(cold_support_now, 1),
        "dTCoastIceNow": round(dT_coast_ice, 1),
        "sector": sector,
    }

    history.append(snapshot)
    history = history[-72:]
    save_json(HISTORY_FILE, history)

    prev_24h = history_prev(history[:-1], 24)
    prev_72h = history_prev(history[:-1], 72)

    ice_24h_ago = float(prev_24h["icePressure"]) if prev_24h else ice_pressure
    ice_72h_ago = float(prev_72h["icePressure"]) if prev_72h else ice_pressure

    ice_pressure_trend_24h = ice_pressure - ice_24h_ago
    ice_pressure_trend_72h = ice_pressure - ice_72h_ago

    ice_anom_72 = [float(x.get("icePressureAnomNow", 0.0)) for x in history]
    cold_72 = [float(x.get("coldSupportNow", 0.0)) for x in history]
    dT_72 = [float(x.get("dTCoastIceNow", 0.0)) for x in history]

    ice_anom_72_mean = avg(ice_anom_72)
    cold_72_mean = avg(cold_72)
    dT_72_mean = avg(dT_72)

    reservoir = 100 * (
        0.55 * norm(ice_anom_72_mean, -12, 12)
        + 0.20 * norm(ice_pressure_anom_now, -12, 12)
        + 0.15 * norm(ice_pressure_trend_24h, -3, 8)
        + 0.10 * norm(ice_pressure_trend_72h, -5, 12)
    )
    reservoir = clamp(reservoir, 0, 100)
    if ice_anom_72_mean <= -8:
        reservoir = min(reservoir, 39)

    sector_score = {"SC": 90, "C": 75, "N": 40}.get(sector, 25)
    coupling = 100 * (
        0.60 * (sector_score / 100.0)
        + 0.25 * norm(sea_low_depth, 5, 30)
        + 0.15 * norm(ice_wind, 4, 20)
    )
    coupling = clamp(coupling, 0, 100)

    thermal_component = 100 * (
        0.55 * norm(dT_72_mean, 3, 20)
        + 0.30 * norm(cold_72_mean, 5, 25)
        + 0.15 * norm(dT_coast_ice, 10, 35)
    )
    thermal_component = clamp(thermal_component, 0, 100)
    reservoir_factor = 0.35 + 0.65 * (reservoir / 100.0)
    katabatic_potential = clamp(thermal_component * reservoir_factor, 0, 100)

    gboost = gradient_boost(gradient)
    trigger = 100 * (
        0.28 * norm(gradient, 0, 65)
        + 0.10 * gboost
        + 0.16 * norm(sf6, 0, 10)
        + 0.10 * norm(sf12, 0, 14)
        + 0.12 * norm(d6, 0, 10)
        + 0.08 * norm(d12, 0, 14)
        + 0.07 * norm(acc_g, 0, 6)
        + 0.04 * norm(acc_s, 0, 6)
        + 0.05 * norm(ice_wind_trend_6h, 0, 6)
    )
    trigger = clamp(trigger, 0, 100)

    potential = clamp(
        potential_index(reservoir, coupling, gradient, d6, ice_wind_trend_6h),
        0,
        100,
    )

    watch = (
        (reservoir >= 30 or potential >= 45)
        and coupling >= 60
        and (
            gradient >= 20
            or d6 >= 2
            or sf6 >= 2
            or acc_g >= 1
            or ice_wind_trend_6h >= 2
        )
    )

    base = 0.58 * trigger + 0.32 * reservoir + 0.10 * potential
    risk = base * (0.60 + 0.40 * (coupling / 100.0))

    if trigger < 20:
        risk = min(risk, 34)
    if trigger < 35 and reservoir < 25:
        risk = min(risk, 24)

    level, phase, horizon = classify_risk(risk)
    if watch and level == "GRN":
        phase = "WATCH"

    trend_tag = "" if trend_status == "ok" else " T?"
    ag_code = int(round(clamp(acc_g, 0, 9)))
    as_code = int(round(clamp(acc_s, 0, 9)))

    message = (
        f"{LOCATION_NAME} {level} {horizon}{trend_tag} "
        f"RES{int(round(reservoir))} TRG{int(round(trigger))} CPL{int(round(coupling))} "
        f"ICE{ice_pressure:.1f} SEA{sea_pressure:.1f} "
        f"GR{gradient:.1f} d6{d6:+.1f} d12{d12:+.1f} "
        f"SF6{sf6:+.1f} SF12{sf12:+.1f} DT{dT_coast_ice:.0f} "
        f"AG{ag_code} AS{as_code}"
    )

    payload = {
        "meta": {
            "source": "DMI HARMONIE",
            "updatedAt": now_str,
            "location": LOCATION_NAME,
            "model": COLLECTION,
            "forecastInfo": "Latest available forecast step",
            "instanceId": latest,
        },
        "inputs": {
            "icePressure": round(ice_pressure, 1),
            "seaPressure": round(sea_pressure, 1),
            "gradient": round(gradient, 1),
            "d6": round(d6, 1),
            "d12": round(d12, 1),
            "sf6": round(sf6, 1),
            "sf12": round(sf12, 1),
            "iceWind": round(ice_wind, 1),
            "iceWindTrend6h": round(ice_wind_trend_6h, 1),
            "coastIceDeltaT": round(dT_coast_ice, 1),
        },
        "scores": {
            "reservoir": int(round(reservoir)),
            "trigger": int(round(trigger)),
            "coupling": int(round(coupling)),
            "potential": int(round(potential)),
            "risk": int(round(risk)),
        },
        "derived": {
            "watch": watch,
            "sector": sector,
            "icePressureAnomNow": round(ice_pressure_anom_now, 1),
            "icePressureAnom72hMean": round(ice_anom_72_mean, 1),
            "icePressureTrend24h": round(ice_pressure_trend_24h, 1),
            "icePressureTrend72h": round(ice_pressure_trend_72h, 1),
            "coldSupport72h": round(cold_72_mean, 1),
            "katabaticPotential": int(round(katabatic_potential)),
            "seaLowDepth": round(sea_low_depth, 1),
            "gradientBoost": round(gboost, 2),
            "accG": round(acc_g, 1),
            "accS": round(acc_s, 1),
            "trendDataStatus": trend_status,
            "selectedTimes": {
                k: None if v is None else {
                    "instanceId": v["instanceId"],
                    "validTime": v["validTime"],
                }
                for k, v in chosen.items()
            },
            "usedReservoirPoints": now_fields["usedReservoirPoints"],
            "usedIcePoints": now_fields["usedIcePoints"],
            "usedSeaPoints": now_fields["usedSeaPoints"],
        },
        "output": {
            "level": level,
            "phase": phase,
            "message": message,
        },
    }

    save_json(DATA_FILE, payload)
    print("Updated data.json/history.json")
    print("trendDataStatus:", trend_status)


if __name__ == "__main__":
    main()
