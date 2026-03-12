import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

BASE = "https://opendataapi.dmi.dk/v1/forecastedr"
COL = "harmonie_ig_sf"

LOCATION_NAME = "TASIILAQ"
DATA_FILE = Path("data.json")
HISTORY_FILE = Path("history.json")

CONNECT_TIMEOUT = 8
READ_TIMEOUT = 40
REQUEST_SLEEP = 0.2
REQUEST_RETRIES = 3
MAX_WORKERS = 3

TIME_TOLERANCE_HOURS = 2.75
HISTORY_KEEP = 144  # 6 døgn ved timekjøring

ICE_PRESSURE_NORMAL_HPA = 1013.25

ICE_POINTS = [
    {"name": "source", "lon": -42.4, "lat": 69.0},
    {"name": "mid", "lon": -41.3, "lat": 68.6},
    {"name": "mouth", "lon": -40.3, "lat": 68.2},
]

SEA_POINTS = [
    {"name": "W1", "lon": -34.9, "lat": 62.8},
    {"name": "W2", "lon": -33.7, "lat": 63.8},
    {"name": "W3", "lon": -32.5, "lat": 64.8},
    {"name": "C1", "lon": -30.5, "lat": 64.6},
    {"name": "C2", "lon": -31.1, "lat": 65.4},
    {"name": "M1", "lon": -30.1, "lat": 63.9},
    {"name": "M2", "lon": -31.8, "lat": 64.3},
    {"name": "M3", "lon": -30.2, "lat": 65.0},
    {"name": "E1", "lon": -29.1, "lat": 65.8},
    {"name": "E2", "lon": -27.5, "lat": 66.4},
]

COAST_POINTS = [
    {"name": "K1", "lon": -38.8, "lat": 65.7},
    {"name": "K2", "lon": -37.6, "lat": 66.5},
]

WEST_NAMES = ["W1", "W2", "W3"]
MID_NAMES = ["C1", "C2", "M1", "M2", "M3"]
EAST_NAMES = ["E1", "E2"]
ALL_STRAIT_NAMES = WEST_NAMES + MID_NAMES + EAST_NAMES

ICE_PARAMS = ["pressure-sealevel", "temperature-2m", "wind-speed-100m"]
SEA_PARAMS = ["pressure-sealevel", "temperature-2m"]

TREND_TOLERANCES = {
    "h6": timedelta(hours=1.5),
    "h12": timedelta(hours=2.0),
    "h24": timedelta(hours=3.0),
    "h72": timedelta(hours=6.0),
}


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def norm(x, lo, hi):
    if x is None or hi == lo:
        return 0.0
    return clamp((x - lo) / (hi - lo), 0.0, 1.0)


def avg(values):
    vals = [v for v in values if is_num(v)]
    return sum(vals) / len(vals) if vals else None


def kelvin_to_celsius(k):
    return k - 273.15


def parse_iso(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def iso_z(dt):
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def now_utc():
    return datetime.now(timezone.utc)


def now_utc_str():
    return now_utc().strftime("%Y-%m-%d %H:%M UTC")


def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def safe_get(arr, idx):
    if not isinstance(arr, list):
        return None
    if idx < 0 or idx >= len(arr):
        return None
    return arr[idx]


def load_json(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fmt_msg_num(x, digits=1, signed=False, none_text="NA"):
    if not is_num(x):
        return none_text
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


def compact_score_tag(prefix, value, uncertain=False):
    if not is_num(value):
        return f"{prefix}?"
    iv = int(round(value))
    return f"{prefix}{iv}{'?' if uncertain else ''}"


def compact_temp_trend_tag(prefix, value):
    if not is_num(value):
        return f"{prefix}?"
    iv = int(round(value))
    return f"{prefix}{iv:+d}"


def compact_gate_tag(prefix, value):
    if not is_num(value):
        return f"{prefix}?"
    return f"{prefix}{int(round(value))}"


def get_json(url, params=None, retries=REQUEST_RETRIES):
    last_err = None

    for attempt in range(retries):
        try:
            time.sleep(REQUEST_SLEEP)
            r = requests.get(url, params=params, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))

            if r.status_code == 429:
                wait = 4 + attempt * 6
                print(f"DMI rate limit (429). Waiting {wait}s...")
                time.sleep(wait)
                continue

            r.raise_for_status()
            return r.json()

        except requests.exceptions.ReadTimeout as e:
            last_err = e
            wait = 2 + attempt * 3
            print(f"DMI read timeout. Waiting {wait}s before retry...")
            time.sleep(wait)

        except requests.exceptions.ConnectTimeout as e:
            last_err = e
            wait = 2 + attempt * 3
            print(f"DMI connect timeout. Waiting {wait}s before retry...")
            time.sleep(wait)

        except Exception as e:
            last_err = e
            wait = 1 + attempt * 2
            print(f"DMI request failed: {type(e).__name__}. Waiting {wait}s...")
            time.sleep(wait)

    raise last_err


def list_instances():
    url = f"{BASE}/collections/{COL}/instances"
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
    return ids


def fetch_position(instance_id, lon, lat, parameter_names):
    url = f"{BASE}/collections/{COL}/instances/{instance_id}/position"
    params = {
        "coords": f"POINT({lon} {lat})",
        "parameter-name": ",".join(parameter_names),
        "crs": "crs84",
        "f": "CoverageJSON",
    }
    return get_json(url, params=params)


def parse_coverage_series(data, parameter_names):
    domain = data.get("domain", {})
    axes = domain.get("axes", {})
    times = axes.get("t", {}).get("values", [])
    ranges = data.get("ranges", {})
    values = {p: ranges.get(p, {}).get("values", []) for p in parameter_names}
    return times, values


def build_empty_cache():
    merged = {"ice": {}, "sea": {}}

    for p in ICE_POINTS:
        merged["ice"][p["name"]] = {
            "lon": p["lon"],
            "lat": p["lat"],
            "rows": [],
        }

    for p in SEA_POINTS + COAST_POINTS:
        merged["sea"][p["name"]] = {
            "lon": p["lon"],
            "lat": p["lat"],
            "rows": [],
        }

    return merged


def fetch_points_parallel(instance_id, points, parameter_names, max_workers=MAX_WORKERS):
    results = {}
    errors = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_position, instance_id, p["lon"], p["lat"], parameter_names): p["name"]
            for p in points
        }

        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                results[name] = None
                errors[name] = f"{type(e).__name__}: {e}"

    return results, errors


def append_instance_to_cache(cache, iid):
    print(f"Fetching DMI series for instance {iid} with parallel point requests")

    ice_results, ice_errors = fetch_points_parallel(
        iid, ICE_POINTS, ICE_PARAMS, max_workers=min(MAX_WORKERS, len(ICE_POINTS))
    )
    sea_results, sea_errors = fetch_points_parallel(
        iid, SEA_POINTS + COAST_POINTS, SEA_PARAMS, max_workers=MAX_WORKERS
    )

    all_errors = {}
    all_errors.update({f"ice:{k}": v for k, v in ice_errors.items()})
    all_errors.update({f"sea:{k}": v for k, v in sea_errors.items()})

    for p in ICE_POINTS:
        data = ice_results.get(p["name"])
        if not isinstance(data, dict):
            continue
        times, values = parse_coverage_series(data, ICE_PARAMS)
        rows = cache["ice"][p["name"]]["rows"]

        for i, t in enumerate(times):
            rows.append(
                {
                    "instanceId": iid,
                    "validTime": t,
                    "dt": parse_iso(t),
                    "pressure-sealevel": safe_get(values.get("pressure-sealevel", []), i),
                    "temperature-2m": safe_get(values.get("temperature-2m", []), i),
                    "wind-speed-100m": safe_get(values.get("wind-speed-100m", []), i),
                }
            )

    for p in SEA_POINTS + COAST_POINTS:
        data = sea_results.get(p["name"])
        if not isinstance(data, dict):
            continue
        times, values = parse_coverage_series(data, SEA_PARAMS)
        rows = cache["sea"][p["name"]]["rows"]

        for i, t in enumerate(times):
            rows.append(
                {
                    "instanceId": iid,
                    "validTime": t,
                    "dt": parse_iso(t),
                    "pressure-sealevel": safe_get(values.get("pressure-sealevel", []), i),
                    "temperature-2m": safe_get(values.get("temperature-2m", []), i),
                }
            )

    for block_type in ["ice", "sea"]:
        for block in cache[block_type].values():
            dedup = {}
            for row in block["rows"]:
                dedup[row["validTime"]] = row
            block["rows"] = sorted(dedup.values(), key=lambda x: x["dt"])

    return all_errors


def build_master_time_axis(cache):
    dedup = {}
    for block_type in ["ice", "sea"]:
        for block in cache[block_type].values():
            for row in block["rows"]:
                dedup[row["validTime"]] = row["dt"]

    axis = [{"validTime": t, "dt": dt} for t, dt in dedup.items()]
    axis.sort(key=lambda x: x["dt"])
    return axis


def find_valid_time(master_axis, target_dt, tolerance_hours=TIME_TOLERANCE_HOURS):
    if not master_axis:
        return None, None, None

    best = None
    best_diff = None

    for item in master_axis:
        diff_h = abs((item["dt"] - target_dt).total_seconds()) / 3600.0
        if best is None or diff_h < best_diff:
            best = item
            best_diff = diff_h

    if best is None or best_diff is None or best_diff > tolerance_hours:
        return None, None, None

    return best["validTime"], best["dt"], best_diff


def get_row_for_valid_time(cache_block, point_name, valid_time):
    for row in cache_block[point_name]["rows"]:
        if row["validTime"] == valid_time:
            return row
    return None


def weighted_mean(vals, weights):
    pairs = [(v, w) for v, w in zip(vals, weights) if is_num(v)]
    if not pairs:
        return None
    sw = sum(w for _, w in pairs)
    if sw == 0:
        return None
    return sum(v * w for v, w in pairs) / sw


def centroid_low(candidates):
    valid = [c for c in candidates if is_num(c.get("pressure"))]
    if not valid:
        return None, None, None, None

    pmax = max(c["pressure"] for c in valid)
    weights = [max(0.1, pmax - c["pressure"] + 0.1) for c in valid]
    sw = sum(weights)

    clon = sum(c["lon"] * w for c, w in zip(valid, weights)) / sw
    clat = sum(c["lat"] * w for c, w in zip(valid, weights)) / sw
    cp = sum(c["pressure"] * w for c, w in zip(valid, weights)) / sw

    nearest_name = None
    nearest_d2 = None
    for c in valid:
        d2 = (c["lon"] - clon) ** 2 + (c["lat"] - clat) ** 2
        if nearest_d2 is None or d2 < nearest_d2:
            nearest_d2 = d2
            nearest_name = c["name"]

    return cp, clon, clat, nearest_name


def mean_pressure_for_names(cache, valid_time, names):
    vals = []
    for name in names:
        row = get_row_for_valid_time(cache["sea"], name, valid_time)
        if not row:
            continue
        p = row.get("pressure-sealevel")
        if is_num(p):
            vals.append(p / 100.0)
    return avg(vals)


def fields_for_valid_time(cache, valid_time):
    if valid_time is None:
        return None

    pressure_vals = []
    temp_vals = []
    wind_vals = []
    quality_flags = []

    for name in ["source", "mid", "mouth"]:
        row = get_row_for_valid_time(cache["ice"], name, valid_time)
        if not row:
            pressure_vals.append(None)
            temp_vals.append(None)
            wind_vals.append(None)
            continue

        p = row.get("pressure-sealevel")
        t = row.get("temperature-2m")
        w = row.get("wind-speed-100m")

        pressure_vals.append((p / 100.0) if is_num(p) else None)
        temp_vals.append(kelvin_to_celsius(t) if is_num(t) else None)
        wind_vals.append(w if is_num(w) else None)

    if sum(1 for v in pressure_vals if is_num(v)) == 0:
        return None

    ice_pressure = weighted_mean(pressure_vals, [0.40, 0.35, 0.25])
    ice_temp_c = weighted_mean(temp_vals, [0.45, 0.35, 0.20])
    ice_wind = weighted_mean(wind_vals, [0.20, 0.35, 0.45])

    if not is_num(ice_temp_c):
        quality_flags.append("missing_ice_temperature_all")
        ice_temp_c = None
    elif sum(1 for v in temp_vals if is_num(v)) < 3:
        quality_flags.append("missing_ice_temperature_partial")

    if not is_num(ice_wind):
        quality_flags.append("missing_ice_wind_all")
        ice_wind = None
    elif sum(1 for v in wind_vals if is_num(v)) < 3:
        quality_flags.append("missing_ice_wind_partial")

    sea_candidates = []
    sea_temps = []

    for name in ALL_STRAIT_NAMES:
        row = get_row_for_valid_time(cache["sea"], name, valid_time)
        if not row:
            continue

        p = row.get("pressure-sealevel")
        t = row.get("temperature-2m")
        meta = cache["sea"][name]

        if is_num(p):
            sea_candidates.append(
                {
                    "name": name,
                    "pressure": p / 100.0,
                    "lon": meta["lon"],
                    "lat": meta["lat"],
                }
            )

        if is_num(t):
            sea_temps.append(kelvin_to_celsius(t))

    for name in ["K1", "K2"]:
        row = get_row_for_valid_time(cache["sea"], name, valid_time)
        if row and is_num(row.get("temperature-2m")):
            sea_temps.append(kelvin_to_celsius(row.get("temperature-2m")))

    if not sea_candidates:
        return None

    sea_min = min(sea_candidates, key=lambda x: x["pressure"])
    centroid_pressure, centroid_lon, centroid_lat, centroid_sector = centroid_low(sea_candidates)

    spread = None
    if is_num(centroid_pressure):
        spread = centroid_pressure - sea_min["pressure"]
        if spread >= 3.5:
            quality_flags.append("sea_field_spread_high")
        elif spread >= 2.0:
            quality_flags.append("sea_field_spread_moderate")

    sea_temp_c = avg(sea_temps)
    if sea_temp_c is None:
        quality_flags.append("missing_sea_temperature_all")
    elif len(sea_temps) < (len(ALL_STRAIT_NAMES) + 2):
        quality_flags.append("missing_sea_temperature_partial")

    west_mean = mean_pressure_for_names(cache, valid_time, WEST_NAMES)
    mid_mean = mean_pressure_for_names(cache, valid_time, MID_NAMES)
    east_mean = mean_pressure_for_names(cache, valid_time, EAST_NAMES)
    coast_mean = mean_pressure_for_names(cache, valid_time, ["K1", "K2"])

    gate_mid = None
    gate_east = None
    vent_index = None

    if is_num(west_mean) and is_num(mid_mean):
        gate_mid = west_mean - mid_mean
    if is_num(west_mean) and is_num(east_mean):
        gate_east = west_mean - east_mean
    if is_num(gate_mid) or is_num(gate_east):
        vent_index = (0.6 * (gate_mid if is_num(gate_mid) else 0.0)) + (0.4 * (gate_east if is_num(gate_east) else 0.0))

    if not is_num(gate_mid):
        quality_flags.append("missing_gate_mid")
    if not is_num(gate_east):
        quality_flags.append("missing_gate_east")

    coast_gate = None
    if is_num(ice_pressure) and is_num(coast_mean):
        coast_gate = ice_pressure - coast_mean
    else:
        quality_flags.append("missing_coast_gate")

    return {
        "icePressure": ice_pressure,
        "iceTempC": ice_temp_c,
        "iceWind": ice_wind,
        "seaPressureMin": sea_min["pressure"],
        "seaMinSector": sea_min["name"],
        "seaMinLon": sea_min["lon"],
        "seaMinLat": sea_min["lat"],
        "seaCentroidPressure": centroid_pressure,
        "seaCentroidLon": centroid_lon,
        "seaCentroidLat": centroid_lat,
        "seaCentroidSector": centroid_sector,
        "seaMinCentroidSpread": spread,
        "seaTempC": sea_temp_c,
        "westMeanPressure": west_mean,
        "midMeanPressure": mid_mean,
        "eastMeanPressure": east_mean,
        "coastSeaPressure": coast_mean,
        "gateMid": gate_mid,
        "gateEast": gate_east,
        "ventilIndex": vent_index,
        "coastGate": coast_gate,
        "qualityFlags": sorted(set(quality_flags)),
        "usedIcePressurePoints": sum(1 for v in pressure_vals if is_num(v)),
        "usedIceTempPoints": sum(1 for v in temp_vals if is_num(v)),
        "usedIceWindPoints": sum(1 for v in wind_vals if is_num(v)),
        "usedSeaPressurePoints": len(sea_candidates),
        "usedSeaTempPoints": len(sea_temps),
    }


def build_snapshot(snapshot_dt, fields):
    ice_pressure = fields["icePressure"]
    ice_temp_c = fields["iceTempC"]
    sea_pressure = fields["seaPressureMin"]
    gradient = (ice_pressure - sea_pressure) if is_num(ice_pressure) and is_num(sea_pressure) else None
    ice_wind = fields["iceWind"]
    ice_pressure_anom_now = (ice_pressure - ICE_PRESSURE_NORMAL_HPA) if is_num(ice_pressure) else None
    cold_support_now = max(0.0, -ice_temp_c) if is_num(ice_temp_c) else None
    dT_coast_ice = (
        max(0.0, fields["seaTempC"] - ice_temp_c)
        if is_num(fields["seaTempC"]) and is_num(ice_temp_c)
        else None
    )

    return {
        "t": iso_z(snapshot_dt),
        "icePressure": round(ice_pressure, 1) if is_num(ice_pressure) else None,
        "iceTempC": round(ice_temp_c, 1) if is_num(ice_temp_c) else None,
        "seaPressure": round(sea_pressure, 1) if is_num(sea_pressure) else None,
        "gradient": round(gradient, 1) if is_num(gradient) else None,
        "iceWind": round(ice_wind, 1) if is_num(ice_wind) else None,
        "ventilIndex": round(fields["ventilIndex"], 1) if is_num(fields["ventilIndex"]) else None,
        "coastGate": round(fields["coastGate"], 1) if is_num(fields["coastGate"]) else None,
        "seaMinLon": round(fields["seaMinLon"], 3) if is_num(fields["seaMinLon"]) else None,
        "seaMinLat": round(fields["seaMinLat"], 3) if is_num(fields["seaMinLat"]) else None,
        "icePressureAnomNow": round(ice_pressure_anom_now, 1) if is_num(ice_pressure_anom_now) else None,
        "coldSupportNow": round(cold_support_now, 1) if is_num(cold_support_now) else None,
        "dTCoastIceNow": round(dT_coast_ice, 1) if is_num(dT_coast_ice) else None,
        "sector": fields["seaMinSector"],
    }


def sort_and_dedup_history(history):
    cleaned = []
    for item in history:
        try:
            dt = parse_iso(item["t"])
            cleaned.append((dt, item))
        except Exception:
            continue

    dedup = {}
    for dt, item in cleaned:
        dedup[item["t"]] = item

    out = sorted(dedup.values(), key=lambda x: parse_iso(x["t"]))
    return out[-HISTORY_KEEP:]


def find_history_snapshot(history, target_dt, tolerance, required_keys=None):
    required_keys = required_keys or []
    best = None
    best_diff = None

    for item in history:
        try:
            dt = parse_iso(item["t"])
        except Exception:
            continue

        if required_keys and not all(is_num(item.get(k)) for k in required_keys):
            continue

        diff = abs(dt - target_dt)
        if diff <= tolerance and (best_diff is None or diff < best_diff):
            best = item
            best_diff = diff

    return best


def missing_target_labels(history, now_dt):
    labels = []

    checks = [
        ("h6", now_dt - timedelta(hours=6), ["icePressure", "seaPressure", "iceWind", "ventilIndex", "coastGate"]),
        ("h12", now_dt - timedelta(hours=12), ["icePressure", "seaPressure", "ventilIndex", "coastGate"]),
    ]

    for label, target_dt, keys in checks:
        snap = find_history_snapshot(history, target_dt, TREND_TOLERANCES[label], required_keys=keys)
        if snap is None:
            labels.append(label)

    return labels


def backfill_until_targets_found(history, older_instance_ids, missing_labels):
    if not missing_labels or not older_instance_ids:
        return history, {}

    cache = build_empty_cache()
    fetch_errors = {}
    remaining = set(missing_labels)

    target_times = {
        "h6": None,
        "h12": None,
    }

    for iid in reversed(older_instance_ids):
        errs = append_instance_to_cache(cache, iid)
        if errs:
            fetch_errors[iid] = errs

        axis = build_master_time_axis(cache)
        added_this_round = 0

        for label in list(remaining):
            hours_back = 6 if label == "h6" else 12
            # referanse mot nåtid i denne kjøringen finnes i history som siste snapshot
            now_snap = history[-1] if history else None
            if not now_snap:
                continue
            now_dt_hist = parse_iso(now_snap["t"])
            target_dt = now_dt_hist - timedelta(hours=hours_back)

            tolerance_hours = max(3.0, TREND_TOLERANCES[label].total_seconds() / 3600.0)
            valid_time, dt_found, diff_h = find_valid_time(axis, target_dt, tolerance_hours=tolerance_hours)
            if valid_time is None:
                continue

            fields = fields_for_valid_time(cache, valid_time)
            if not fields:
                continue

            snapshot = build_snapshot(dt_found, fields)
            history.append(snapshot)
            history = sort_and_dedup_history(history)
            remaining.discard(label)
            added_this_round += 1

        if added_this_round:
            print(f"Backfill added {added_this_round} snapshots after instance {iid}")

        if not remaining:
            print("Backfill stopping early: all missing targets filled")
            break

    return history, fetch_errors


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
        0.30 * (reservoir / 100.0)
        + 0.26 * (coupling / 100.0)
        + 0.24 * norm(gradient, 15, 40)
        + 0.10 * norm(d6 if is_num(d6) else 0.0, 0, 8)
        + 0.10 * norm(ice_wind_trend_6h if is_num(ice_wind_trend_6h) else 0.0, 0, 6)
    )


def build_payload(now_dt):
    now_str = now_dt.strftime("%Y-%m-%d %H:%M UTC")
    history = sort_and_dedup_history(load_json(HISTORY_FILE, []))

    instance_ids = list_instances()
    latest = instance_ids[-1]
    older_instance_ids = instance_ids[:-1][-5:]

    cache = build_empty_cache()
    fetch_errors_now = append_instance_to_cache(cache, latest)

    axis = build_master_time_axis(cache)
    valid_now, dt_now, diff_now = find_valid_time(axis, now_dt, tolerance_hours=TIME_TOLERANCE_HOURS)
    if valid_now is None:
        raise RuntimeError("Fant ikke gyldig now-tid i nyeste DMI-instance.")

    now_fields = fields_for_valid_time(cache, valid_now)
    if now_fields is None:
        raise RuntimeError("Kunne ikke lese now-felter fra DMI-data.")

    current_snapshot = build_snapshot(dt_now, now_fields)
    history.append(current_snapshot)
    history = sort_and_dedup_history(history)

    missing_labels = missing_target_labels(history, dt_now)
    fetch_errors_backfill = {}
    if missing_labels:
        print("Missing history targets:", missing_labels)
        history, fetch_errors_backfill = backfill_until_targets_found(history, older_instance_ids, missing_labels)

    save_json(HISTORY_FILE, history)

    fetch_errors = {}
    if fetch_errors_now:
        fetch_errors[latest] = fetch_errors_now
    fetch_errors.update(fetch_errors_backfill)

    snap_6 = find_history_snapshot(
        history, dt_now - timedelta(hours=6), TREND_TOLERANCES["h6"],
        required_keys=["icePressure", "seaPressure", "iceWind", "ventilIndex", "coastGate"]
    )
    snap_12 = find_history_snapshot(
        history, dt_now - timedelta(hours=12), TREND_TOLERANCES["h12"],
        required_keys=["icePressure", "seaPressure", "ventilIndex", "coastGate"]
    )
    snap_24 = find_history_snapshot(
        history, dt_now - timedelta(hours=24), TREND_TOLERANCES["h24"],
        required_keys=["iceTempC", "icePressure"]
    )
    snap_72 = find_history_snapshot(
        history, dt_now - timedelta(hours=72), TREND_TOLERANCES["h72"],
        required_keys=["iceTempC", "icePressure"]
    )

    count = sum(1 for x in [current_snapshot, snap_6, snap_12] if x is not None)
    if count == 3:
        trend_status = "ok"
    elif count == 2:
        trend_status = "partial"
    else:
        trend_status = "insufficient_distinct_steps"

    quality_flags = sorted(set(now_fields.get("qualityFlags", [])))
    if fetch_errors:
        quality_flags.append("partial_point_fetch_errors")

    ice_pressure = current_snapshot["icePressure"]
    ice_temp_c = current_snapshot["iceTempC"]
    sea_pressure = current_snapshot["seaPressure"]
    gradient = current_snapshot["gradient"]
    ice_wind = current_snapshot["iceWind"]
    vent_now = current_snapshot["ventilIndex"]
    coast_gate_now = current_snapshot["coastGate"]
    sector = current_snapshot["sector"]

    d6 = (gradient - snap_6["gradient"]) if snap_6 and is_num(snap_6.get("gradient")) and is_num(gradient) else None
    d12 = (gradient - snap_12["gradient"]) if snap_12 and is_num(snap_12.get("gradient")) and is_num(gradient) else None

    sf6 = (snap_6["seaPressure"] - sea_pressure) if snap_6 and is_num(snap_6.get("seaPressure")) and is_num(sea_pressure) else None
    sf12 = (snap_12["seaPressure"] - sea_pressure) if snap_12 and is_num(snap_12.get("seaPressure")) and is_num(sea_pressure) else None

    ice_wind_trend_6h = (ice_wind - snap_6["iceWind"]) if snap_6 and is_num(snap_6.get("iceWind")) and is_num(ice_wind) else None

    vent_d6 = (vent_now - snap_6["ventilIndex"]) if snap_6 and is_num(snap_6.get("ventilIndex")) and is_num(vent_now) else None
    vent_d12 = (vent_now - snap_12["ventilIndex"]) if snap_12 and is_num(snap_12.get("ventilIndex")) and is_num(vent_now) else None

    coast_gate_d6 = (coast_gate_now - snap_6["coastGate"]) if snap_6 and is_num(snap_6.get("coastGate")) and is_num(coast_gate_now) else None
    coast_gate_d12 = (coast_gate_now - snap_12["coastGate"]) if snap_12 and is_num(snap_12.get("coastGate")) and is_num(coast_gate_now) else None

    acc_uncertain = False
    if is_num(d6) and is_num(d12):
        acc_g = d6 - (d12 - d6)
    elif is_num(d6):
        acc_g = d6
        acc_uncertain = True
    else:
        acc_g = None
        acc_uncertain = True

    if is_num(sf6) and is_num(sf12):
        acc_s = sf6 - (sf12 - sf6)
    elif is_num(sf6):
        acc_s = sf6
        acc_uncertain = True
    else:
        acc_s = None
        acc_uncertain = True

    if trend_status == "partial":
        quality_flags.append("partial_trend_window")
    if acc_uncertain:
        quality_flags.append("acceleration_estimated")
    if not is_num(vent_now):
        quality_flags.append("missing_ventil_index")
    if not is_num(coast_gate_now):
        quality_flags.append("missing_coast_gate")

    sea_motion_km6 = None
    if snap_6 and is_num(now_fields["seaMinLon"]) and is_num(now_fields["seaMinLat"]):
        prev_lon = snap_6.get("seaMinLon")
        prev_lat = snap_6.get("seaMinLat")
        if is_num(prev_lon) and is_num(prev_lat):
            dx = (now_fields["seaMinLon"] - prev_lon) * math.cos(math.radians(now_fields["seaMinLat"])) * 111.0
            dy = (now_fields["seaMinLat"] - prev_lat) * 111.0
            sea_motion_km6 = math.sqrt(dx * dx + dy * dy)
            if sea_motion_km6 >= 180:
                quality_flags.append("sea_min_motion_high")

    ice_pressure_anom_now = current_snapshot["icePressureAnomNow"]
    cold_support_now = current_snapshot["coldSupportNow"]
    dT_coast_ice = current_snapshot["dTCoastIceNow"]
    sea_low_depth = max(0.0, 1000.0 - sea_pressure) if is_num(sea_pressure) else None

    ice_temp_trend_24h = None
    ice_temp_trend_72h = None
    if snap_24 and is_num(ice_temp_c) and is_num(snap_24.get("iceTempC")):
        ice_temp_trend_24h = ice_temp_c - snap_24["iceTempC"]
    else:
        quality_flags.append("missing_ice_temp_trend_24h")

    if snap_72 and is_num(ice_temp_c) and is_num(snap_72.get("iceTempC")):
        ice_temp_trend_72h = ice_temp_c - snap_72["iceTempC"]
    else:
        quality_flags.append("missing_ice_temp_trend_72h")

    if is_num(ice_temp_trend_24h) and ice_temp_trend_24h <= -3.0:
        quality_flags.append("cold_reservoir_building_24h")
    if is_num(ice_temp_trend_72h) and ice_temp_trend_72h <= -5.0:
        quality_flags.append("cold_reservoir_building_72h")

    ice_pressure_trend_24h = (
        ice_pressure - snap_24["icePressure"]
        if snap_24 and is_num(snap_24.get("icePressure")) and is_num(ice_pressure)
        else None
    )
    ice_pressure_trend_72h = (
        ice_pressure - snap_72["icePressure"]
        if snap_72 and is_num(snap_72.get("icePressure")) and is_num(ice_pressure)
        else None
    )

    ice_anom_72 = [float(x.get("icePressureAnomNow", 0.0)) for x in history if is_num(x.get("icePressureAnomNow"))]
    cold_72 = [float(x.get("coldSupportNow", 0.0)) for x in history if is_num(x.get("coldSupportNow"))]
    dT_72 = [float(x.get("dTCoastIceNow", 0.0)) for x in history if is_num(x.get("dTCoastIceNow"))]

    ice_anom_72_mean = avg(ice_anom_72) if ice_anom_72 else 0.0
    cold_72_mean = avg(cold_72) if cold_72 else 0.0
    dT_72_mean = avg(dT_72) if dT_72 else 0.0

    reservoir = 100 * (
        0.50 * norm(ice_anom_72_mean, -12, 12)
        + 0.20 * norm(ice_pressure_anom_now, -12, 12)
        + 0.15 * norm(ice_pressure_trend_24h, -3, 8)
        + 0.10 * norm(ice_pressure_trend_72h, -5, 12)
        + 0.05 * (norm(-ice_temp_trend_24h, 0, 8) if is_num(ice_temp_trend_24h) else 0.0)
    )
    reservoir = clamp(reservoir, 0, 100)
    if is_num(ice_anom_72_mean) and ice_anom_72_mean <= -8:
        reservoir = min(reservoir, 39)

    sector_score = {
        "W1": 90, "W2": 92, "W3": 88,
        "C1": 78, "C2": 72,
        "M1": 80, "M2": 84, "M3": 82,
        "E1": 50, "E2": 35,
    }.get(sector, 25)

    coupling = 100 * (
        0.40 * (sector_score / 100.0)
        + 0.16 * norm(sea_low_depth, 5, 30)
        + 0.10 * norm(ice_wind, 4, 20)
        + 0.18 * norm(vent_now, 4, 16)
        + 0.16 * norm(coast_gate_now, 8, 30)
    )
    coupling = clamp(coupling, 0, 100)

    thermal_component = 100 * (
        0.45 * norm(dT_72_mean, 3, 20)
        + 0.25 * norm(cold_72_mean, 5, 25)
        + 0.15 * norm(dT_coast_ice, 10, 35)
        + 0.10 * (norm(-ice_temp_trend_24h, 0, 8) if is_num(ice_temp_trend_24h) else 0.0)
        + 0.05 * (norm(-ice_temp_trend_72h, 0, 12) if is_num(ice_temp_trend_72h) else 0.0)
    )
    thermal_component = clamp(thermal_component, 0, 100)

    reservoir_factor = 0.35 + 0.65 * (reservoir / 100.0)
    katabatic_potential = clamp(thermal_component * reservoir_factor, 0, 100)

    gboost = gradient_boost(gradient)
    trigger = 100 * (
        0.23 * norm(gradient, 0, 65)
        + 0.08 * gboost
        + 0.15 * norm(sf6 if is_num(sf6) else None, 0, 10)
        + 0.09 * norm(sf12 if is_num(sf12) else None, 0, 14)
        + 0.10 * norm(d6 if is_num(d6) else None, 0, 10)
        + 0.06 * norm(d12 if is_num(d12) else None, 0, 14)
        + 0.05 * norm(acc_g if is_num(acc_g) else None, 0, 6)
        + 0.03 * norm(acc_s if is_num(acc_s) else None, 0, 6)
        + 0.05 * norm(ice_wind_trend_6h if is_num(ice_wind_trend_6h) else None, 0, 6)
        + 0.06 * norm(vent_now, 4, 16)
        + 0.03 * norm(vent_d6 if is_num(vent_d6) else None, 0, 6)
        + 0.04 * norm(coast_gate_now, 8, 30)
        + 0.03 * norm(coast_gate_d6 if is_num(coast_gate_d6) else None, 0, 8)
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
            or (is_num(d6) and d6 >= 2)
            or (is_num(sf6) and sf6 >= 2)
            or (is_num(acc_g) and acc_g >= 1)
            or (is_num(ice_wind_trend_6h) and ice_wind_trend_6h >= 2)
            or (is_num(vent_now) and vent_now >= 8)
            or (is_num(coast_gate_now) and coast_gate_now >= 16)
        )
    )

    base = 0.58 * trigger + 0.28 * reservoir + 0.14 * potential
    risk = base * (0.56 + 0.44 * (coupling / 100.0))

    if trigger < 20:
        risk = min(risk, 34)
    if trigger < 35 and reservoir < 25:
        risk = min(risk, 24)

    level, phase, horizon = classify_risk(risk)
    if watch and level == "GRN":
        phase = "WATCH"

    trend_tag = "" if trend_status == "ok" else " T?"
    ag_tag = compact_score_tag("AG", acc_g, uncertain=acc_uncertain)
    as_tag = compact_score_tag("AS", acc_s, uncertain=acc_uncertain)
    ct24_tag = compact_temp_trend_tag("CT24", ice_temp_trend_24h)
    ct72_tag = compact_temp_trend_tag("CT72", ice_temp_trend_72h)
    vg_tag = compact_gate_tag("VG", vent_now)
    cg_tag = compact_gate_tag("CG", coast_gate_now)

    message = (
        f"{LOCATION_NAME} {level} {horizon}{trend_tag} "
        f"RES{int(round(reservoir))} TRG{int(round(trigger))} CPL{int(round(coupling))} "
        f"ICE{ice_pressure:.1f} SEA{sea_pressure:.1f} "
        f"GR{gradient:.1f} "
        f"d6{fmt_msg_num(d6, signed=True)} "
        f"d12{fmt_msg_num(d12, signed=True)} "
        f"SF6{fmt_msg_num(sf6, signed=True)} "
        f"SF12{fmt_msg_num(sf12, signed=True)} "
        f"DT{fmt_msg_num(dT_coast_ice, digits=0)} "
        f"{ct24_tag} {ct72_tag} "
        f"{vg_tag} {cg_tag} "
        f"{ag_tag} {as_tag}"
    )

    payload = {
        "meta": {
            "source": "DMI HARMONIE",
            "updatedAt": now_str,
            "location": LOCATION_NAME,
            "model": COL,
            "forecastInfo": "Latest available forecast step",
            "instanceId": latest,
            "lastSuccessfulUpdate": now_str,
            "lastAttemptFailed": None,
            "stale": False,
        },
        "inputs": {
            "icePressure": round(ice_pressure, 1) if is_num(ice_pressure) else None,
            "seaPressure": round(sea_pressure, 1) if is_num(sea_pressure) else None,
            "gradient": round(gradient, 1) if is_num(gradient) else None,
            "d6": round(d6, 1) if is_num(d6) else None,
            "d12": round(d12, 1) if is_num(d12) else None,
            "sf6": round(sf6, 1) if is_num(sf6) else None,
            "sf12": round(sf12, 1) if is_num(sf12) else None,
            "iceWind": round(ice_wind, 1) if is_num(ice_wind) else None,
            "iceWindTrend6h": round(ice_wind_trend_6h, 1) if is_num(ice_wind_trend_6h) else None,
            "coastIceDeltaT": round(dT_coast_ice, 1) if is_num(dT_coast_ice) else None,
            "iceTempTrend24h": round(ice_temp_trend_24h, 1) if is_num(ice_temp_trend_24h) else None,
            "iceTempTrend72h": round(ice_temp_trend_72h, 1) if is_num(ice_temp_trend_72h) else None,
            "ventilIndex": round(vent_now, 1) if is_num(vent_now) else None,
            "ventilD6": round(vent_d6, 1) if is_num(vent_d6) else None,
            "ventilD12": round(vent_d12, 1) if is_num(vent_d12) else None,
            "coastSeaPressure": round(now_fields["coastSeaPressure"], 1) if is_num(now_fields["coastSeaPressure"]) else None,
            "coastGate": round(coast_gate_now, 1) if is_num(coast_gate_now) else None,
            "coastGateD6": round(coast_gate_d6, 1) if is_num(coast_gate_d6) else None,
            "coastGateD12": round(coast_gate_d12, 1) if is_num(coast_gate_d12) else None,
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
            "usedInstanceIds": [latest],
            "fetchErrors": fetch_errors,
            "seaMinLon": round(now_fields["seaMinLon"], 3) if is_num(now_fields["seaMinLon"]) else None,
            "seaMinLat": round(now_fields["seaMinLat"], 3) if is_num(now_fields["seaMinLat"]) else None,
            "seaCentroidPressure": round(now_fields["seaCentroidPressure"], 1) if is_num(now_fields["seaCentroidPressure"]) else None,
            "seaCentroidLon": round(now_fields["seaCentroidLon"], 3) if is_num(now_fields["seaCentroidLon"]) else None,
            "seaCentroidLat": round(now_fields["seaCentroidLat"], 3) if is_num(now_fields["seaCentroidLat"]) else None,
            "seaCentroidSector": now_fields["seaCentroidSector"],
            "seaMinCentroidSpread": round(now_fields["seaMinCentroidSpread"], 1) if is_num(now_fields["seaMinCentroidSpread"]) else None,
            "seaMinMotionKm6": round(sea_motion_km6, 1) if is_num(sea_motion_km6) else None,
            "westMeanPressure": round(now_fields["westMeanPressure"], 1) if is_num(now_fields["westMeanPressure"]) else None,
            "midMeanPressure": round(now_fields["midMeanPressure"], 1) if is_num(now_fields["midMeanPressure"]) else None,
            "eastMeanPressure": round(now_fields["eastMeanPressure"], 1) if is_num(now_fields["eastMeanPressure"]) else None,
            "coastSeaPressure": round(now_fields["coastSeaPressure"], 1) if is_num(now_fields["coastSeaPressure"]) else None,
            "gateMid": round(now_fields["gateMid"], 1) if is_num(now_fields["gateMid"]) else None,
            "gateEast": round(now_fields["gateEast"], 1) if is_num(now_fields["gateEast"]) else None,
            "ventilIndex": round(vent_now, 1) if is_num(vent_now) else None,
            "ventilD6": round(vent_d6, 1) if is_num(vent_d6) else None,
            "ventilD12": round(vent_d12, 1) if is_num(vent_d12) else None,
            "coastGate": round(coast_gate_now, 1) if is_num(coast_gate_now) else None,
            "coastGateD6": round(coast_gate_d6, 1) if is_num(coast_gate_d6) else None,
            "coastGateD12": round(coast_gate_d12, 1) if is_num(coast_gate_d12) else None,
            "icePressureAnomNow": round(ice_pressure_anom_now, 1) if is_num(ice_pressure_anom_now) else None,
            "icePressureAnom72hMean": round(ice_anom_72_mean, 1) if is_num(ice_anom_72_mean) else None,
            "icePressureTrend24h": round(ice_pressure_trend_24h, 1) if is_num(ice_pressure_trend_24h) else None,
            "icePressureTrend72h": round(ice_pressure_trend_72h, 1) if is_num(ice_pressure_trend_72h) else None,
            "coldSupport72h": round(cold_72_mean, 1) if is_num(cold_72_mean) else None,
            "katabaticPotential": int(round(katabatic_potential)),
            "seaLowDepth": round(sea_low_depth, 1) if is_num(sea_low_depth) else None,
            "gradientBoost": round(gboost, 2),
            "accG": round(acc_g, 1) if is_num(acc_g) else None,
            "accS": round(acc_s, 1) if is_num(acc_s) else None,
            "accUncertain": acc_uncertain,
            "trendDataStatus": trend_status,
            "qualityFlags": sorted(set(quality_flags)),
            "selectedTimes": {
                "now": {"validTime": valid_now, "diffHours": round(diff_now, 2)} if valid_now else None,
                "h6_from_history": snap_6["t"] if snap_6 else None,
                "h12_from_history": snap_12["t"] if snap_12 else None,
                "h24_from_history": snap_24["t"] if snap_24 else None,
                "h72_from_history": snap_72["t"] if snap_72 else None,
            },
            "usedIcePressurePoints": now_fields["usedIcePressurePoints"],
            "usedIceTempPoints": now_fields["usedIceTempPoints"],
            "usedIceWindPoints": now_fields["usedIceWindPoints"],
            "usedSeaPressurePoints": now_fields["usedSeaPressurePoints"],
            "usedSeaTempPoints": now_fields["usedSeaTempPoints"],
        },
        "output": {
            "level": level,
            "phase": phase,
            "message": message,
        },
    }

    return payload


def write_stale_payload(error):
    existing = load_json(DATA_FILE, None)
    err_name = type(error).__name__
    err_text = f"{err_name}"

    if isinstance(existing, dict) and existing.get("inputs") and existing.get("scores") and existing.get("derived"):
        meta = existing.get("meta", {})
        derived = existing.get("derived", {})
        quality_flags = derived.get("qualityFlags", [])
        if not isinstance(quality_flags, list):
            quality_flags = []

        quality_flags = sorted(set(quality_flags + ["stale_due_to_failed_update"]))

        meta["updatedAt"] = now_utc_str()
        meta["source"] = "DMI HARMONIE"
        meta["location"] = LOCATION_NAME
        meta["model"] = COL
        meta["forecastInfo"] = f"Latest update failed: {err_name}"
        meta["lastAttemptFailed"] = err_text
        meta["stale"] = True
        meta.setdefault("lastSuccessfulUpdate", meta.get("updatedAt"))

        derived["trendDataStatus"] = f"stale: {err_name}"
        derived["qualityFlags"] = quality_flags

        existing["meta"] = meta
        existing["derived"] = derived

        existing_output = existing.get("output", {})
        existing_output["phase"] = f"STALE ({existing_output.get('phase', 'UNKNOWN')})"
        existing["output"] = existing_output

        save_json(DATA_FILE, existing)
        print(f"Preserved last good dataset; marked as stale due to {err_name}")
        return

    fallback = {
        "meta": {
            "source": "DMI HARMONIE",
            "updatedAt": now_utc_str(),
            "location": LOCATION_NAME,
            "model": COL,
            "forecastInfo": f"Update failed: {err_name}",
            "instanceId": "-",
            "lastSuccessfulUpdate": None,
            "lastAttemptFailed": err_text,
            "stale": True,
        },
        "inputs": {
            "icePressure": None,
            "seaPressure": None,
            "gradient": None,
            "d6": None,
            "d12": None,
            "sf6": None,
            "sf12": None,
            "iceWind": None,
            "iceWindTrend6h": None,
            "coastIceDeltaT": None,
            "iceTempTrend24h": None,
            "iceTempTrend72h": None,
            "ventilIndex": None,
            "ventilD6": None,
            "ventilD12": None,
            "coastSeaPressure": None,
            "coastGate": None,
            "coastGateD6": None,
            "coastGateD12": None,
        },
        "scores": {
            "reservoir": None,
            "trigger": None,
            "coupling": None,
            "potential": None,
            "risk": None,
        },
        "derived": {
            "watch": False,
            "sector": None,
            "usedInstanceIds": [],
            "fetchErrors": {},
            "seaMinLon": None,
            "seaMinLat": None,
            "seaCentroidPressure": None,
            "seaCentroidLon": None,
            "seaCentroidLat": None,
            "seaCentroidSector": None,
            "seaMinCentroidSpread": None,
            "seaMinMotionKm6": None,
            "westMeanPressure": None,
            "midMeanPressure": None,
            "eastMeanPressure": None,
            "coastSeaPressure": None,
            "gateMid": None,
            "gateEast": None,
            "ventilIndex": None,
            "ventilD6": None,
            "ventilD12": None,
            "coastGate": None,
            "coastGateD6": None,
            "coastGateD12": None,
            "icePressureAnomNow": None,
            "icePressureAnom72hMean": None,
            "icePressureTrend24h": None,
            "icePressureTrend72h": None,
            "coldSupport72h": None,
            "katabaticPotential": None,
            "seaLowDepth": None,
            "gradientBoost": None,
            "accG": None,
            "accS": None,
            "accUncertain": False,
            "trendDataStatus": f"error: {err_name}",
            "qualityFlags": ["hard_failure"],
            "selectedTimes": {
                "now": None,
                "h6_from_history": None,
                "h12_from_history": None,
                "h24_from_history": None,
                "h72_from_history": None,
            },
            "usedIcePressurePoints": 0,
            "usedIceTempPoints": 0,
            "usedIceWindPoints": 0,
            "usedSeaPressurePoints": 0,
            "usedSeaTempPoints": 0,
        },
        "output": {
            "level": "GRN",
            "phase": "ERROR",
            "message": f"{LOCATION_NAME} ERROR DMI {err_name}",
        },
    }
    save_json(DATA_FILE, fallback)
    print(f"No prior good dataset; wrote fallback due to {err_name}")


if __name__ == "__main__":
    try:
        payload = build_payload(now_utc())
        save_json(DATA_FILE, payload)
        print("Updated data.json/history.json successfully")
        print("trendDataStatus:", payload["derived"]["trendDataStatus"])
        print("used instances:", payload["derived"]["usedInstanceIds"])
        if payload["derived"].get("qualityFlags"):
            print("qualityFlags:", payload["derived"]["qualityFlags"])
    except Exception as e:
        print("Update failed:", repr(e))
        write_stale_payload(e)
