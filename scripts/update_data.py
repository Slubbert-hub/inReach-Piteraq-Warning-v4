import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    import requests
except ImportError:
    raise RuntimeError("Python package 'requests' is not installed on runner.")

BASE_URL = "https://opendataapi.dmi.dk/v1/forecastedr"
COLLECTION = "harmonie_ig_sf"

ICE_PRESSURE_NORMAL_HPA = 1013.25

# Redusert antall punkt for fart
RESERVOIR_CORE_POINTS = [
    (-40.2, 68.8),
    (-39.5, 69.3),
]

ICE_CORRIDOR_POINTS = [
    (-41.5, 68.1),
    (-39.0, 69.4),
]

SEA_POINTS = [
    (-29.0, 65.0),  # SC
    (-27.0, 66.0),  # C
    (-24.8, 67.0),  # N
]
SEA_LABELS = ["SC", "C", "N"]

DATA_FILE = Path("data.json")
HISTORY_FILE = Path("history.json")
LOCATION_NAME = "TASIILAQ"

REQUEST_SLEEP = 0.2
REQUEST_TIMEOUT = 15
TIME_TOLERANCE_HOURS = 2.5


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def norm(x, lo, hi):
    if hi == lo:
        return 0.0
    return clamp((x - lo) / (hi - lo), 0.0, 1.0)


def avg(values):
    return sum(values) / len(values) if values else 0.0


def weighted_mean(values, weights):
    if not values:
        return None
    use_w = weights[: len(values)]
    den = sum(use_w)
    if den == 0:
        return None
    return sum(v * w for v, w in zip(values, use_w)) / den


def kelvin_to_celsius(k):
    return k - 273.15


def iso_z(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_json(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def get_json(url, params=None, retries=3):
    last_err = None
    for attempt in range(retries):
        try:
            time.sleep(REQUEST_SLEEP)
            r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(0.8 + attempt)
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


def fetch_position(instance_id, lon, lat, parameter_names, dt_exact=None):
    url = f"{BASE_URL}/collections/{COLLECTION}/instances/{instance_id}/position"
    params = {
        "coords": f"POINT({lon} {lat})",
        "parameter-name": ",".join(parameter_names),
        "crs": "crs84",
        "f": "CoverageJSON",
    }
    if dt_exact is not None:
        params["datetime"] = iso_z(dt_exact)
    return get_json(url, params=params)


def parse_coverage_series(data, parameter_names):
    domain = data.get("domain", {})
    axes = domain.get("axes", {})
    times = axes.get("t", {}).get("values", [])
    ranges = data.get("ranges", {})
    values = {p: ranges.get(p, {}).get("values", []) for p in parameter_names}
    return times, values


def nearest_index(times, target_dt):
    if not times:
        return None
    dts = [parse_iso(t) for t in times]
    return min(range(len(dts)), key=lambda i: abs((dts[i] - target_dt).total_seconds()))


def collect_valid_time_pool(instances, lon, lat):
    pool = []
    for iid in instances:
        try:
            data = fetch_position(iid, lon, lat, ["pressure-sealevel"])
            times, _ = parse_coverage_series(data, ["pressure-sealevel"])
            for t in times:
                pool.append({"instanceId": iid, "validTime": t, "dt": parse_iso(t)})
        except Exception:
            continue
    uniq = {}
    for item in pool:
        uniq[(item["instanceId"], item["validTime"])] = item
    return sorted(uniq.values(), key=lambda x: x["dt"])


def nearest_candidate(candidates, target_dt, used_keys):
    best = None
    best_diff = None
    for c in candidates:
        key = (c["instanceId"], c["validTime"])
        if key in used_keys:
            continue
        diff_h = abs((c["dt"] - target_dt).total_seconds()) / 3600.0
        if diff_h > TIME_TOLERANCE_HOURS:
            continue
        if best is None or diff_h < best_diff:
            best = c
            best_diff = diff_h
    return best


def choose_trend_targets(instances):
    pool = collect_valid_time_pool(instances, RESERVOIR_CORE_POINTS[0][0], RESERVOIR_CORE_POINTS[0][1])

    now_dt = datetime.now(timezone.utc)
    targets = {
        "now": now_dt,
        "m6": now_dt - timedelta(hours=6),
        "m12": now_dt - timedelta(hours=12),
    }

    chosen = {}
    used_keys = set()
    for key in ["now", "m6", "m12"]:
        c = nearest_candidate(pool, targets[key], used_keys)
        chosen[key] = c
        if c:
            used_keys.add((c["instanceId"], c["validTime"]))

    count = sum(1 for v in chosen.values() if v is not None)
    if count == 3:
        status = "ok"
    elif count == 2:
        status = "partial"
    else:
        status = "insufficient_distinct_steps"

    return chosen, status


def fetch_weighted_ice_pressure_at(instance_id, valid_dt):
    vals = []
    for lon, lat in RESERVOIR_CORE_POINTS:
        try:
            data = fetch_position(instance_id, lon, lat, ["pressure-sealevel"], dt_exact=valid_dt)
            _, ranges = parse_coverage_series(data, ["pressure-sealevel"])
            if ranges["pressure-sealevel"]:
                vals.append(ranges["pressure-sealevel"][0] / 100.0)
        except Exception:
            continue
    return weighted_mean(vals, [0.45, 0.55])


def fetch_weighted_ice_wind_at(instance_id, valid_dt):
    vals = []
    for lon, lat in ICE_CORRIDOR_POINTS:
        try:
            data = fetch_position(instance_id, lon, lat, ["wind-speed-100m"], dt_exact=valid_dt)
            _, ranges = parse_coverage_series(data, ["wind-speed-100m"])
            if ranges["wind-speed-100m"]:
                vals.append(ranges["wind-speed-100m"][0])
        except Exception:
            continue
    return weighted_mean(vals, [0.45, 0.55])


def fetch_sea_min_at(instance_id, valid_dt):
    vals = []
    for i, (lon, lat) in enumerate(SEA_POINTS):
        try:
            data = fetch_position(instance_id, lon, lat, ["pressure-sealevel"], dt_exact=valid_dt)
            _, ranges = parse_coverage_series(data, ["pressure-sealevel"])
            if ranges["pressure-sealevel"]:
                vals.append((ranges["pressure-sealevel"][0] / 100.0, SEA_LABELS[i]))
        except Exception:
            continue
    return min(vals, key=lambda x: x[0]) if vals else (None, "?")


def fetch_now_fields(latest_instance_id):
    now_dt = datetime.now(timezone.utc)

    core_pressures = []
    for lon, lat in RESERVOIR_CORE_POINTS:
        try:
            data = fetch_position(latest_instance_id, lon, lat, ["pressure-sealevel"])
            times, ranges = parse_coverage_series(data, ["pressure-sealevel"])
            idx = nearest_index(times, now_dt)
            if idx is not None and ranges["pressure-sealevel"]:
                core_pressures.append(ranges["pressure-sealevel"][idx] / 100.0)
        except Exception:
            continue

    ice_temps = []
    ice_winds = []
    for lon, lat in ICE_CORRIDOR_POINTS:
        try:
            data = fetch_position(latest_instance_id, lon, lat, ["temperature-2m", "wind-speed-100m"])
            times, ranges = parse_coverage_series(data, ["temperature-2m", "wind-speed-100m"])
            idx = nearest_index(times, now_dt)
            if idx is not None:
                if ranges["temperature-2m"]:
                    ice_temps.append(kelvin_to_celsius(ranges["temperature-2m"][idx]))
                if ranges["wind-speed-100m"]:
                    ice_winds.append(ranges["wind-speed-100m"][idx])
        except Exception:
            continue

    sea_pressures = []
    sea_temps = []
    for i, (lon, lat) in enumerate(SEA_POINTS):
        try:
            data = fetch_position(latest_instance_id, lon, lat, ["pressure-sealevel", "temperature-2m"])
            times, ranges = parse_coverage_series(data, ["pressure-sealevel", "temperature-2m"])
            idx = nearest_index(times, now_dt)
            if idx is not None:
                if ranges["pressure-sealevel"]:
                    sea_pressures.append((ranges["pressure-sealevel"][idx] / 100.0, SEA_LABELS[i]))
                if ranges["temperature-2m"]:
                    sea_temps.append(kelvin_to_celsius(ranges["temperature-2m"][idx]))
        except Exception:
            continue

    if not core_pressures or not sea_pressures:
        raise RuntimeError("Manglende nådata fra DMI.")

    return {
        "icePressure": weighted_mean(core_pressures, [0.45, 0.55]),
        "iceTempC": avg(ice_temps),
        "iceWind": weighted_mean(ice_winds, [0.45, 0.55]) if ice_winds else 0.0,
        "seaPressure": min(sea_pressures, key=lambda x: x[0])[0],
        "seaTempC": avg(sea_temps),
        "sector": min(sea_pressures, key=lambda x: x[0])[1],
        "usedReservoirPoints": len(core_pressures),
        "usedIcePoints": len(ice_temps),
        "usedSeaPoints": len(sea_pressures),
    }


def fetch_trigger_multi_instance(instances):
    chosen, status = choose_trend_targets(instances)

    def get_vals(key):
        c = chosen.get(key)
        if not c:
            return None, None, None, "?"
        dt = c["dt"]
        iid = c["instanceId"]
        return (
            fetch_weighted_ice_pressure_at(iid, dt),
            fetch_sea_min_at(iid, dt)[0],
            fetch_weighted_ice_wind_at(iid, dt),
            fetch_sea_min_at(iid, dt)[1],
        )

    ice_now, sea_now, wind_now, sector_now = get_vals("now")
    ice_m6, sea_m6, wind_m6, _ = get_vals("m6")
    ice_m12, sea_m12, wind_m12, _ = get_vals("m12")

    if ice_now is None or sea_now is None:
        return {
            "gradientNow": 0.0,
            "d6": 0.0,
            "d12": 0.0,
            "sf6": 0.0,
            "sf12": 0.0,
            "accG": 0.0,
            "accS": 0.0,
            "iceWindTrend6h": 0.0,
            "sectorNow": sector_now,
            "trendDataStatus": "insufficient_distinct_steps",
            "selectedTimes": {},
        }

    grad_now = ice_now - sea_now
    grad_m6 = (ice_m6 - sea_m6) if (ice_m6 is not None and sea_m6 is not None) else None
    grad_m12 = (ice_m12 - sea_m12) if (ice_m12 is not None and sea_m12 is not None) else None

    d6 = (grad_now - grad_m6) if grad_m6 is not None else 0.0
    d12 = (grad_now - grad_m12) if grad_m12 is not None else 0.0
    sf6 = (sea_m6 - sea_now) if sea_m6 is not None else 0.0
    sf12 = (sea_m12 - sea_now) if sea_m12 is not None else 0.0
    acc_g = d6 - (d12 - d6) if grad_m6 is not None and grad_m12 is not None else 0.0
    acc_s = sf6 - (sf12 - sf6) if sea_m6 is not None and sea_m12 is not None else 0.0
    ice_wind_trend_6h = (wind_now - wind_m6) if (wind_now is not None and wind_m6 is not None) else 0.0

    selected = {}
    for k, c in chosen.items():
        selected[k] = None if c is None else {
            "instanceId": c["instanceId"],
            "validTime": c["validTime"],
        }

    return {
        "gradientNow": grad_now,
        "d6": d6,
        "d12": d12,
        "sf6": sf6,
        "sf12": sf12,
        "accG": acc_g,
        "accS": acc_s,
        "iceWindTrend6h": ice_wind_trend_6h,
        "sectorNow": sector_now,
        "trendDataStatus": status,
        "selectedTimes": selected,
    }


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
        0.35 * (reservoir / 100.0) +
        0.20 * (coupling / 100.0) +
        0.25 * norm(gradient, 15, 40) +
        0.10 * norm(d6, 0, 8) +
        0.10 * norm(ice_wind_trend_6h, 0, 6)
    )


def main():
    now_dt = datetime.now(timezone.utc)
    now_str = now_dt.strftime("%Y-%m-%d %H:%M UTC")

    history = load_json(HISTORY_FILE, [])
    instances = list_instances()
    latest = instances[-1]

    now_fields = fetch_now_fields(latest)
    trig = fetch_trigger_multi_instance(instances)

    ice_pressure = now_fields["icePressure"]
    ice_temp_c = now_fields["iceTempC"]
    ice_wind = now_fields["iceWind"]
    sea_pressure = now_fields["seaPressure"]
    sea_temp_c = now_fields["seaTempC"]
    sector = now_fields["sector"]

    gradient = trig["gradientNow"]
    d6 = trig["d6"]
    d12 = trig["d12"]
    sf6 = trig["sf6"]
    sf12 = trig["sf12"]
    acc_g = trig["accG"]
    acc_s = trig["accS"]
    ice_wind_trend_6h = trig["iceWindTrend6h"]
    trend_status = trig["trendDataStatus"]

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
        0.55 * norm(ice_anom_72_mean, -12, 12) +
        0.20 * norm(ice_pressure_anom_now, -12, 12) +
        0.15 * norm(ice_pressure_trend_24h, -3, 8) +
        0.10 * norm(ice_pressure_trend_72h, -5, 12)
    )
    reservoir = clamp(reservoir, 0, 100)
    if ice_anom_72_mean <= -8:
        reservoir = min(reservoir, 39)

    sector_score = {"SC": 90, "C": 75, "N": 40}.get(sector, 25)
    coupling = 100 * (
        0.60 * (sector_score / 100.0) +
        0.25 * norm(sea_low_depth, 5, 30) +
        0.15 * norm(ice_wind, 4, 20)
    )
    coupling = clamp(coupling, 0, 100)

    thermal_component = 100 * (
        0.55 * norm(dT_72_mean, 3, 20) +
        0.30 * norm(cold_72_mean, 5, 25) +
        0.15 * norm(dT_coast_ice, 10, 35)
    )
    thermal_component = clamp(thermal_component, 0, 100)
    reservoir_factor = 0.35 + 0.65 * (reservoir / 100.0)
    katabatic_potential = clamp(thermal_component * reservoir_factor, 0, 100)

    gboost = gradient_boost(gradient)
    trigger = 100 * (
        0.28 * norm(gradient, 0, 65) +
        0.10 * gboost +
        0.16 * norm(sf6, 0, 10) +
        0.10 * norm(sf12, 0, 14) +
        0.12 * norm(d6, 0, 10) +
        0.08 * norm(d12, 0, 14) +
        0.07 * norm(acc_g, 0, 6) +
        0.04 * norm(acc_s, 0, 6) +
        0.05 * norm(ice_wind_trend_6h, 0, 6)
    )
    trigger = clamp(trigger, 0, 100)

    potential = clamp(potential_index(reservoir, coupling, gradient, d6, ice_wind_trend_6h), 0, 100)

    watch = (
        (reservoir >= 30 or potential >= 45)
        and coupling >= 60
        and (
            gradient >= 20 or d6 >= 2 or sf6 >= 2 or acc_g >= 1 or ice_wind_trend_6h >= 2
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
            "selectedTimes": trig["selectedTimes"],
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
