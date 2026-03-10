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
REQUEST_SLEEP = 1.0
REQUEST_RETRIES = 4
TIME_TOLERANCE_HOURS = 2.5

ICE_PRESSURE_NORMAL_HPA = 1013.25

# 2 ispunkter
ICE_POINTS = [
    {"name": "core", "lon": -39.5, "lat": 69.3},
    {"name": "corridor", "lon": -41.5, "lat": 68.1},
]

# 3 punkter i Danmarkstredet
SEA_POINTS = [
    {"name": "SC", "lon": -29.0, "lat": 65.0},
    {"name": "C", "lon": -27.0, "lat": 66.0},
    {"name": "N", "lon": -24.8, "lat": 67.0},
]

ICE_PARAMS = ["pressure-sealevel", "temperature-2m", "wind-speed-100m"]
SEA_PARAMS = ["pressure-sealevel", "temperature-2m"]


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
                print(f"DMI rate limit (429). Waiting {wait}s...")
                time.sleep(wait)
                continue

            r.raise_for_status()
            return r.json()

        except Exception as e:
            last_err = e
            time.sleep(1.5 + attempt)

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

    # bare siste 3 kjøringer
    return ids[-3:]


def fetch_position(instance_id, lon, lat, parameter_names):
    url = f"{BASE_URL}/collections/{COLLECTION}/instances/{instance_id}/position"
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


def nearest_time_index(times, target_dt, tolerance_hours=TIME_TOLERANCE_HOURS):
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

    if best_diff is None or best_diff > tolerance_hours:
        return None
    return best_i


def collect_all_point_data(instances):
    """
    Ett kall per punkt per instance, flere parametre i samme kall.
    cache[instance]["ice"/"sea"][point_name] = {"times": [...], "values": {...}}
    """
    cache = {}

    for iid in instances:
        cache[iid] = {"ice": {}, "sea": {}}
        print(f"Fetching point data for {iid}")

        for p in ICE_POINTS:
            data = fetch_position(iid, p["lon"], p["lat"], ICE_PARAMS)
            times, values = parse_coverage_series(data, ICE_PARAMS)
            cache[iid]["ice"][p["name"]] = {"times": times, "values": values}

        for p in SEA_POINTS:
            data = fetch_position(iid, p["lon"], p["lat"], SEA_PARAMS)
            times, values = parse_coverage_series(data, SEA_PARAMS)
            cache[iid]["sea"][p["name"]] = {"times": times, "values": values}

    return cache


def build_time_pool(cache):
    pool = []
    for iid, block in cache.items():
        times = block["ice"]["core"]["times"]
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


def extract_point_values(cache, instance_id, point_type, point_name, valid_time):
    block = cache[instance_id][point_type][point_name]
    times = block["times"]
    values = block["values"]

    try:
        idx = times.index(valid_time)
    except ValueError:
        return None

    out = {}
    for k, arr in values.items():
        out[k] = arr[idx] if idx < len(arr) else None
    return out


def fields_for(cache, choice):
    if choice is None:
        return None

    iid = choice["instanceId"]
    vt = choice["validTime"]

    # is
    ice_pressures = []
    ice_temps = []
    ice_winds = []

    for name in ["core", "corridor"]:
        vals = extract_point_values(cache, iid, "ice", name, vt)
        if not vals:
            continue

        p = vals.get("pressure-sealevel")
        t = vals.get("temperature-2m")
        w = vals.get("wind-speed-100m")

        if p is not None:
            ice_pressures.append(p / 100.0)
        if t is not None:
            ice_temps.append(kelvin_to_celsius(t))
        if w is not None:
            ice_winds.append(w)

    # strait
    sea_candidates = []
    sea_temps = []

    for name in ["SC", "C", "N"]:
        vals = extract_point_values(cache, iid, "sea", name, vt)
        if not vals:
            continue

        p = vals.get("pressure-sealevel")
        t = vals.get("temperature-2m")

        if p is not None:
            sea_candidates.append((p / 100.0, name))
        if t is not None:
            sea_temps.append(kelvin_to_celsius(t))

    if not ice_pressures or not sea_candidates:
        return None

    ice_pressure = 0.55 * ice_pressures[0] + 0.45 * ice_pressures[-1] if len(ice_pressures) == 2 else ice_pressures[0]
    ice_wind = 0.55 * ice_winds[0] + 0.45 * ice_winds[-1] if len(ice_winds) == 2 else (ice_winds[0] if ice_winds else 0.0)
    sea_pressure, sector = min(sea_candidates, key=lambda x: x[0])

    return {
        "icePressure": ice_pressure,
        "iceTempC": avg(ice_temps),
        "iceWind": ice_wind,
        "seaPressure": sea_pressure,
        "seaTempC": avg(sea_temps),
        "sector": sector,
        "usedReservoirPoints": len(ice_pressures),
        "usedIcePoints": len(ice_temps),
        "usedSeaPoints": len(sea_candidates),
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

    cache = collect_all_point_data(instances)
    pool = build_time_pool(cache)
    chosen, trend_status = choose_trend_targets(pool)

    if chosen["now"] is None:
        raise RuntimeError("Fant ikke gyldig now-tid i DMI-data.")

    now_fields = fields_for(cache, chosen["now"])
    m6_fields = fields_for(cache, chosen["m6"])
    m12_fields = fields_for(cache, chosen["m12"])

    if now_fields is None:
        raise RuntimeError("Kunne ikke lese now-felter fra point-data.")

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

    # Reservoir
    reservoir = 100 * (
        0.55 * norm(ice_anom_72_mean, -12, 12)
        + 0.20 * norm(ice_pressure_anom_now, -12, 12)
        + 0.15 * norm(ice_pressure_trend_24h, -3, 8)
        + 0.10 * norm(ice_pressure_trend_72h, -5, 12)
    )
    reservoir = clamp(reservoir, 0, 100)
    if ice_anom_72_mean <= -8:
        reservoir = min(reservoir, 39)

    # Coupling
    sector_score = {"SC": 90, "C": 75, "N": 40}.get(sector, 25)
    coupling = 100 * (
        0.60 * (sector_score / 100.0)
        + 0.25 * norm(sea_low_depth, 5, 30)
        + 0.15 * norm(ice_wind, 4, 20)
    )
    coupling = clamp(coupling, 0, 100)

    # Katabatisk potensial
    thermal_component = 100 * (
        0.55 * norm(dT_72_mean, 3, 20)
        + 0.30 * norm(cold_72_mean, 5, 25)
        + 0.15 * norm(dT_coast_ice, 10, 35)
    )
    thermal_component = clamp(thermal_component, 0, 100)
    reservoir_factor = 0.35 + 0.65 * (reservoir / 100.0)
    katabatic_potential = clamp(thermal_component * reservoir_factor, 0, 100)

    # Trigger
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

    # Potential
    potential = clamp(
        potential_index(reservoir, coupling, gradient, d6, ice_wind_trend_6h),
        0,
        100,
    )

    # Watch
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

    # Risk
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
