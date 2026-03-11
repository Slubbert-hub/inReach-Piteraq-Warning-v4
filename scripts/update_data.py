import json
import math
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

BASE = "https://opendataapi.dmi.dk/v1/forecastedr"
COL = "harmonie_ig_sf"

LOCATION_NAME = "TASIILAQ"
DATA_FILE = Path("data.json")
HISTORY_FILE = Path("history.json")

REQUEST_TIMEOUT = 45
REQUEST_SLEEP = 1.2
REQUEST_RETRIES = 5
TIME_TOLERANCE_HOURS = 2.75

ICE_PRESSURE_NORMAL_HPA = 1013.25

# Operativ piteraq-korridor på isen
ICE_POINTS = [
    {"name": "source", "lon": -42.4, "lat": 69.0},
    {"name": "mid",    "lon": -41.3, "lat": 68.6},
    {"name": "mouth",  "lon": -40.3, "lat": 68.2},
]

# 5-punkts havfelt i Danmark-stredet / utenfor Ammassalik
# Primærmodell bruker minimumstrykk. Sentroid brukes kun som kontroll.
SEA_POINTS = [
    {"name": "SW", "lon": -30.2, "lat": 64.9},
    {"name": "S",  "lon": -29.0, "lat": 65.3},
    {"name": "C",  "lon": -28.0, "lat": 65.9},
    {"name": "N",  "lon": -26.9, "lat": 66.5},
    {"name": "NE", "lon": -25.8, "lat": 67.0},
]

ICE_PARAMS = ["pressure-sealevel", "temperature-2m", "wind-speed-100m"]
SEA_PARAMS = ["pressure-sealevel", "temperature-2m"]


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


def get_json(url, params=None, retries=REQUEST_RETRIES):
    last_err = None

    for attempt in range(retries):
        try:
            time.sleep(REQUEST_SLEEP)
            r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)

            if r.status_code == 429:
                wait = 15 + attempt * 10
                print(f"DMI rate limit (429). Waiting {wait}s...")
                time.sleep(wait)
                continue

            r.raise_for_status()
            return r.json()

        except requests.exceptions.ReadTimeout as e:
            last_err = e
            wait = 10 + attempt * 10
            print(f"DMI read timeout. Waiting {wait}s before retry...")
            time.sleep(wait)

        except requests.exceptions.ConnectTimeout as e:
            last_err = e
            wait = 10 + attempt * 10
            print(f"DMI connect timeout. Waiting {wait}s before retry...")
            time.sleep(wait)

        except Exception as e:
            last_err = e
            wait = 3 + attempt * 3
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
    return ids[-3:]


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


def collect_all_point_data(instances):
    cache = {}
    for iid in instances:
        cache[iid] = {"ice": {}, "sea": {}}
        print(f"Fetching point data for {iid}")

        for p in ICE_POINTS:
            data = fetch_position(iid, p["lon"], p["lat"], ICE_PARAMS)
            times, values = parse_coverage_series(data, ICE_PARAMS)
            cache[iid]["ice"][p["name"]] = {
                "times": times, "values": values, "lon": p["lon"], "lat": p["lat"]
            }

        for p in SEA_POINTS:
            data = fetch_position(iid, p["lon"], p["lat"], SEA_PARAMS)
            times, values = parse_coverage_series(data, SEA_PARAMS)
            cache[iid]["sea"][p["name"]] = {
                "times": times, "values": values, "lon": p["lon"], "lat": p["lat"]
            }

    return cache


def build_time_pool(cache):
    pool = []
    for iid, block in cache.items():
        times = block["ice"]["source"]["times"]
        for t in times:
            pool.append({"instanceId": iid, "validTime": t, "dt": parse_iso(t)})

    uniq = {}
    for item in pool:
        uniq[(item["instanceId"], item["validTime"])] = item
    return sorted(uniq.values(), key=lambda x: x["dt"])


def choose_nearest_distinct(pool):
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
        out[k] = safe_get(arr, idx)
    return out


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


def fields_for(cache, choice):
    if choice is None:
        return None

    iid = choice["instanceId"]
    vt = choice["validTime"]

    pressure_vals = []
    temp_vals = []
    wind_vals = []
    quality_flags = []

    for name in ["source", "mid", "mouth"]:
        vals = extract_point_values(cache, iid, "ice", name, vt)
        if not vals:
            pressure_vals.append(None)
            temp_vals.append(None)
            wind_vals.append(None)
            continue

        p = vals.get("pressure-sealevel")
        t = vals.get("temperature-2m")
        w = vals.get("wind-speed-100m")

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
        ice_temp_c = -20.0
    elif sum(1 for v in temp_vals if is_num(v)) < 3:
        quality_flags.append("missing_ice_temperature_partial")

    if not is_num(ice_wind):
        quality_flags.append("missing_ice_wind_all")
        ice_wind = 0.0
    elif sum(1 for v in wind_vals if is_num(v)) < 3:
        quality_flags.append("missing_ice_wind_partial")

    sea_candidates = []
    sea_temps = []

    for name in ["SW", "S", "C", "N", "NE"]:
        vals = extract_point_values(cache, iid, "sea", name, vt)
        if not vals:
            continue

        p = vals.get("pressure-sealevel")
        t = vals.get("temperature-2m")
        meta = cache[iid]["sea"][name]

        if is_num(p):
            sea_candidates.append({
                "name": name,
                "pressure": p / 100.0,
                "lon": meta["lon"],
                "lat": meta["lat"],
            })

        if is_num(t):
            sea_temps.append(kelvin_to_celsius(t))

    if not sea_candidates:
        return None

    # Primær: minimumstrykk
    sea_min = min(sea_candidates, key=lambda x: x["pressure"])

    # Sekundær: sentroid kun som kontrollmål
    centroid_pressure, centroid_lon, centroid_lat, centroid_sector = centroid_low(sea_candidates)

    spread = None
    if is_num(centroid_pressure):
        spread = centroid_pressure - sea_min["pressure"]
        if spread >= 3.5:
            quality_flags.append("sea_field_spread_high")
        elif spread >= 2.0:
            quality_flags.append("sea_field_spread_moderate")

    if sea_temps:
        sea_temp_c = avg(sea_temps)
        if len(sea_temps) < 5:
            quality_flags.append("missing_sea_temperature_partial")
    else:
        quality_flags.append("missing_sea_temperature_all")
        sea_temp_c = 0.0

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
        "qualityFlags": sorted(set(quality_flags)),
        "usedIcePressurePoints": sum(1 for v in pressure_vals if is_num(v)),
        "usedIceTempPoints": sum(1 for v in temp_vals if is_num(v)),
        "usedIceWindPoints": sum(1 for v in wind_vals if is_num(v)),
        "usedSeaPressurePoints": len(sea_candidates),
        "usedSeaTempPoints": len(sea_temps),
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
        + 0.10 * norm(d6 if is_num(d6) else 0.0, 0, 8)
        + 0.10 * norm(ice_wind_trend_6h if is_num(ice_wind_trend_6h) else 0.0, 0, 6)
    )


def main():
    now_dt = datetime.now(timezone.utc)
    now_str = now_dt.strftime("%Y-%m-%d %H:%M UTC")
    history = load_json(HISTORY_FILE, [])

    instances = list_instances()
    latest = instances[-1]

    cache = collect_all_point_data(instances)
    pool = build_time_pool(cache)
    chosen, trend_status = choose_nearest_distinct(pool)

    if chosen["now"] is None:
        raise RuntimeError("Fant ikke gyldig now-tid i DMI-data.")

    now_fields = fields_for(cache, chosen["now"])
    m6_fields = fields_for(cache, chosen["m6"])
    m12_fields = fields_for(cache, chosen["m12"])

    if now_fields is None:
        raise RuntimeError("Kunne ikke lese now-felter fra DMI-data.")

    quality_flags = sorted(set(
        now_fields.get("qualityFlags", [])
        + (m6_fields.get("qualityFlags", []) if m6_fields else [])
        + (m12_fields.get("qualityFlags", []) if m12_fields else [])
    ))

    ice_pressure = now_fields["icePressure"]
    ice_temp_c = now_fields["iceTempC"]
    ice_wind = now_fields["iceWind"]

    # Primær sjøvariabel er minimumstrykk
    sea_pressure = now_fields["seaPressureMin"]
    sector = now_fields["seaMinSector"]

    gradient = ice_pressure - sea_pressure

    gradient_m6 = (m6_fields["icePressure"] - m6_fields["seaPressureMin"]) if m6_fields else None
    gradient_m12 = (m12_fields["icePressure"] - m12_fields["seaPressureMin"]) if m12_fields else None

    d6 = (gradient - gradient_m6) if is_num(gradient_m6) else None
    d12 = (gradient - gradient_m12) if is_num(gradient_m12) else None

    sf6 = (m6_fields["seaPressureMin"] - sea_pressure) if m6_fields else None
    sf12 = (m12_fields["seaPressureMin"] - sea_pressure) if m12_fields else None

    ice_wind_trend_6h = (ice_wind - m6_fields["iceWind"]) if m6_fields and is_num(m6_fields["iceWind"]) else None

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

    if trend_status == "partial" and "partial_trend_window" not in quality_flags:
        quality_flags.append("partial_trend_window")

    if acc_uncertain and "acceleration_estimated" not in quality_flags:
        quality_flags.append("acceleration_estimated")

    # Kontroll: bevegelse i minimumspunktet
    sea_motion_km6 = None
    if m6_fields and is_num(m6_fields["seaMinLon"]) and is_num(m6_fields["seaMinLat"]):
        dx = (now_fields["seaMinLon"] - m6_fields["seaMinLon"]) * math.cos(math.radians(now_fields["seaMinLat"])) * 111.0
        dy = (now_fields["seaMinLat"] - m6_fields["seaMinLat"]) * 111.0
        sea_motion_km6 = math.sqrt(dx * dx + dy * dy)
        if sea_motion_km6 >= 180:
            quality_flags.append("sea_min_motion_high")

    ice_pressure_anom_now = ice_pressure - ICE_PRESSURE_NORMAL_HPA
    dT_coast_ice = max(0.0, now_fields["seaTempC"] - ice_temp_c)
    cold_support_now = max(0.0, -ice_temp_c)
    sea_low_depth = max(0.0, 1000.0 - sea_pressure)

    snapshot = {
        "t": now_dt.isoformat(),
        "icePressure": round(ice_pressure, 1),
        "seaPressure": round(sea_pressure, 1),
        "gradient": round(gradient, 1),
        "iceWind": round(ice_wind, 1) if is_num(ice_wind) else None,
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

    ice_24h_ago = float(prev_24h["icePressure"]) if prev_24h and is_num(prev_24h.get("icePressure")) else ice_pressure
    ice_72h_ago = float(prev_72h["icePressure"]) if prev_72h and is_num(prev_72h.get("icePressure")) else ice_pressure

    ice_pressure_trend_24h = ice_pressure - ice_24h_ago
    ice_pressure_trend_72h = ice_pressure - ice_72h_ago

    ice_anom_72 = [float(x.get("icePressureAnomNow", 0.0)) for x in history if is_num(x.get("icePressureAnomNow"))]
    cold_72 = [float(x.get("coldSupportNow", 0.0)) for x in history if is_num(x.get("coldSupportNow"))]
    dT_72 = [float(x.get("dTCoastIceNow", 0.0)) for x in history if is_num(x.get("dTCoastIceNow"))]

    ice_anom_72_mean = avg(ice_anom_72) if ice_anom_72 else 0.0
    cold_72_mean = avg(cold_72) if cold_72 else 0.0
    dT_72_mean = avg(dT_72) if dT_72 else 0.0

    reservoir = 100 * (
        0.55 * norm(ice_anom_72_mean, -12, 12)
        + 0.20 * norm(ice_pressure_anom_now, -12, 12)
        + 0.15 * norm(ice_pressure_trend_24h, -3, 8)
        + 0.10 * norm(ice_pressure_trend_72h, -5, 12)
    )
    reservoir = clamp(reservoir, 0, 100)
    if ice_anom_72_mean <= -8:
        reservoir = min(reservoir, 39)

    sector_score = {"SW": 85, "S": 90, "C": 70, "N": 45, "NE": 35}.get(sector, 25)
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
        0.30 * norm(gradient, 0, 65)
        + 0.10 * gboost
        + 0.18 * norm(sf6 if is_num(sf6) else 0.0, 0, 10)
        + 0.10 * norm(sf12 if is_num(sf12) else 0.0, 0, 14)
        + 0.12 * norm(d6 if is_num(d6) else 0.0, 0, 10)
        + 0.06 * norm(d12 if is_num(d12) else 0.0, 0, 14)
        + 0.06 * norm(acc_g if is_num(acc_g) else 0.0, 0, 6)
        + 0.03 * norm(acc_s if is_num(acc_s) else 0.0, 0, 6)
        + 0.05 * norm(ice_wind_trend_6h if is_num(ice_wind_trend_6h) else 0.0, 0, 6)
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
    ag_tag = compact_score_tag("AG", acc_g, uncertain=acc_uncertain)
    as_tag = compact_score_tag("AS", acc_s, uncertain=acc_uncertain)

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
        },
        "inputs": {
            "icePressure": round(ice_pressure, 1),
            "seaPressure": round(sea_pressure, 1),
            "gradient": round(gradient, 1),
            "d6": round(d6, 1) if is_num(d6) else None,
            "d12": round(d12, 1) if is_num(d12) else None,
            "sf6": round(sf6, 1) if is_num(sf6) else None,
            "sf12": round(sf12, 1) if is_num(sf12) else None,
            "iceWind": round(ice_wind, 1) if is_num(ice_wind) else None,
            "iceWindTrend6h": round(ice_wind_trend_6h, 1) if is_num(ice_wind_trend_6h) else None,
            "coastIceDeltaT": round(dT_coast_ice, 1) if is_num(dT_coast_ice) else None,
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
            "seaMinLon": round(now_fields["seaMinLon"], 3) if is_num(now_fields["seaMinLon"]) else None,
            "seaMinLat": round(now_fields["seaMinLat"], 3) if is_num(now_fields["seaMinLat"]) else None,
            "seaCentroidPressure": round(now_fields["seaCentroidPressure"], 1) if is_num(now_fields["seaCentroidPressure"]) else None,
            "seaCentroidLon": round(now_fields["seaCentroidLon"], 3) if is_num(now_fields["seaCentroidLon"]) else None,
            "seaCentroidLat": round(now_fields["seaCentroidLat"], 3) if is_num(now_fields["seaCentroidLat"]) else None,
            "seaCentroidSector": now_fields["seaCentroidSector"],
            "seaMinCentroidSpread": round(now_fields["seaMinCentroidSpread"], 1) if is_num(now_fields["seaMinCentroidSpread"]) else None,
            "seaMinMotionKm6": round(sea_motion_km6, 1) if is_num(sea_motion_km6) else None,
            "icePressureAnomNow": round(ice_pressure_anom_now, 1),
            "icePressureAnom72hMean": round(ice_anom_72_mean, 1),
            "icePressureTrend24h": round(ice_pressure_trend_24h, 1),
            "icePressureTrend72h": round(ice_pressure_trend_72h, 1),
            "coldSupport72h": round(cold_72_mean, 1),
            "katabaticPotential": int(round(katabatic_potential)),
            "seaLowDepth": round(sea_low_depth, 1),
            "gradientBoost": round(gboost, 2),
            "accG": round(acc_g, 1) if is_num(acc_g) else None,
            "accS": round(acc_s, 1) if is_num(acc_s) else None,
            "accUncertain": acc_uncertain,
            "trendDataStatus": trend_status,
            "qualityFlags": quality_flags,
            "selectedTimes": {
                k: None if v is None else {
                    "instanceId": v["instanceId"],
                    "validTime": v["validTime"],
                }
                for k, v in chosen.items()
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

    save_json(DATA_FILE, payload)
    print("Updated data.json/history.json")
    print("trendDataStatus:", trend_status)
    if quality_flags:
        print("qualityFlags:", quality_flags)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        fallback = load_json(
            DATA_FILE,
            {
                "meta": {},
                "inputs": {},
                "scores": {},
                "derived": {},
                "output": {},
            },
        )
        fallback["meta"] = {
            "source": "DMI HARMONIE",
            "updatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "location": LOCATION_NAME,
            "model": COL,
            "forecastInfo": f"Update failed: {type(e).__name__}",
            "instanceId": "-",
        }
        fallback["derived"] = fallback.get("derived", {})
        fallback["derived"]["trendDataStatus"] = f"error: {type(e).__name__}"
        fallback["derived"]["qualityFlags"] = ["hard_failure"]
        fallback["output"] = fallback.get("output", {})
        fallback["output"]["phase"] = "ERROR"
        fallback["output"]["message"] = f"{LOCATION_NAME} ERROR DMI {type(e).__name__}"
        save_json(DATA_FILE, fallback)
        print("Script failed:", repr(e))
