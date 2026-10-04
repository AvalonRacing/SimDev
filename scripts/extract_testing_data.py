"""Extract speed, cornering radius, driver inputs and body roll from a test run.

Inputs (one run, two loggers):
  log_imu*.csv         25 Hz GPS + IMU logger (RaceBox-style export)
  log_trx_*.csv        100 Hz transmitter log (steering / throttle in %)

Outputs (written to --out):
  run_merged.csv       25 Hz time series on the IMU clock, inputs aligned
  corners.csv          one row per corner pass (apex values)
  corners_by_turn.csv  corner passes grouped by track position (median values)
  *.png                diagnostic plots

Findings about the sensors (see the printed report):
  - Speed is in m/s. GForce/Gyro axes: X points rearward, Y right, Z up.
  - The transmitter clock starts ~15.9 s after the IMU clock; the offset is
    found by correlating steering with yaw curvature r/v.
  - The accelerometer cannot give roll/pitch: while driving it reads a
    +0.14 g forward offset and a 0.96 g vertical (vibration rectification),
    and its lateral scale disagrees with v*r by ~5 %. That is several degrees
    of error, larger than the body roll itself. Roll therefore comes from the
    roll gyro; pitch is below the noise floor and is not output.

Usage:
  python scripts/extract_testing_data.py [--raw TestingData_Raw] [--out TestingData_Processed]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

G = 9.80665
EARTH_R = 6371000.0
FS_IMU = 25.0
DT = 1.0 / FS_IMU

DRIVE_SPEED = 2.0      # m/s, samples used for fits
MAX_RADIUS = 50.0      # m, larger radii are reported as straight (NaN)
CORNER_AY = 0.4        # g, |lateral accel| that marks a corner
CORNER_MIN_S = 0.3     # s, shortest corner kept
TURN_HALFWIDTH = 0.03  # lap fraction either side of a turn's apex peak
MIN_PASSES = 10        # passes needed to call a cluster a turn
ROLL_TAU = 1.0         # s, gyro/model crossover of the roll estimate


def lowpass(x: np.ndarray, fc: float) -> np.ndarray:
    b, a = butter(2, fc / (FS_IMU / 2))
    return filtfilt(b, a, x)


def first_order(x: np.ndarray, tau: float) -> np.ndarray:
    """Causal first-order low-pass with time constant tau."""
    k = np.exp(-DT / tau)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = k * out[i - 1] + (1 - k) * x[i]
    return out


def leaky_integral(w: np.ndarray, tau: float) -> np.ndarray:
    """Integral of w, high-passed with time constant tau (removes gyro drift)."""
    k = np.exp(-DT / tau)
    out = np.zeros_like(w)
    for i in range(1, len(w)):
        out[i] = k * out[i - 1] + w[i] * DT
    return out


def load_imu(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    stamp = pd.to_datetime(raw["Time"])
    df = pd.DataFrame({"time": stamp, "t": (stamp - stamp.iloc[0]).dt.total_seconds()})
    lat0 = np.radians(raw["Latitude"].iloc[0])
    df["x_m"] = np.radians(raw["Longitude"] - raw["Longitude"].iloc[0]) * EARTH_R * np.cos(lat0)
    df["y_m"] = np.radians(raw["Latitude"] - raw["Latitude"].iloc[0]) * EARTH_R
    df["lat"], df["lon"], df["lap"] = raw["Latitude"], raw["Longitude"], raw["Lap"]
    df["speed_ms"] = raw["Speed"]
    # Logger axes: X rearward, Y right, Z up. Flip to forward / left.
    df["f_fwd"] = -raw["GForceX"] * G
    df["f_left"] = -raw["GForceY"] * G
    df["f_up"] = raw["GForceZ"] * G
    for axis in "XYZ":
        df[f"gyro{axis}"] = np.radians(raw[f"Gyro{axis}"])
    return df


def load_trx(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, skiprows=3)
    rec = raw["REC TIME"].str.strip("'").str.split(":", expand=True).astype(float)
    t = rec[0] * 3600 + rec[1] * 60 + rec[2]
    return pd.DataFrame({
        "t": t,
        "steer_pct": pd.to_numeric(raw["ST(%)"]),
        "throttle_pct": pd.to_numeric(raw["TH(%)"]),
    })


def driving_window(imu: pd.DataFrame) -> tuple[float, float]:
    """From the first move to the final stop (the car is carried after that)."""
    moving = imu["speed_ms"].to_numpy() > 0.5
    t = imu["t"].to_numpy()
    start = t[np.argmax(moving)]
    stopped = imu["speed_ms"].rolling(25, center=True).max().to_numpy() < 0.4
    late_stops = np.where(stopped & (t > start))[0]
    end = t[late_stops[0]] if len(late_stops) else t[-1]
    return start, end


def gyro_bias(imu: pd.DataFrame, start: float) -> dict[str, float]:
    """Median gyro reading while parked: before the start if the log has a
    parked lead-in, otherwise any parked spell (a log can start mid-run)."""
    parked = (imu["speed_ms"].rolling(25, center=True).max() < 0.3).to_numpy()
    still = parked & (imu["t"] < start - 1.0).to_numpy()
    if still.sum() < 25:
        still = parked
    if still.sum() < 25:
        print("warning: no parked spell in the log, gyro bias assumed zero")
        return {a: 0.0 for a in "XYZ"}
    return {a: float(imu.loc[still, f"gyro{a}"].median()) for a in "XYZ"}


def find_offset(imu: pd.DataFrame, trx: pd.DataFrame, mask: np.ndarray) -> tuple[float, float]:
    """IMU time = transmitter time + offset; maximises corr(steer, r/v).

    The two logs need not cover the same span, so every offset that leaves at
    least half of the shorter log overlapping the other is tried: a coarse
    0.1 s sweep, then 0.01 s around the best match.
    """
    ti = imu["t"].to_numpy()
    curv = (imu["yaw_rate"] / imu["speed_ms"].clip(lower=1.0)).to_numpy()
    trx_len = trx["t"].iloc[-1]
    need = 0.5 * min(mask.sum(), trx_len * FS_IMU)

    def score(off: float) -> float:
        st = np.interp(ti - off, trx["t"], trx["steer_pct"], left=np.nan, right=np.nan)
        m = mask & ~np.isnan(st)
        if m.sum() < need:
            return -1.0
        return float(np.corrcoef(st[m], curv[m])[0, 1])

    coarse = np.arange(-trx_len, ti[-1], 0.1)
    best = float(coarse[np.argmax([score(o) for o in coarse])])
    fine = np.arange(best - 0.2, best + 0.2, 0.01)
    scores = [score(o) for o in fine]
    return float(fine[np.argmax(scores)]), max(scores)


def segments(flag: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[0, flag.astype(int), 0])
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("TestingData_Raw"))
    ap.add_argument("--out", type=Path, default=Path("TestingData_Processed"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    imu = load_imu(next(args.raw.glob("log_imu*.csv")))
    trx = load_trx(next(args.raw.glob("log_trx_*.csv")))
    t = imu["t"].to_numpy()
    v = imu["speed_ms"].to_numpy()

    start, end = driving_window(imu)
    imu["driving"] = (t >= start) & (t < end)
    fit = imu["driving"].to_numpy() & (v > DRIVE_SPEED)

    bias = gyro_bias(imu, start)
    r = lowpass(imu["gyroZ"].to_numpy() - bias["Z"], 2.0)
    imu["yaw_rate"] = r
    gx = imu["gyroX"].to_numpy() - bias["X"]

    # Yaw leaks into the roll gyro (sensor tilted on its mount); remove it.
    tilt_x = np.polyfit(r[fit], gx[fit], 1)[0]
    roll_rate = -(gx - tilt_x * r)          # positive = rolling right (right side down)

    # --- radius ---------------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        radius = v / r                      # positive = left-hand corner
    radius[(np.abs(radius) > MAX_RADIUS) | (v < 1.0)] = np.nan
    imu["radius_m"] = radius

    xs, ys = lowpass(imu["x_m"].to_numpy(), 2.0), lowpass(imu["y_m"].to_numpy(), 2.0)
    course = np.unwrap(np.arctan2(np.gradient(ys, t), np.gradient(xs, t)))
    course_rate = lowpass(np.gradient(course, t), 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        radius_gps = v / course_rate
    radius_gps[(np.abs(radius_gps) > MAX_RADIUS) | (v < 1.0)] = np.nan
    imu["radius_gps_m"] = radius_gps

    ay = lowpass(v * r, 1.5) / G            # path lateral accel, + = left
    ax = np.gradient(lowpass(v, 1.5), t) / G
    imu["ay_g"], imu["ax_g"] = ay, ax

    # --- roll -----------------------------------------------------------
    # Gyro gives the dynamic part; its high-passed integral is regressed on the
    # identically high-passed ay to get the roll gradient K. The low-frequency
    # part (long corners) is then taken from K*ay.
    roll_hp = leaky_integral(roll_rate, ROLL_TAU)
    ay_hp = ay - first_order(ay, ROLL_TAU)
    k_roll, _ = np.polyfit(ay_hp[fit], roll_hp[fit], 1)
    roll_corr = np.corrcoef(ay_hp[fit], roll_hp[fit])[0, 1]
    roll = roll_hp + first_order(k_roll * ay, ROLL_TAU)
    imu["roll_deg"] = np.degrees(roll)

    # Pitch check only (not output): same method on the pitch gyro.
    gy = imu["gyroY"].to_numpy() - bias["Y"]
    gy = gy - np.polyfit(r[fit], gy[fit], 1)[0] * r
    pitch_hp = leaky_integral(gy - np.median(gy[fit]), ROLL_TAU)
    ax_hp = ax - first_order(ax, ROLL_TAU)
    k_pitch = np.polyfit(ax_hp[fit], pitch_hp[fit], 1)[0]
    pitch_corr = np.corrcoef(ax_hp[fit], pitch_hp[fit])[0, 1]

    # --- inputs ---------------------------------------------------------
    offset, steer_corr = find_offset(imu, trx, fit)
    for col in ("steer_pct", "throttle_pct"):
        imu[col] = np.interp(t - offset, trx["t"], trx[col], left=np.nan, right=np.nan)

    curvature = r / np.maximum(v, 1.0)
    has_inputs = imu["steer_pct"].notna().to_numpy()
    linear = fit & has_inputs & (np.abs(imu["steer_pct"]) < 50)
    steer_gain = np.polyfit(imu.loc[linear, "steer_pct"], curvature[linear], 1)
    lock = fit & (np.abs(imu["steer_pct"]) > 95)
    lock_radius = 1 / np.median(np.abs(curvature[lock]))

    # --- corners --------------------------------------------------------
    dist = np.cumsum(v) * DT
    lap_start = imu.groupby("lap")["t"].transform("idxmin")
    lap_dist = dist - dist[lap_start]
    lap_len = pd.Series(lap_dist).groupby(imu["lap"]).transform("max")
    imu["lap_fraction"] = lap_dist / lap_len
    rows = []
    in_corner = imu["driving"].to_numpy() & (np.abs(ay) > CORNER_AY) & (v > 1.0)
    for i0, i1 in segments(in_corner):
        if (i1 - i0) * DT < CORNER_MIN_S:
            continue
        seg = imu.iloc[i0:i1]
        apex = seg.index[np.nanargmin(np.abs(seg["radius_m"]).fillna(np.inf).to_numpy())]
        a = imu.loc[apex]
        rows.append({
            "apex_idx": apex, "lap": int(a["lap"]), "t_entry_s": seg["t"].iloc[0], "t_exit_s": seg["t"].iloc[-1],
            "duration_s": (i1 - i0) * DT, "direction": "left" if a["ay_g"] > 0 else "right",
            "apex_x_m": a["x_m"], "apex_y_m": a["y_m"],
            "apex_radius_m": abs(a["radius_m"]), "apex_radius_gps_m": abs(a["radius_gps_m"]),
            "apex_speed_ms": a["speed_ms"], "min_speed_ms": seg["speed_ms"].min(),
            "entry_speed_ms": seg["speed_ms"].iloc[0], "exit_speed_ms": seg["speed_ms"].iloc[-1],
            "apex_ay_g": abs(a["ay_g"]), "max_ay_g": seg["ay_g"].abs().max(),
            "apex_steer_pct": a["steer_pct"], "mean_steer_pct": seg["steer_pct"].mean(),
            "apex_throttle_pct": a["throttle_pct"], "mean_throttle_pct": seg["throttle_pct"].mean(),
            "apex_roll_deg": a["roll_deg"],
        })
    corners = pd.DataFrame(rows)

    # Group passes into physical turns by where in the lap the apex is. GPS
    # scatters the line by a few metres from lap to lap, which is the size of
    # a turn here, so lap fraction clusters far more cleanly than x/y.
    corners["lap_fraction"] = imu.loc[corners.pop("apex_idx"), "lap_fraction"].to_numpy()
    corners["turn"] = -1
    full = corners["lap"].between(1, imu["lap"].max() - 1)
    groups = []
    for direction, sub in corners[full].groupby("direction"):
        # Apex density along the lap; each peak is one turn, each pass goes to
        # the nearest peak within TURN_HALFWIDTH.
        frac = sub["lap_fraction"].to_numpy()
        hist, _ = np.histogram(frac, bins=200, range=(0, 1))
        dens = np.convolve(np.r_[hist[-5:], hist, hist[:5]], np.hanning(7), "same")[5:-5]
        peaks, _ = find_peaks(dens, height=MIN_PASSES / 3, distance=4)
        centres = (peaks + 0.5) / 200
        gap = np.abs(frac[:, None] - centres[None, :])
        gap = np.minimum(gap, 1 - gap)
        nearest = gap.argmin(axis=1)
        for k in range(len(centres)):
            idx = sub.index[(nearest == k) & (gap.min(axis=1) < TURN_HALFWIDTH)]
            if len(idx) >= MIN_PASSES:
                groups.append(idx)
    groups.sort(key=lambda idx: corners.loc[idx, "lap_fraction"].median())
    for n, idx in enumerate(groups, start=1):
        corners.loc[idx, "turn"] = n
    turns = corners[corners["turn"] > 0]
    num = turns.select_dtypes("number").drop(columns=["lap", "turn", "t_entry_s", "t_exit_s"])
    by_turn = num.groupby(turns["turn"]).median()
    by_turn.insert(0, "direction", turns.groupby("turn")["direction"].first())
    by_turn.insert(0, "passes", turns.groupby("turn").size())

    # --- write ----------------------------------------------------------
    cols = ["time", "t", "lap", "lap_fraction", "driving", "lat", "lon", "x_m", "y_m", "speed_ms",
            "ax_g", "ay_g", "yaw_rate", "radius_m", "radius_gps_m",
            "steer_pct", "throttle_pct", "roll_deg"]
    out = imu[cols].rename(columns={"t": "t_s"})
    out["yaw_rate"] = np.degrees(out["yaw_rate"])
    out = out.rename(columns={"yaw_rate": "yaw_rate_degs"})
    out.round({c: 7 if c in ("lat", "lon") else 4 for c in out.columns if c != "time"}).to_csv(args.out / "run_merged.csv", index=False)
    corners.round(3).to_csv(args.out / "corners.csv", index=False)
    by_turn.round(3).to_csv(args.out / "corners_by_turn.csv")

    agree = fit & np.isfinite(radius) & np.isfinite(radius_gps) & (np.abs(ay) > CORNER_AY)
    print(f"driving window      : {start:.2f} s .. {end:.2f} s (IMU clock)")
    print(f"gyro bias (deg/s)   : " + ", ".join(f"{k}={np.degrees(b):.2f}" for k, b in bias.items()))
    print(f"transmitter offset  : IMU t = TRX t + {offset:.2f} s  (corr steer~r/v = {steer_corr:.3f})")
    covered = imu.loc[imu["driving"], "steer_pct"].notna()
    print(f"inputs cover        : {covered.mean()*100:.0f} % of the driving window "
          f"(transmitter log {trx['t'].iloc[-1]:.1f} s long)")
    print(f"steering gain       : curvature = {steer_gain[0]*100:.3f} 1/m per 100 % below 50 % "
          f"(+ = left); median radius at full lock {lock_radius:.2f} m")
    print(f"radius yaw vs GPS   : median ratio {np.median(np.abs(radius[agree] / radius_gps[agree])):.3f} in corners")
    print(f"sensor tilt (roll)  : {np.degrees(np.arcsin(tilt_x)):.2f} deg yaw->roll gyro leak removed")
    print(f"roll gradient       : {np.degrees(k_roll):+.2f} deg/g (+ = leans outward), corr {roll_corr:.2f}")
    print(f"pitch gradient      : {np.degrees(k_pitch):+.2f} deg/g, corr {pitch_corr:.2f}  -> not resolvable, not output")
    accel_up = imu.loc[fit, "f_up"].mean() / G
    accel_fwd = (imu.loc[fit, "f_fwd"].mean() - 0) / G
    print(f"accel while driving : mean fwd {accel_fwd:+.3f} g, mean up {accel_up:.3f} g (expected 0 and ~1.0)")
    print(f"corners             : {len(corners)} passes, {len(by_turn)} turns with >={MIN_PASSES} passes")
    print(by_turn[["passes", "direction", "apex_radius_m", "apex_speed_ms", "apex_ay_g",
                   "apex_steer_pct", "apex_throttle_pct", "apex_roll_deg", "lap_fraction"]].round(2).to_string())

    plot(imu, corners, by_turn, args.out)


def plot(imu: pd.DataFrame, corners: pd.DataFrame, by_turn: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
    plt.rcParams.update({"axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted, "axes.grid": True, "grid.color": grid,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    drv = imu[imu["driving"]]

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(drv["x_m"], drv["y_m"], c=drv["speed_ms"], s=3, cmap="Blues", vmin=0)
    fig.colorbar(sc, label="speed (m/s)")
    for turn, row in by_turn.iterrows():
        ax.annotate(f"T{turn}\nR {row.apex_radius_m:.1f} m\n{row.apex_speed_ms:.1f} m/s",
                    (row.apex_x_m, row.apex_y_m), fontsize=7, color=ink, ha="center")
    ax.set(aspect="equal", xlabel="east (m)", ylabel="north (m)", title="Track map, coloured by speed")
    fig.tight_layout(); fig.savefig(out / "track_speed.png", dpi=150); plt.close(fig)

    fig, axs = plt.subplots(5, 1, figsize=(11, 10), sharex=True)
    lap = drv[(drv["lap"] >= 10) & (drv["lap"] <= 12)]
    for a, (col, lab, c) in zip(axs, [("speed_ms", "speed (m/s)", blue),
                                      ("radius_m", "radius (m), + left", orange),
                                      ("steer_pct", "steering (%)", aqua),
                                      ("throttle_pct", "throttle (%)", blue),
                                      ("roll_deg", "roll (deg), + right down", orange)]):
        a.plot(lap["t"], lap[col], color=c, lw=1.2)
        a.set_ylabel(lab)
    axs[1].plot(lap["t"], lap["radius_gps_m"], color=muted, lw=0.8, label="from GPS course")
    axs[1].set_ylim(-15, 15); axs[1].legend(loc="upper right", frameon=False)
    axs[-1].set_xlabel("time (s, IMU clock)")
    axs[0].set_title("Laps 10-12")
    fig.tight_layout(); fig.savefig(out / "laps_timeseries.png", dpi=150); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    m = drv["speed_ms"] > DRIVE_SPEED
    axs[0].scatter(drv.loc[m, "steer_pct"], 1 / drv.loc[m, "radius_m"], s=2, color=blue, alpha=0.3)
    axs[0].set(xlabel="steering (%)", ylabel="curvature 1/R (1/m), + left", title="Steering vs curvature")
    axs[1].scatter(drv.loc[m, "ay_g"], drv.loc[m, "roll_deg"], s=2, color=orange, alpha=0.3)
    axs[1].set(xlabel="lateral accel (g), + left", ylabel="roll (deg), + right side down", title="Roll vs lateral accel")
    fig.tight_layout(); fig.savefig(out / "steer_roll.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
