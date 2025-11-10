"""
wearable_pipeline.py
--------------------
Integration helpers for pulling wearable data (Polar AccessLink OAuth2), ingesting local files (CSV/TCX),
building real-time sliding-window features, and running predictions with your lactate & recovery models.

Usage (quick demo):
    from wearable_pipeline import PolarClient, MockStream, run_realtime_loop
    from model_utils import load_model, make_features, predict_lactate, predict_recovery

    lactate_model = load_model("models/lactate_lightgbm_model.joblib")
    recovery_model = load_model("models/recovery_lightgbm_model.joblib")

    # Mock a 1 Hz live feed for 3 minutes
    stream = MockStream(freq_hz=1).generator(duration_sec=180)
    run_realtime_loop(stream, lactate_model, recovery_model, window_sec=30)

Environment variables expected for OAuth2 web flow (recommended to set via .env or Streamlit Cloud secrets):
    POLAR_CLIENT_ID=your_client_id
    POLAR_CLIENT_SECRET=your_client_secret
    POLAR_REDIRECT_URI=https://your-redirect-url

Notes:
- Polar AccessLink docs: https://www.polar.com/accesslink-api/
- For hackathon/demo, the mock simulator ensures your app can run without a device
"""

import os
import time
import math
import json
import queue
import base64
import logging
import threading
from dataclasses import dataclass
from typing import Dict, Iterable, Generator, Optional, Tuple

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from xml.etree import ElementTree as ET

# Optional model imports (safe if not present at import time)
try:
    from model_utils import prepare_features, make_features, predict_lactate, predict_recovery
except Exception:
    prepare_features = None
    make_features = None
    predict_lactate = None
    predict_recovery = None

logger = logging.getLogger("wearable_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ==========================================================
# Polar OAuth2 Web Flow Client
# ==========================================================
POLAR_AUTH_URL = "https://flow.polar.com/oauth2/authorization"
POLAR_TOKEN_URL = "https://polarremote.com/v2/oauth2/token"
POLAR_ACCESSLINK_BASE = "https://www.polaraccesslink.com/v3"


@dataclass
class PolarToken:
    access_token: str
    refresh_token: str
    expires_at: float  # epoch seconds

    @property
    def is_expired(self) -> bool:
        # Renew 60s early
        return time.time() > (self.expires_at - 60)


class PolarClient:
    """
    OAuth2 Client for Polar AccessLink (Web Redirect Flow).
    You will host the redirect URI and capture 'code' to exchange for tokens.
    """

    def __init__(self,
                 client_id: Optional[str] = None,
                 client_secret: Optional[str] = None,
                 redirect_uri: Optional[str] = None):
        self.client_id = client_id or os.getenv("POLAR_CLIENT_ID", "")
        self.client_secret = client_secret or os.getenv("POLAR_CLIENT_SECRET", "")
        self.redirect_uri = redirect_uri or os.getenv("POLAR_REDIRECT_URI", "")
        self.token: Optional[PolarToken] = None

        if not self.client_id or not self.client_secret or not self.redirect_uri:
            logger.warning("PolarClient: Missing POLAR_CLIENT_ID/SECRET/REDIRECT_URI env vars. OAuth will not work until set.")

    # ---------- Auth URLs ----------
    def build_auth_url(self, state: str = "xyz") -> str:
        """
        Returns the URL the user should open in a browser to authorize your app.
        After authorization, Polar redirects to redirect_uri?code=...&state=...
        """
        return (
            f"{POLAR_AUTH_URL}?response_type=code"
            f"&client_id={self.client_id}"
            f"&redirect_uri={self.redirect_uri}"
            f"&state={state}"
        )

    def exchange_code_for_token(self, code: str) -> PolarToken:
        """
        Exchange the 'code' from the redirect for an access + refresh token.
        """
        auth_str = f"{self.client_id}:{self.client_secret}".encode("utf-8")
        basic = base64.b64encode(auth_str).decode("utf-8")
        headers = {"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
        }
        resp = requests.post(POLAR_TOKEN_URL, headers=headers, data=data, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        self.token = PolarToken(
            access_token=payload["access_token"],
            refresh_token=payload.get("refresh_token", ""),
            expires_at=time.time() + payload.get("expires_in", 3600),
        )
        return self.token

    def refresh_access_token(self) -> PolarToken:
        if not self.token or not self.token.refresh_token:
            raise RuntimeError("No refresh token available")
        auth_str = f"{self.client_id}:{self.client_secret}".encode("utf-8")
        basic = base64.b64encode(auth_str).decode("utf-8")
        headers = {"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded"}
        data = {"grant_type": "refresh_token", "refresh_token": self.token.refresh_token}
        resp = requests.post(POLAR_TOKEN_URL, headers=headers, data=data, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        self.token = PolarToken(
            access_token=payload["access_token"],
            refresh_token=payload.get("refresh_token", self.token.refresh_token),
            expires_at=time.time() + payload.get("expires_in", 3600),
        )
        return self.token

    # ---------- API helpers ----------
    def _auth_headers(self) -> Dict[str, str]:
        if not self.token:
            raise RuntimeError("No token, please authenticate first")
        if self.token.is_expired:
            self.refresh_access_token()
        return {"Authorization": f"Bearer {self.token.access_token}"}

    def get_user_information(self) -> dict:
        url = f"{POLAR_ACCESSLINK_BASE}/users"
        r = requests.get(url, headers=self._auth_headers(), timeout=30)
        r.raise_for_status()
        return r.json()

    def get_daily_continuous_hr(self, date_from: str, date_to: str) -> dict:
        """
        Fetch continuous heart rate summary (if enabled on user account).
        date_from/to in 'YYYY-MM-DD'.
        """
        url = f"{POLAR_ACCESSLINK_BASE}/users/continuous-heart-rate"
        params = {"from": date_from, "to": date_to}
        r = requests.get(url, headers=self._auth_headers(), params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def list_exercises(self, date_from: str, date_to: str) -> dict:
        """
        List exercises/sessions in a date range.
        """
        url = f"{POLAR_ACCESSLINK_BASE}/exercises"
        params = {"from": date_from, "to": date_to}
        r = requests.get(url, headers=self._auth_headers(), params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def get_exercise(self, exercise_id: str) -> dict:
        url = f"{POLAR_ACCESSLINK_BASE}/exercises/{exercise_id}"
        r = requests.get(url, headers=self._auth_headers(), timeout=30)
        r.raise_for_status()
        return r.json()

    def get_exercise_samples(self, exercise_id: str) -> dict:
        """
        Fetch samples for a specific exercise (HR, power, speed). Polar returns links to actual samples; follow them.
        """
        url = f"{POLAR_ACCESSLINK_BASE}/exercises/{exercise_id}/samples"
        r = requests.get(url, headers=self._auth_headers(), timeout=30)
        r.raise_for_status()
        return r.json()


# ==========================================================
# Local File Ingestion (CSV/TCX)
# ==========================================================

def read_csv_session(path: str) -> pd.DataFrame:
    """
    Read a CSV of time-series with columns like: time, heart_rate, power, pace (flexible names accepted).
    Ensures a clean DataFrame with at least: time(s), heart_rate(bpm), power(W), pace(m/s or min/km).
    """
    df = pd.read_csv(path)
    # Normalize column names
    cols = {c.lower().strip(): c for c in df.columns}
    def get_col(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    time_col = get_col("time","timestamp","sec","seconds")
    hr_col = get_col("heart_rate","hr","bpm")
    power_col = get_col("power","watts","w")
    pace_col = get_col("pace","speed","velocity")

    # Basic checks
    if time_col is None:
        # generate seconds if missing
        df["time"] = np.arange(len(df))
        time_col = "time"
    if hr_col is None:
        df["heart_rate"] = np.nan
        hr_col = "heart_rate"
    if power_col is None:
        df["power"] = np.nan
        power_col = "power"
    if pace_col is None:
        df["pace"] = np.nan
        pace_col = "pace"

    out = pd.DataFrame({
        "time": pd.to_numeric(df[time_col], errors="coerce"),
        "heart_rate": pd.to_numeric(df[hr_col], errors="coerce"),
        "power": pd.to_numeric(df[power_col], errors="coerce"),
        "pace": pd.to_numeric(df[pace_col], errors="coerce"),
    }).fillna(method="ffill").fillna(0)
    return out


def read_tcx_session(path: str) -> pd.DataFrame:
    """
    Parse a TCX file (Training Center XML) and extract time, HR, speed (pace), and power if available.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    ns = {"tcx":"http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}

    times, hr, speed, power = [], [], [], []
    for tp in root.findall(".//tcx:Trackpoint", ns):
        tm = tp.findtext("./tcx:Time", default=None, namespaces=ns)
        if tm:
            try:
                dt = datetime.fromisoformat(tm.replace("Z","+00:00"))
            except Exception:
                try:
                    dt = datetime.strptime(tm, "%Y-%m-%dT%H:%M:%S.%fZ")
                except Exception:
                    continue
            times.append(dt)

            h = tp.findtext("./tcx:HeartRateBpm/tcx:Value", default=None, namespaces=ns)
            hr.append(float(h) if h else np.nan)

            spd = tp.findtext("./tcx:Extensions/tcx:TPX/tcx:Speed", default=None, namespaces=ns)
            speed.append(float(spd) if spd else np.nan)

            pwr = tp.findtext("./tcx:Extensions/tcx:TPX/tcx:Watts", default=None, namespaces=ns)
            power.append(float(pwr) if pwr else np.nan)

    if not times:
        return pd.DataFrame(columns=["time","heart_rate","power","pace"])

    # Convert times to relative seconds
    t0 = times[0]
    sec = [(t - t0).total_seconds() for t in times]
    df = pd.DataFrame({
        "time": sec,
        "heart_rate": pd.Series(hr).fillna(method="ffill").fillna(0),
        "power": pd.Series(power).fillna(method="ffill").fillna(0),
        "pace": pd.Series(speed).fillna(method="ffill").fillna(0),  # speed may be m/s
    })
    return df


# ==========================================================
# Real-time Streaming (Mock and Runner)
# ==========================================================

class MockStream:
    """
    Generates synthetic HR, power, pace at a desired frequency for demos/testing.
    Produces a dict per tick: {"time": t, "heart_rate": bpm, "power": w, "pace": m_per_s}
    """

    def __init__(self, freq_hz: int = 1, hr_base: int = 140, power_base: int = 220, seed: int = 42):
        self.freq_hz = int(freq_hz)
        self.dt = 1.0 / max(1, self.freq_hz)
        self.hr_base = hr_base
        self.power_base = power_base
        self.rng = np.random.default_rng(seed)

    def generator(self, duration_sec: int = 180) -> Generator[Dict, None, None]:
        t = 0.0
        for _ in range(int(duration_sec * self.freq_hz)):
            # Simple physiology-ish trends: HR drifts up slowly; power oscillates intervals
            interval = 60.0
            interval_factor = 0.5 * (1 + math.sin(2*math.pi*(t/interval)))
            power = self.power_base * (0.8 + 0.4 * interval_factor) + self.rng.normal(0, 5)
            hr = self.hr_base + 0.08 * t + self.rng.normal(0, 1.0) + 0.02 * (power - self.power_base)
            pace = 3.0 + self.rng.normal(0, 0.05)  # m/s placeholder

            yield {"time": t, "heart_rate": float(hr), "power": float(power), "pace": float(pace)}
            t += self.dt
            time.sleep(self.dt)


def run_realtime_loop(stream: Iterable[Dict],
                      lactate_model,
                      recovery_model=None,
                      window_sec: int = 30) -> None:
    """
    Consume a (real or mock) stream of dicts with keys time, heart_rate, power, pace.
    Maintain a sliding window, compute features, and print live predictions to console.
    Hook this into Streamlit for live charts.
    """
    if make_features is None or predict_lactate is None:
        logger.warning("model_utils functions not available; install/import model_utils.py for live inference")
        return

    buffer = []
    for sample in stream:
        buffer.append(sample)
        # Keep only last 'window_sec' of data based on time stamps
        t_now = sample["time"]
        buffer = [s for s in buffer if (t_now - s["time"]) <= window_sec]

        df_win = pd.DataFrame(buffer)
        feats = make_features(df_win.rename(columns={"hr": "heart_rate"}) if "hr" in df_win.columns else df_win)
        lac = float(predict_lactate(lactate_model, feats)[0])
        msg = f"[t={t_now:5.1f}s] Lactate ≈ {lac:.2f} mmol/L"

        if recovery_model is not None and predict_recovery is not None:
            rec = float(predict_recovery(recovery_model, feats)[0])
            msg += f" | Recovery ≈ {rec:.0f}/100"

        print(msg)


# ==========================================================
# Batch Prediction Helpers
# ==========================================================

def predict_from_session_df(df: pd.DataFrame, lactate_model, recovery_model=None) -> pd.DataFrame:
    """
    Given a full session DataFrame with columns ['time','heart_rate','power','pace'],
    compute features and return a DataFrame with predictions appended.
    """
    if prepare_features is None or predict_lactate is None:
        raise RuntimeError("model_utils.prepare_features / predict_lactate not available")

    df_feat = prepare_features(df)
    y_lac = predict_lactate(lactate_model, df_feat)
    out = df_feat.copy()
    out["pred_lactate"] = y_lac

    if recovery_model is not None and predict_recovery is not None:
        out["pred_recovery"] = predict_recovery(recovery_model, df_feat)
    return out


# ==========================================================
# Minimal CLI demo
# ==========================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Wearable pipeline demo")
    parser.add_argument("--csv", help="Path to session CSV")
    parser.add_argument("--tcx", help="Path to session TCX")
    parser.add_argument("--mock", action="store_true", help="Run mock real-time stream")
    parser.add_argument("--window", type=int, default=30, help="Sliding window seconds for live mode")
    args = parser.parse_args()

    if args.mock:
        print("Starting mock stream... (Ctrl+C to stop)")
        stream = MockStream(freq_hz=1).generator(duration_sec=120)
        run_realtime_loop(stream, lactate_model=None)  # Note: supply models in your app

    elif args.csv or args.tcx:
        if args.csv:
            df = read_csv_session(args.csv)
        else:
            df = read_tcx_session(args.tcx)
        print(df.head())
        print("→ Load models in your app and call predict_from_session_df(df, lactate_model, recovery_model)")

    else:
        print("Nothing to do. Use --mock or provide --csv/--tcx.")
