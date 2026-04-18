import os
import pandas as pd
import numpy as np
import joblib
from scipy.stats import skew, kurtosis
from app.ml.sensor_configs import SENSOR_CONFIGS, LABEL_ROOT, DATA_ROOT


class MasterDataLoader:
    """
    Orchestrates the loading and early fusion of data from all available sensors.

    Strategy:
    - Aligns every sensor stream to the 1Hz behavior label timeline using pd.merge_asof.
    - Each sensor gets rolling window statistics (mean, std, min, max, skew, kurtosis)
      computed at its native frequency before alignment.
    - Missing sensors produce NaN columns — XGBoost handles these natively, enabling
      graceful degradation when a sensor is offline.
    - Labels are ordinally encoded to integer class IDs for XGBoost compatibility.
    """

    # Canonical behavior classes — every label file must map to these.
    BEHAVIOR_CLASSES = ["Grazing", "Ruminating", "Resting", "Walking", "Other"]

    def __init__(self, sampling_rate: float = 1.0):
        self.sampling_rate = sampling_rate
        self.sensor_configs = SENSOR_CONFIGS
        # Populated after loading so the training script can inspect them.
        self.feature_names: list[str] = []
        self.label_encoder: dict[str, int] = {}

    # ------------------------------------------------------------------
    #  Public helpers
    # ------------------------------------------------------------------

    def get_cow_date_pairs(self) -> list[tuple[str, str, str]]:
        """
        Scans LABEL_ROOT and returns [(cow_id, date_suffix, file_path), ...].
        date_suffix is the raw filename part, e.g. '0725.csv'.
        """
        pairs = []
        if not os.path.exists(LABEL_ROOT):
            print(f"[MasterDataLoader] LABEL_ROOT not found: {LABEL_ROOT}")
            return pairs
        for f in os.listdir(LABEL_ROOT):
            if f.endswith(".csv"):
                parts = f.split("_")
                if len(parts) < 2:
                    continue
                cow_id = parts[0]
                date_suffix = parts[1]
                pairs.append((cow_id, date_suffix, os.path.join(LABEL_ROOT, f)))
        return pairs

    def get_feature_names(self) -> list[str]:
        """Returns the ordered list of feature column names from the last load call."""
        return list(self.feature_names)

    # ------------------------------------------------------------------
    #  Main loading interfaces
    # ------------------------------------------------------------------

    def load_fused_data(self, cow_ids=None, dates=None) -> pd.DataFrame:
        """
        Loads and fuses data for specific cows / dates.

        Args:
            cow_ids: list of cow ID strings, e.g. ['C001']. None = all.
            dates:   list of MMDD strings, e.g. ['1010'].  None = all.
        Returns:
            Fused DataFrame with feature columns + 'behavior' (int class ID).
        """
        pairs = self.get_cow_date_pairs()
        if cow_ids:
            pairs = [p for p in pairs if p[0] in cow_ids]
        if dates:
            pairs = [p for p in pairs if p[1].replace(".csv", "") in dates]
        return self._align_pairs(pairs)

    def load_fused_dataset(self, limit_files: int | None = None) -> pd.DataFrame:
        """
        Loads all available aligned data. Optionally cap to *limit_files* label files
        for quick sanity checks.
        """
        pairs = self.get_cow_date_pairs()
        if limit_files:
            pairs = pairs[:limit_files]
        return self._align_pairs(pairs)

    def stratified_sample(
        self,
        cow_date_pairs: list[tuple],
        max_sessions: int = 50,
        random_state: int = 42,
    ) -> list[tuple]:
        """
        Stratified Date-Cow Sampling:
        Randomly selects *max_sessions* (cow, date) sessions to fit in memory
        while preserving temporal continuity within each selected session.
        """
        rng = np.random.default_rng(random_state)
        if len(cow_date_pairs) <= max_sessions:
            return cow_date_pairs
        indices = rng.choice(len(cow_date_pairs), size=max_sessions, replace=False)
        return [cow_date_pairs[i] for i in indices]

    # ------------------------------------------------------------------
    #  Sensor loading
    # ------------------------------------------------------------------

    def load_sensor_data(
        self, sensor_name: str, cow_id: str, date_suffix: str
    ) -> dict | None:
        """
        Loads one sensor file for a given cow + date and computes rolling features.

        Returns:
            {"type": "dynamic", "data": <DataFrame with 'timestamp' + feature cols>}
            {"type": "static",  "data": <dict of column→scalar>}
            None  →  file not found / unreadable (sensor will produce NaN columns).
        """
        config = self.sensor_configs[sensor_name]
        is_global = config.get("is_global", False)
        is_excel = config.get("is_excel", False)
        date_str = date_suffix.replace(".csv", "")

        sensor_file = self._resolve_sensor_file(
            sensor_name, config, cow_id, date_str, is_global
        )

        if not sensor_file or not os.path.exists(sensor_file):
            return None

        try:
            df = pd.read_excel(sensor_file) if is_excel else pd.read_csv(sensor_file)
        except Exception as e:
            print(f"  [load_sensor_data] Error reading {sensor_file}: {e}")
            return None

        # ---- Timestamp normalization ----------------------------------------
        if "timestamp" not in df.columns:
            if sensor_name == "weather":
                df = self._add_weather_timestamp(df, date_str)
            else:
                # No timestamp → treat as daily static feature
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                return {"type": "static", "data": df[num_cols].mean().to_dict()}

        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("float64")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        if df.empty:
            return None

        # ---- Rolling features ----------------------------------------------
        freq = config.get("freq", 1)
        window_size = config.get("window_size", 10)
        win_samples = max(1, int(window_size * freq))

        num_cols = [
            c
            for c in df.select_dtypes(include=[np.number]).columns
            if c != "timestamp"
        ]
        if not num_cols:
            return None

        feats_df = pd.DataFrame({"timestamp": df["timestamp"]})
        for col in num_cols:
            series = df[col]
            roll = series.rolling(window=win_samples, center=True, min_periods=1)
            prefix = f"{sensor_name}_{col}"
            feats_df[f"{prefix}_mean"] = roll.mean()
            feats_df[f"{prefix}_std"] = roll.std()
            feats_df[f"{prefix}_min"] = roll.min()
            feats_df[f"{prefix}_max"] = roll.max()
            # Use native C-vectorized pandas rolling for a ~1000x speedup
            feats_df[f"{prefix}_skew"] = roll.skew().fillna(0.0)
            feats_df[f"{prefix}_kurt"] = roll.kurt().fillna(0.0)

        return {"type": "dynamic", "data": feats_df}

    # ------------------------------------------------------------------
    #  Internal alignment helpers
    # ------------------------------------------------------------------

    def _align_pairs(self, pairs: list[tuple]) -> pd.DataFrame:
        """Iterates over (cow_id, date_suffix, label_path) tuples and fuses them."""
        all_aligned: list[pd.DataFrame] = []

        for cow_id, date_suffix, label_path in pairs:
            print(f"[Fusing] {cow_id} / {date_suffix} ...")
            try:
                session_df = self._align_single_session(cow_id, date_suffix, label_path)
                if session_df is not None and not session_df.empty:
                    all_aligned.append(session_df)
            except Exception as e:
                print(f"  [Error] {label_path}: {e}")

        if not all_aligned:
            print("[MasterDataLoader] No sessions loaded. Returning empty DataFrame.")
            return pd.DataFrame()

        final_df = pd.concat(all_aligned, axis=0, ignore_index=True)

        # Persist feature column names (everything except 'behavior' and metadata)
        meta_cols = {"timestamp", "behavior", "datetime", "label_idx", "sensor_idx"}
        self.feature_names = [c for c in final_df.columns if c not in meta_cols]

        print(
            f"[MasterDataLoader] Fused {len(all_aligned)} sessions -> "
            f"{final_df.shape[0]} rows x {len(self.feature_names)} features"
        )
        return final_df

    def _align_single_session(
        self, cow_id: str, date_suffix: str, label_path: str
    ) -> pd.DataFrame | None:
        """
        Merges all sensor feature streams onto the 1Hz label timeline for one session.
        Each sensor is independently aligned via merge_asof; missing data → NaN.
        """
        # --- Load labels -------------------------------------------------------
        try:
            labels_df = pd.read_csv(label_path)
        except Exception as e:
            print(f"  [Error] reading labels {label_path}: {e}")
            return None

        labels_df.columns = labels_df.columns.str.strip()
        if "behavior" not in labels_df.columns:
            print(f"  [Warning] 'behavior' column missing in {label_path}")
            return None

        labels_df["timestamp"] = pd.to_numeric(
            labels_df["timestamp"], errors="coerce"
        ).astype("float64")
        labels_df = labels_df.dropna(subset=["timestamp"]).sort_values("timestamp")

        # Encode string behavior labels -> integer class IDs
        labels_df["behavior"] = self._encode_labels(labels_df["behavior"])

        fused_df = labels_df.copy()
        sensors_ok = 0

        # --- Merge each sensor -------------------------------------------------
        for sensor_name in self.sensor_configs:
            try:
                result = self.load_sensor_data(sensor_name, cow_id, date_suffix)
                if result is None:
                    # Insert NaN columns so the feature schema stays consistent
                    self._insert_nan_columns(fused_df, sensor_name)
                    continue

                sensors_ok += 1
                if result["type"] == "static":
                    for col, val in result["data"].items():
                        col_name = f"{sensor_name}_{col}_static"
                        fused_df[col_name] = val
                else:
                    sensor_feats = result["data"].sort_values("timestamp")
                    fused_df = pd.merge_asof(
                        fused_df,
                        sensor_feats,
                        on="timestamp",
                        direction="nearest",
                        tolerance=30,  # max 30s gap; beyond → NaN (graceful degradation)
                    )
            except Exception as e:
                print(f"  [Error] merging sensor '{sensor_name}': {e}")

        fused_df["_cow_id"] = cow_id
        fused_df["_date"] = date_suffix.replace(".csv", "")

        print(
            f"  Merged {sensors_ok}/{len(self.sensor_configs)} sensors -> "
            f"{fused_df.shape[0]} rows, {fused_df.shape[1]} cols"
        )
        return fused_df

    # ------------------------------------------------------------------
    #  Utility / private helpers
    # ------------------------------------------------------------------

    def _resolve_sensor_file(
        self,
        sensor_name: str,
        config: dict,
        cow_id: str,
        date_str: str,
        is_global: bool,
    ) -> str | None:
        """Returns the filesystem path for a sensor file, or None if not determinable."""
        if is_global:
            if sensor_name == "weather":
                mm, dd = date_str[:2], date_str[2:]
                return os.path.join(
                    DATA_ROOT, config["sub_path"], f"{mm}_{dd}_2023.xlsx"
                )
            elif "file_name" in config:
                return os.path.join(DATA_ROOT, config["sub_path"], config["file_name"])
            return None

        sensor_id = (
            cow_id.replace("C", "T")
            if config.get("id_prefix") == "T"
            else cow_id
        )
        sensor_folder = os.path.join(DATA_ROOT, config["sub_path"], sensor_id)

        if os.path.isdir(sensor_folder):
            # Look for a date-specific file inside the folder
            for fname in os.listdir(sensor_folder):
                # Match by date string (e.g. '0725') anywhere in the filename
                if date_str in fname and fname.endswith(".csv"):
                    return os.path.join(sensor_folder, fname)
            return None  # Date not found
        else:
            # Single-file sensor (e.g. milk daily CSV)
            candidate = os.path.join(DATA_ROOT, config["sub_path"], f"{sensor_id}.csv")
            return candidate if os.path.exists(candidate) else None

    def _add_weather_timestamp(self, df: pd.DataFrame, date_str: str) -> pd.DataFrame:
        """Converts a weather 'Time' column (e.g. '12:00 AM') to a Unix timestamp."""
        year, mm, dd = "2023", date_str[:2], date_str[2:]

        def parse_time(t_str):
            try:
                return pd.to_datetime(f"{year}-{mm}-{dd} {t_str}").timestamp()
            except Exception:
                return np.nan

        df = df.copy()
        df["timestamp"] = df["Time"].apply(parse_time)
        return df

    def _encode_labels(self, series: pd.Series) -> pd.Series:
        """
        Maps string behavior labels to integer class IDs.
        Builds/extends self.label_encoder on first call.
        Unknown labels get assigned the next available integer.
        """
        unique_labels = series.dropna().unique()
        for lbl in unique_labels:
            if lbl not in self.label_encoder:
                self.label_encoder[lbl] = len(self.label_encoder)
        return series.map(self.label_encoder).fillna(-1).astype(int)

    def _insert_nan_columns(self, df: pd.DataFrame, sensor_name: str) -> None:
        """
        When a sensor file is absent, insert NaN placeholder columns so the
        fused feature schema is consistent across sessions.
        This preserves XGBoost's ability to handle missingness natively.
        """
        config = self.sensor_configs.get(sensor_name, {})
        feature_cols = config.get("feature_cols", [])
        suffixes = ["mean", "std", "min", "max", "skew", "kurt"]
        for col in feature_cols:
            for suf in suffixes:
                placeholder = f"{sensor_name}_{col}_{suf}"
                if placeholder not in df.columns:
                    df[placeholder] = np.nan


# ---------------------------------------------------------------------------
#  Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    loader = MasterDataLoader()
    fused = loader.load_fused_dataset(limit_files=1)
    if not fused.empty:
        print(f"\nFused shape  : {fused.shape}")
        print(f"Feature count: {len(loader.get_feature_names())}")
        print(f"Behaviors    : {fused['behavior'].value_counts().to_dict()}")
        print(f"Label map    : {loader.label_encoder}")
        print(f"Sample cols  : {fused.columns.tolist()[:10]}")
    else:
        print("No data loaded — check DATA_ROOT / LABEL_ROOT in sensor_configs.py")
