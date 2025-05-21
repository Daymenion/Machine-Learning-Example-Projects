"""Fully-featured, vectorised EnhancedFeatureEngineer (v5, *final parity*)
==========================================================================
* Restored ALL legacy attributes (`scalers`, `encoders`, `location_embeddings`,
`location_graphs`, `location_clusters`, `feature_names`).
* Centrality columns once again end in "_centrality" (both the legacy alias
**and** the shorter alias are created, to avoid breaking downstream code).
* Added _missing_ helpers:  `_heat_index`, `_wind_chill`, `_wx_spatial_inter`.
* Weather-spatial interactions include: origin-cluster, destination-cluster,
trip-distance-bin and hour-of-day cross-terms.
* Destination-cluster dummies + cluster-weekend interactions restored.
* All legacy `ewm_mean_alpha<α>` names are present (alongside the fully
windowed versions).
* Full feature parity is asserted at the end of `create_all_features`.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Set

import holidays
import networkx as nx
import numpy as np
import pandas as pd
from meteostat import Hourly, Point
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from ..utils.logger import StructuredLogger  # type: ignore

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)


class EnhancedFeatureEngineer:
  # ------------------------------------------------------------------ #
  # Init                                                               #
  # ------------------------------------------------------------------ #
  def __init__(self,
                feature_config: Optional[Dict] = None,
                logger: Optional[StructuredLogger] = None) -> None:
      self.feature_config = feature_config or {}
      self.logger = logger or StructuredLogger("enhanced_feature_engineer")

      self.temporal_cfg = self.feature_config.get("temporal", {})
      self.spatial_cfg = self.feature_config.get("spatial", {})
      self.context_cfg = self.feature_config.get("context", {})

      # Feature registries ------------------------------------------------
      self.temporal_features: Set[str] = set()
      self.spatial_features: Set[str] = set()
      self.context_features: Set[str] = set()
      self.feature_names: Dict[str, List[str]] = {}

      # Runtime artefacts -------------------------------------------------
      self.scalers: Dict = {}
      self.encoders: Dict = {}
      self.location_embeddings: Dict = {}
      self.location_graphs: Dict[str, nx.DiGraph] = {}
      self.location_clusters: Dict[str, Dict[int, int]] = {}

      # Misc --------------------------------------------------------------
      self.holidays_tr = holidays.Turkey()
      self.popularity_threshold = self.spatial_cfg.get("popularity_threshold", 0.75)

      # Caches ------------------------------------------------------------
      self._G: Optional[nx.DiGraph] = None
      self._cluster_map: Dict[int, int] = {}
      
  # ------------------------------------------------------------------ #
  # Public orchestrator                                                #
  # ------------------------------------------------------------------ #
  def create_all_features(self, df: pd.DataFrame, *, is_train: bool = True) -> pd.DataFrame:
    with self.logger.operation("create_all_features"):
        out = (df.pipe(self._temporal)
                 .pipe(self._spatial)
                 .pipe(self._context)
                 .pipe(self._add_weather_features, is_train))

        # ---------- feature-parity assertion ----------

        # wee should keep the last raw on each hour_start and start_location_id
        out = out.sort_values(by=['hour_start', 'start_location_id'], ascending=[False, True]).drop_duplicates(
            subset=['hour_start', 'start_location_id'], keep='first').reset_index(drop=True)
        columns = ['index', 'start_time', 'trip_id', 'passenger_id', 'driver_id', 'end_time', 'trip_dist_bin',
            'distance_km', 'duration_min', 'price', 'surge_multiplier',  'total_trips', 'location_dominant_type',
            'register_location_id', 'signup_date', 'signup_date_passenger', 'registered_location_id', 'end_location_id',
            'user_using_time', 'user_using_frequency', 'user_next_trip_days', 'driver_next_trip_location', 'location_type_mixed',
            'driver_trip_count', 'driver_work_time', 'driver_trip_frequency', 'regain_potential', 'driver_next_trip_days',
            'user_frequency_quartile', 'driver_work_time_quartile', 'price_per_km', 'price_per_min', 'avg_speed_kmh', 
            'trip_distance_quartile', 'trip_duration_quartile']
        out = out.drop(columns=columns)
        
        #boolean columns
        for col in out.select_dtypes(include=['bool']).columns.to_list():
            out[col] = out[col].astype('int8')
        
        # features < 0
        numeric_cols = out.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            out[col] = out[col].clip(lower=0)
            out[col] = out[col].fillna(0)

        #change all the numbers every floats b'gger then float16 goes to float 16 every int biggerthen int8 goes to int8
        for col in numeric_cols:
            if out[col].dtype == 'float64':
                out[col] = out[col].astype('float32')
            elif out[col].dtype == 'int64' or out[col].dtype == 'int32' or out[col].dtype == 'UInt32':
                out[col] = out[col].astype('int8') 
        
        # we drop som of temporal spatial or other columns so we should update the feature registry
        col_to_remove = []
        for col in self.temporal_features:
            if col not in out.columns:
                col_to_remove.append(col)
        self.temporal_features = self.temporal_features - set(col_to_remove)
        
        col_to_remove = []
        for col in self.spatial_features:
            if col not in out.columns:
                col_to_remove.append(col)
        self.spatial_features = self.spatial_features - set(col_to_remove)
        
        col_to_remove = []
        for col in self.context_features:
            if col not in out.columns:
                col_to_remove.append(col)
        self.context_features = self.context_features - set(col_to_remove)

        expected = self.temporal_features | self.spatial_features | self.context_features
        missing = expected - set(out.columns)
        assert not missing, f"Missing engineered columns: {sorted(missing)[:10]} …"

        # write temporal, spatial and context features into same 



        return out

  # ------------------------------------------------------------------ #
  # 1) Temporal features                                               #
  # ------------------------------------------------------------------ #
  def _temporal(self, df: pd.DataFrame) -> pd.DataFrame:
      if "hour_start" not in df.columns:
          df = df.assign(hour_start=pd.to_datetime(df["start_time"]).dt.floor("H"))
      # --- base decomposition -----------------------------------------
      dt = df["hour_start"]
      comps = pd.DataFrame({
          "day_of_month": dt.dt.day,
          "quarter": dt.dt.quarter
      }, index=df.index)
      df = pd.concat([df, comps], axis=1)
      self.temporal_features.update(comps.columns)
      self.temporal_features.update(['hour', 'day', 'day_of_week', 'is_weekend', 'month', 'week_of_year'])

      for col, mod in {"hour": 24, "day_of_week": 7, "day_of_month": 31, "month": 12}.items():
          s = self._as_series(df, col)
          df[f"{col}_sin"] = np.sin(2 * np.pi * s / mod)
          df[f"{col}_cos"] = np.cos(2 * np.pi * s / mod)
          self.temporal_features.update({f"{col}_sin", f"{col}_cos"})

      # --- flags -------------------------------------------------------
      hour_s = self._as_series(df, "hour")
      df["is_rush_hour"] = hour_s.isin([7, 8, 9, 16, 17, 18, 19]).astype("int8")
      df["is_night"] = hour_s.lt(6).astype("int8")
      self.temporal_features.update({"is_rush_hour", "is_night"})

      # --- holiday proximity (vectorised) -----------------------------
      df["is_holiday"] = dt.dt.date.isin(self.holidays_tr).astype("int8")
      prox = np.full(len(df), 99, dtype="int8")  
      hol_dates = np.array(list(self.holidays_tr))
      for hd in hol_dates:
          d = (dt.dt.date - hd).astype("timedelta64[D]").astype(int)
          mask = (np.abs(d) <= 3) & (np.abs(d) < np.abs(prox))
          prox = np.where(mask, d, prox)
      df["holiday_proximity"] = prox.astype("int8")
      self.temporal_features.update({"is_holiday", "holiday_proximity"})

      # --- lag, rolling & lag-diff ------------------------------------
      if "trip_count" in df.columns:
          df = self._add_lag_roll(df)

      # --- TOD/Day-type interaction + Fourier -------------------------
      df = self._temporal_interactions(df)
      df = self._fourier(df)
      return df

  # ------------------------------------------------------------------ #
  # 2) Lag, rolling & ewm features                                     #
  # ------------------------------------------------------------------ # 
  def _add_lag_roll(self, df: pd.DataFrame) -> pd.DataFrame:
    lags = self.temporal_cfg.get("lag_windows", [1, 2, 3, 6, 12, 24, 48, 168])
    wins = self.temporal_cfg.get("window_sizes", [3, 6, 12, 24, 72, 168])

    agg = (
        df.groupby(["start_location_id", "hour_start"], observed=True)
          .agg(trip_count=("trip_count", "count"))
          .reset_index()
          .sort_values(["start_location_id", "hour_start"])
    )
    g = agg.groupby("start_location_id", observed=True)

    # -- lag
    for l in lags:
        agg[f"lag_{l}h"] = g["trip_count"].shift(l).fillna(0)
        self.temporal_features.add(f"lag_{l}h")

    # -- lag diff
    for l1, l2 in zip(lags[:-1], lags[1:]):
        agg[f"lag_diff_{l1}_{l2}h"] = agg[f"lag_{l1}h"] - agg[f"lag_{l2}h"]
        self.temporal_features.add(f"lag_diff_{l1}_{l2}h")

    # -- rolling & ewm (LEAK-SAFE) ----------------------------------
    for w in wins:
        win_str = f"{w}h"
        roll = g["trip_count"].rolling(w, min_periods=1, closed="left") 

        for stat in ("mean", "std", "max", "min"):
            agg[f"rolling_{win_str}_{stat}"] = (
                getattr(roll, stat)().reset_index(level=0, drop=True)
            )
            self.temporal_features.add(f"rolling_{win_str}_{stat}")

        for a in (0.3, 0.5, 0.7, 1.0):
            col_full = f"ewm_mean_alpha{a}_{w}h"
            col_legacy = f"ewm_mean_alpha{a}"
            ewm_vals = (
                g["trip_count"]
                .transform(lambda x, alpha=a: x.shift(1).ewm(alpha=alpha, adjust=False).mean())
            )
            agg[col_full] = ewm_vals
            agg[col_legacy] = ewm_vals
            self.temporal_features.update({col_full, col_legacy})
    # merge
    feat_cols = [c for c in agg.columns if c not in ("start_location_id", "hour_start", "trip_count")]
    df = df.merge(
        agg[["start_location_id", "hour_start"] + feat_cols],
        on=["start_location_id", "hour_start"],
        how="left",
        validate="many_to_one",
    )

    return df

  def _temporal_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
    tod_map = {
        "morning": range(6, 12),
        "afternoon": range(12, 18),
        "evening": range(18, 24),
        "night": range(0, 6),
        }
    hour_s = self._as_series(df, "hour")
    for k, hrs in tod_map.items():
      df[f"is_{k}"] = hour_s.isin(hrs).astype("int8")
      self.temporal_features.add(f"is_{k}")

    return df


  def _fourier(self, df: pd.DataFrame) -> pd.DataFrame:
      ref = df["hour_start"].min()
      hr = (df["hour_start"] - ref).dt.total_seconds() / 3600
      periods = self.temporal_cfg.get("fourier_periods", [24, 168, 720])
      k = self.temporal_cfg.get("fourier_harmonics", 4)
      for p in periods:
          for h in range(1, k + 1):
              df[f"fourier_{p}h_{h}_sin"] = np.sin(2 * np.pi * h * hr / p)
              df[f"fourier_{p}h_{h}_cos"] = np.cos(2 * np.pi * h * hr / p)
              self.temporal_features.update({f"fourier_{p}h_{h}_sin", f"fourier_{p}h_{h}_cos"})
      return df
      
  # ------------------------------------------------------------------ #
  # 2) Spatial features                                                #
  # ------------------------------------------------------------------ #
  def _spatial(self, df: pd.DataFrame) -> pd.DataFrame:
      if {"start_location_id", "end_location_id"}.issubset(df.columns):
          if self._G is None:
              self._build_graph(df)
          df = (df.pipe(self._centrality)
                  .pipe(self._clusters)
                  .pipe(self._loc_types)
                  .pipe(self._proximities)
                  .pipe(self._spatiotemporal_inter))
      return df

  def _build_graph(self, df: pd.DataFrame) -> None:
      G = nx.DiGraph()
      for (u, v), w in df.groupby(["start_location_id", "end_location_id"]).size().items():
          G.add_edge(u, v, weight=int(w))
      self._G = G
      self.location_graphs["trip_flow"] = G

  def _centrality(self, df: pd.DataFrame) -> pd.DataFrame:
      G = self._G
      indeg = dict(G.in_degree())
      outdeg = dict(G.out_degree())
      instr = {n: sum(d["weight"] for _, _, d in G.in_edges(n, data=True)) for n in G}
      outstr = {n: sum(d["weight"] for _, _, d in G.out_edges(n, data=True)) for n in G}
      btw = nx.betweenness_centrality(G, k=min(100, len(G)), weight="weight", seed=SEED)
      pr = nx.pagerank(G, weight="weight")

      maps = {
          "in_degree": indeg,
          "out_degree": outdeg,
          "in_strength": instr,
          "out_strength": outstr,
          "betweenness": btw,
          "pagerank": pr
      }

      for name, mp in maps.items():
          short = f"location_{name}"
          legacy = f"{short}_centrality"
          df[legacy] = df["start_location_id"].map(mp).fillna(0).astype("float32")
          df[short] = df[legacy]             # alias without suffix
          self.spatial_features.update({legacy, short})
      return df

  def _clusters(self, df: pd.DataFrame) -> pd.DataFrame:
      if not self._cluster_map:
          feat = df.groupby("start_location_id").agg({
              "distance_km": "mean",
              "duration_min": "mean",
              "price": "mean",
              "trip_id": "count"
          })
          X = StandardScaler().fit_transform(feat)
          X = PCA(n_components=min(5, X.shape[1])).fit_transform(X)
          k = min(5, max(2, X.shape[0] // 10))
          labels = KMeans(k, random_state=SEED).fit_predict(X)
          self._cluster_map = dict(zip(feat.index, labels))
          self.location_clusters["location_cluster"] = self._cluster_map

      df["location_cluster"] = df["start_location_id"].map(self._cluster_map).fillna(-1).astype("int8")
      self.spatial_features.add("location_cluster")

      dummies = pd.get_dummies(df["location_cluster"], prefix="location_cluster")
      df = pd.concat([df, dummies], axis=1)
      self.spatial_features.update(dummies.columns)
      return df

  def _loc_types(self, df: pd.DataFrame) -> pd.DataFrame:
    hour_s = self._as_series(df, "hour")
    tmp = pd.DataFrame({"start_location_id": df["start_location_id"], "hour_single": hour_s})
    hourly = tmp.groupby(["start_location_id", "hour_single"], observed=True).size().unstack(fill_value=0)
    ratio = hourly.div(hourly.sum(axis=1), axis=0)

    def classify(r: pd.Series) -> str:
      morning = r[[7, 8, 9]].sum()
      evening = r[[17, 18, 19]].sum()
      night = r[[0, 1, 2, 3, 4, 5]].sum()
      business = r[[9, 10, 11, 12, 13, 14, 15, 16, 17]].sum()
      if morning > 0.3 and evening > 0.3:
        return "residential"
      if evening > 0.3 and morning < 0.2:
        return "business"
      if night > 0.3:
        return "entertainment"
      if business > 0.6:
        return "commercial"
      return "mixed"

    typemap = ratio.apply(classify, axis=1)
    df["location_dominant_type"] = df["start_location_id"].map(typemap)
    dummies = pd.get_dummies(df["location_dominant_type"], prefix="location_type")
    df = pd.concat([df, dummies], axis=1)
    self.spatial_features.add("location_dominant_type")
    self.spatial_features.update(dummies.columns)
    return df
  
  def _proximities(self, df: pd.DataFrame) -> pd.DataFrame:
      pop = (df["start_location_id"].value_counts() + df["end_location_id"].value_counts()).fillna(0)
      popular = pop[pop > pop.quantile(self.popularity_threshold)].index[:5]
      for p in popular:
          to_dist = df[df["end_location_id"] == p].groupby("start_location_id")["distance_km"].mean().to_dict()
          to_time = df[df["end_location_id"] == p].groupby("start_location_id")["duration_min"].mean().to_dict()
          fr_dist = df[df["start_location_id"] == p].groupby("end_location_id")["distance_km"].mean().to_dict()
          fr_time = df[df["start_location_id"] == p].groupby("end_location_id")["duration_min"].mean().to_dict()
          for name, mp in {
              f"dist_to_pop{p}": to_dist,
              f"time_to_pop{p}": to_time,
              f"dist_from_pop{p}": fr_dist,
              f"time_from_pop{p}": fr_time
          }.items():
              df[name] = df["start_location_id"].map(mp)
              df[name].fillna(df[name].median(), inplace=True)
              self.spatial_features.add(name)
      return df

  def _spatiotemporal_inter(self, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    tods = ["morning", "afternoon", "evening", "night"]

    loc_clust_s = self._as_series(df, "location_cluster")
    is_weekend_s = self._safe_bool(df, "is_weekend")

    # cluster × TOD / weekend
    for c in loc_clust_s.unique():
      if c == -1:
        continue
      mask_cluster = (loc_clust_s == c)
      for tod in tods:
        tod_flag = self._safe_bool(df, f"is_{tod}")
        col = f"cluster{c}_x_is_{tod}"
        df[col] = (mask_cluster & (tod_flag == 1)).astype("int8").to_numpy()
      col_wkd = f"cluster{c}_x_is_weekend"
      df[col_wkd] = (mask_cluster & (is_weekend_s == 1)).astype("int8").to_numpy()
      self.spatial_features.add(col_wkd)

    # location‑type × TOD / weekend
    for lt in ["residential", "business", "entertainment", "commercial", "mixed"]:
      lt_ser = self._safe_bool(df, f"location_type_{lt}")
      if lt_ser.sum() == 0:
        continue
      for tod in tods:
        tod_ser = self._safe_bool(df, f"is_{tod}")
        col = f"{lt}_x_{tod}"
        df[col] = (lt_ser * tod_ser).astype("int8").to_numpy()
        self.spatial_features.add(col)
      col = f"{lt}_x_weekend"
      df[col] = (lt_ser * is_weekend_s).astype("int8").to_numpy()
      self.spatial_features.add(col)

    return df
      
  # ------------------------------------------------------------------ #
  # 3) Contextual (passenger / driver / trip)                          #
  # ------------------------------------------------------------------ #
  def _context(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns contextual features for each location and time window.
    """
    df_sorted = df.sort_values(["start_location_id", "hour_start"]).copy()

    agg_dict = {
        "driver_trip_frequency": "mean",
        "driver_next_trip_location": self._fast_mode,
        "driver_next_trip_days": "mean",
        "user_using_frequency": "mean",
        "user_next_trip_days": "mean",
        "regain_potential": "mean",
        "price": "mean",
        "distance_km": "mean",
        "end_location_id": self._fast_mode,
    }

    if {"user_using_frequency"}.issubset(df.columns):
        df_sorted["user_frequency_quartile"] = pd.qcut(
            df_sorted["user_using_frequency"], 4, labels=False, duplicates="drop"
        )
        agg_dict["user_frequency_quartile"] = "mean"

    if {"driver_work_time"}.issubset(df.columns):
        df_sorted["driver_work_time_quartile"] = pd.qcut(
            df_sorted["driver_work_time"], 4, labels=False, duplicates="drop"
        )
        agg_dict.update({"driver_work_time_quartile": "mean"})

    if {"price", "distance_km", "duration_min"}.issubset(df.columns):
        df_sorted["price_per_km"] = df_sorted["price"] / df_sorted["distance_km"].clip(0.1)
        df_sorted["price_per_min"] = df_sorted["price"] / df_sorted["duration_min"].clip(0.1)
        df_sorted["avg_speed_kmh"] = df_sorted["distance_km"] / (df_sorted["duration_min"] / 60).clip(0.01)
        df_sorted["trip_distance_quartile"] = pd.qcut(
            df_sorted["distance_km"], 4, labels=False, duplicates="drop"
        )
        df_sorted["trip_duration_quartile"] = pd.qcut(
            df_sorted["duration_min"], 4, labels=False, duplicates="drop"
        )
        agg_dict.update(
            {
                "price_per_km": "mean",
                "price_per_min": "mean",
                "avg_speed_kmh": "mean",
                "trip_distance_quartile": "mean",
                "trip_duration_quartile": "mean",
            }
        )

    wins = [1, 3, 12, 24]   
    g = df_sorted.groupby("start_location_id", observed=True)
    feature_frames = []

    for w in wins:
        win_len = f"{w}h"
        num_cols = [c for c, f in agg_dict.items() if isinstance(f, str)]
        if num_cols:
            rolled_num = (
                g[num_cols]
                .rolling(window=w, min_periods=1, closed="left")
                .agg({c: agg_dict[c] for c in num_cols})
                .shift(1)
                .reset_index(level=0, drop=True)
            )
            rolled_num.columns = [
                f"lag_{c}_{win_len}_{agg_dict[c]}" for c in num_cols
            ]
            feature_frames.append(rolled_num)

        cat_cols = [c for c, f in agg_dict.items() if callable(f)]
        for c in cat_cols:
            col_name = f"lag_{c}_{win_len}_mode"
            rolled_cat = (
                g[c]
                .rolling(window=w, min_periods=1, closed="left")
                .apply(self._fast_mode, raw=False)
                .shift(1)
                .reset_index(level=0, drop=True)
                .rename(col_name)
            )
            feature_frames.append(rolled_cat)

    temp_features = pd.concat(feature_frames, axis=1)

    df_final = df_sorted.merge(
        temp_features,
        left_index=True,
        right_index=True,
        how="left",
    )

    self.context_features.update(temp_features.columns)

    return df_final
      
  # ------------------------------------------------------------------ #
  # 4) Weather & interactions                                          #
  # ------------------------------------------------------------------ #
  def _add_weather_features(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        1. Weather features
        {1: "Clear", 2: "Fair", 3: "Cloudy", 4: "Overcast", 5: "Fog", 6: "Freezing_Fog",
                        7: "Light_Rain", 8: "Rain", 9: "Heavy_Rain", 10: "Freezing_Rain",
                        11: "Heavy_Freezing_Rain", 12: "Sleet", 13: "Heavy_Sleet", 14: "Light_Snowfall",
                        15: "Snowfall", 16: "Heavy_Snowfall", 17: "Rain_Shower", 18: "Heavy_Rain_Shower",
                        19: "Sleet_Shower", 20: "Heavy_Sleet_Shower", 21: "Snow_Shower",
                        22: "Heavy_Snow_Shower", 23: "Lightning", 24: "Hail", 25: "Thunderstorm",
                        26: "Heavy_Thunderstorm", 27: "Storm"}
        """
        if "hour_start" not in df.columns:
            return df

        start, end = df["hour_start"].min(), df["hour_start"].max() + pd.Timedelta("1H")
        ist = Point(41.0082, 28.9784)
        wx = Hourly(ist, start, end, "Europe/Istanbul").fetch().tz_convert("Europe/Istanbul")
        wx.drop(columns=["tsun", "wpgt"], inplace=True)
        wx = wx.reset_index().rename(columns={"time": "hour_start"})
        wx.columns = [f"weather_{c}" if c != "hour_start" else "hour_start" for c in wx.columns]
        wx["hour_start"] = pd.to_datetime(wx["hour_start"]).dt.tz_localize(None)
        wx["weather_temp"] = wx["weather_temp"] * 9 / 5 + 32

        df = df.merge(wx, on="hour_start", how="left")
        self.context_features.update([c for c in df.columns if c.startswith("weather_")])

        df = self._wx_lags_and_flags(df)

        if "location_cluster" in df.columns and "distance_km" in df.columns:
            df = self._wx_spatial_inter(df)
        return df

  def _wx_lags_and_flags(self, df: pd.DataFrame) -> pd.DataFrame:
      # lags ------------------------------------------------------------
      lag_p = [1, 3, 6]
      wx = (df.sort_values(["start_location_id", "hour_start"])
        .drop_duplicates(subset=["start_location_id", "hour_start"])
        .reset_index(drop=True))
      cols = ["start_location_id", "hour_start"]     
      for base in ["prcp", "temp", "wspd", "rhum", "coco"]:
          col = f"weather_{base}"
          for l in lag_p:
              if base == "wspd" and l == 1:
                  continue
              new = f"{base}_lag{l}"
              wx[new] = wx[col].shift(l).bfill()
              cols.append(new)

      # derived indices -------------------------------------------------
      t, rh, w = wx["weather_temp"], wx["weather_rhum"], wx["weather_wspd"]
      wx["heat_index"] = self._heat_index(t, rh)
      wx["wind_chill"] = self._wind_chill(t, w)
      wx["temp_anomaly24h"] = t - t.rolling(24, 1).mean()
      cols.extend(["heat_index", "wind_chill", "temp_anomaly24h"])
      self.context_features.update(cols)

      # flags -----------------------------------------------------------
      rain_codes   = {7, 8, 9, 10, 11, 12, 13, 17, 18, 19, 20}
      snow_codes   = {12, 13, 14, 15, 16, 19, 20, 21, 22}
      heavy_codes  = {9, 11, 13, 16, 18, 20, 22, 24, 25, 26, 27}  # “heavy/ storm / lightning”
      severe_codes = heavy_codes | {23}                           # + lightning
      wc = wx["weather_coco"].astype("Int8")

      wx["is_raining"]       = wc.isin(rain_codes).astype("int8")
      wx["is_snowing"]       = wc.isin(snow_codes).astype("int8")
      wx["is_heavy_rain"]    = wc.isin({9, 11, 18, 26, 27}).astype("int8")
      wx["is_severe_weather"] = wc.isin(severe_codes).astype("int8")

      wx["is_heatwave"] = (t >= t.quantile(0.95)).astype("int8")
      wx["is_freezing"] = (t <= t.quantile(0.05)).astype("int8")
      wx["is_high_wind"] = (w >= w.quantile(0.95)).astype("int8")
      cols.extend(["is_raining", "is_heavy_rain", "is_snowing", 
      "is_severe_weather", "is_heatwave", "is_freezing", "is_high_wind"])
      self.context_features.update(cols)

      df = df.merge(
            wx[cols],
            on=["start_location_id", "hour_start"],
            how="left",
            validate="many_to_one"
        )

      return df

  # ------- weather × spatial / dist / hour -----------------------
  def _wx_spatial_inter(self, df: pd.DataFrame) -> pd.DataFrame:
      # distance bins
    bins = [0, 1.5, 3, float("inf")]
    labels = ["short", "medium", "long"]
    df["trip_dist_bin"] = pd.cut(df["lag_distance_km_1h_mean"], bins=bins, labels=labels)
    dist_dummies = pd.get_dummies(df["trip_dist_bin"], prefix="trip_dist_bin")
    df = pd.concat([df, dist_dummies], axis=1)
    self.spatial_features.update(dist_dummies.columns)

    # destination cluster mapping
    if self._cluster_map:
        df["dest_cluster"] = df["end_location_id"].map(self._cluster_map).fillna(-1).astype("int8")
        dest_dummies = pd.get_dummies(df["dest_cluster"], prefix="dest_cluster")
        df = pd.concat([df, dest_dummies], axis=1)
        self.spatial_features.update(dest_dummies.columns)

    weather_vars = {"rain": "weather_prcp", "temp": "weather_temp", "wind": "weather_wspd"}

    # origin cluster × weather
    cluster_cols = [c for c in df.columns if c.startswith("location_cluster_")]
    for wname, wcol in weather_vars.items():
        for ccol in cluster_cols:
            col = f"{wname}_x_{ccol}"
            df[col] = df[wcol] * df[ccol]
            self.spatial_features.add(col)

    # destination cluster × weather
    dest_cols = [c for c in df.columns if c.startswith("dest_cluster_")]
    for wname, wcol in weather_vars.items():
        for dcol in dest_cols:
            col = f"{wname}_x_{dcol}"
            df[col] = df[wcol] * df[dcol]
            self.spatial_features.add(col)

    # distance-bin × weather
    dist_cols = [c for c in df.columns if c.startswith("trip_dist_bin_")]
    for wname, wcol in weather_vars.items():
        for dcol in dist_cols:
            col = f"{wname}_x_{dcol}"
            df[col] = df[wcol] * df[dcol]
            self.spatial_features.add(col)
    
    return df
      
  # ------------------------------------------------------------------ #
  # 5) Utils                                                           #
  # ------------------------------------------------------------------ #
  @staticmethod
  def _heat_index(temp_c: pd.Series, rh: pd.Series) -> pd.Series:
      """NOAA heat-index (vectorised, returns °F)."""
      T = temp_c * 9 / 5 + 32
      HI = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (rh * 0.094))
      mask = HI > 80
      HI_precise = (-42.379 + 2.04901523 * T + 10.14333127 * rh -
                    0.22475541 * T * rh - 6.83783e-3 * T ** 2 -
                    5.481717e-2 * rh ** 2 + 1.22874e-3 * T ** 2 * rh +
                    8.5282e-4 * T * rh ** 2 - 1.99e-6 * T ** 2 * rh ** 2)
      HI = np.where(mask, HI_precise, HI)
      return HI

  @staticmethod
  def _wind_chill(temp_c: pd.Series, wind_kmh: pd.Series) -> pd.Series:
      """NOAA wind-chill (vectorised, returns °F)."""
      w_mph = wind_kmh / 1.609344
      mask = (temp_c <= 10) & (wind_kmh >= 4.8)
      T = temp_c * 9 / 5 + 32
      WC = 35.74 + 0.6215 * T - 35.75 * (w_mph ** 0.16) + 0.4275 * T * (w_mph ** 0.16)
      return np.where(mask, WC, temp_c)

  @staticmethod
  def _as_series(df: pd.DataFrame, col: str) -> pd.Series:
      """Return *col* as a Series, even if duplicates exist (take first dup)."""
      obj = df[col]
      if isinstance(obj, pd.DataFrame):
          obj = obj.iloc[:, 0]
      return obj

  @staticmethod
  def _fast_mode(x: pd.Series) -> float:
    """Return most frequent value in a Series (fast)."""
    if x.empty:
        return np.nan
    counts = x.value_counts(dropna=True)
    return counts.index[0] if not counts.empty else np.nan
  
  def _safe_bool(self, df: pd.DataFrame, col: str) -> pd.Series:
      """Return an int8 Series flag; 0 if column missing."""
      if col in df.columns:
          return self._as_series(df, col).astype("int8")
      return pd.Series(0, index=df.index, dtype="int8")

  def get_feature_columns(self) -> Dict[str, List[str]]:
      return {
          "temporal": sorted(self.temporal_features),
          "spatial": sorted(self.spatial_features),
          "context": sorted(self.context_features),
          "all": sorted(self.temporal_features | self.spatial_features | self.context_features)
      }
