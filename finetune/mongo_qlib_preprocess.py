#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mongo-backed data preprocessor that mirrors the Qlib-based one.

It reads kline documents from MongoDB collections (schema like the user's example),
transforms them into per-symbol pandas DataFrames with a datetime index and a unified
feature set, splits by time ranges, and saves train/val/test dicts as .pkl files
compatible with the Qlib-style downstream loader the user already has.

Assumptions / conventions
-------------------------
- Collections are named like "{SYMBOL}_{INTERVAL}_Binance" (configurable).
- Time fields in Mongo are millisecond timestamps (e.g., endtime in ms).
- OHLCV numeric fields may be strings in Mongo; we'll coerce to float.
- Output format exactly matches qlib_data_preprocess.py: each .pkl contains
  a dict {symbol: DataFrame[features]} with DateTimeIndex.

Usage
-----
python mongo_qlib_preprocess.py

This script expects a project-local `config.py` defining a `Config` class with fields:
- dataset_path: str (where to save pkl files)
- feature_list: List[str] (e.g., ['open','high','low','close','vol','amt','vwap'])
- lookback_window: int
- predict_window: int
- train_time_range: Tuple[pd.Timestamp|str, pd.Timestamp|str]
- val_time_range: Tuple[pd.Timestamp|str, pd.Timestamp|str]
- test_time_range: Tuple[pd.Timestamp|str, pd.Timestamp|str]
- Optionally:
  - symbols: List[str] (if omitted, discover from Mongo by collection names)
  - interval: str (e.g. '4h')
  - mongo_uri: str  (env var MONGODB_URI also supported)
  - mongo_db:  str  (e.g. 'market_info')
  - coll_name_tpl: str (default '{symbol}_{interval}_Binance')
  - time_field: str (default 'endtime', used for sorting & slicing)

Dependencies
------------
pip install pymongo pandas numpy python-dotenv

Notes
-----
- Imports `MongoShell` and `MongoKlineSource` from your wrappers if available;
  otherwise provides minimal fallbacks.
"""

from __future__ import annotations
import os
import sys
sys.path.append("../")

import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Project config
from config import Config  # must exist in the project

# Prefer the user's wrappers; fall back to local minimal implementations if absent.
try:
    from iowrapper.mongo_shell import MongoShell  # user's wrapper
except Exception as e:
    MongoShell = None
    _mongo_import_error = e

try:
    from iowrapper.data_adapters import MongoKlineSource  # user's wrapper
except Exception as e:
    MongoKlineSource = None
    _adapter_import_error = e


# -----------------------------
# Helpers
# -----------------------------
def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


@dataclass
class MongoPreprocessCfg:
    mongo_uri: Optional[str] = None
    mongo_db: Optional[str] = None
    interval: str = "4h"
    coll_name_tpl: str = "{symbol}_{interval}_Binance"
    time_field: str = "endtime"
    symbols: Optional[List[str]] = None  # if None, discover from db


class MongoQlibLikePreprocessor:
    def __init__(self):
        self.config = Config()  # user's central config
        # tie-in Mongo-specific fields (allow overriding via Config if present)
        self.mcfg = MongoPreprocessCfg(
            mongo_uri=getattr(self.config, "mongo_uri", None),
            mongo_db=getattr(self.config, "mongo_db", None),
            interval=getattr(self.config, "interval", "4h"),
            coll_name_tpl="{symbol}_{interval}_Binance",
            time_field=getattr(self.config, "time_field", "endtime"),
            symbols=getattr(self.config, "symbols", None),
        )

        # storage for processed per-symbol frames
        self.data: Dict[str, pd.DataFrame] = {}

        # initialize Mongo client
        if MongoShell is None:
            raise ImportError(
                "Could not import MongoShell wrapper. "
                f"Original error: {_mongo_import_error}. Make sure your wrappers are importable."
            )
        self.mongo = MongoShell(uri=self.mcfg.mongo_uri, db=self.mcfg.mongo_db)

    # -----------------------------
    # Discovery
    # -----------------------------
    def _discover_symbols(self) -> List[str]:
        if self.mcfg.symbols:  # explicit list wins
            return list({s.upper() for s in self.mcfg.symbols})
        if self.mcfg.mongo_db is None:
            raise ValueError("mongo_db must be set in Config when auto-discovering symbols.")
        db = self.mongo.get_db(self.mcfg.mongo_db)
        colls = db.list_collection_names()
        suf = f"_{self.mcfg.interval}_Binance"
        symbols = []
        for name in colls:
            if name.endswith(suf):
                sym = name[: -len(suf)]
                if sym:
                    symbols.append(sym.upper())
        symbols = sorted(list(set(symbols)))
        if not symbols:
            raise ValueError(f"No collections matched '*{suf}' in db '{self.mcfg.mongo_db}'.")
        
        print(f"Discovered {len(symbols)} symbols from MongoDB. tHere are some examples: {symbols[:5]}")
        return symbols

    # -----------------------------
    # Loading
    # -----------------------------
    def load_mongo_data(self):
        """Load & process data per symbol from Mongo into self.data."""
        print("Loading and processing data from Mongo...")

        symbols = self._discover_symbols()
        print(f"Symbols ({len(symbols)}): {', '.join(symbols)}")

        if MongoKlineSource is None:
            raise ImportError(
                "Could not import MongoKlineSource wrapper. "
                f"Original error: {_adapter_import_error}. Make sure your wrappers are importable."
            )

        for sym in symbols:
            coll = self.mcfg.coll_name_tpl.format(symbol=sym, interval=self.mcfg.interval)
            src = MongoKlineSource(
                mongo=self.mongo,
                db=self.mcfg.mongo_db,
                coll=coll,
                symbol=sym,
                interval=self.mcfg.interval,
                start_ms=None,  # pull full range and filter later
                end_ms=None,
                time_field=self.mcfg.time_field,
            )
            try:
                raw = src.fetch()
            except Exception as e:
                print(f"[WARN] Skip {sym}: failed to fetch from '{coll}': {e}")
                continue

            # Basic numeric coercion
            numeric_cols = [
                "open", "high", "low", "close", "volume",
                "quotevolume", "activebuyquotevolume", "activebuyvolume"
            ]
            raw = _coerce_numeric(raw, numeric_cols)

            # Build the time index (from endtime by default) & sort
            if self.mcfg.time_field not in raw.columns:
                print(f"[WARN] Skip {sym}: time field '{self.mcfg.time_field}' missing.")
                continue
            raw = raw.sort_values(self.mcfg.time_field).reset_index(drop=True)

            # build the time index
            tms = pd.to_datetime(
                pd.to_numeric(raw[self.mcfg.time_field], errors="coerce"),
                unit="ms",
                utc=True,
                errors="coerce",
            )
            raw.index = tms.dt.tz_localize(None) # make naive UTC
            raw = raw.loc[~raw.index.isna()]
            raw = raw.drop(columns=[c for c in ["starttime", "eventtime"] if c in raw.columns])

            # align with the qlib preprocessor's expectations
            raw.index.name = "datetime"

            # Compute derived features
            # vwap = quotevolume / volume (robust to zero volume)
            if "quotevolume" in raw.columns and "volume" in raw.columns:
                with np.errstate(divide="ignore", invalid="ignore"):
                    vwap = raw["quotevolume"] / raw["volume"]
                vwap = vwap.replace([np.inf, -np.inf], np.nan)
                raw["vwap"] = vwap

            # Align to the expected feature list from Config
            # Add vol and amt like qlib preprocessor
            raw["vol"] = raw.get("volume")
            ohlc4 = (raw["open"] + raw["high"] + raw["low"] + raw["close"]) / 4.0
            raw["amt"] = ohlc4 * raw["vol"]

            # Final feature selection
            feat_cols = list(self.config.feature_list)
            missing_feats = [c for c in feat_cols if c not in raw.columns]
            if missing_feats:
                # For missing features, create NaNs (better than failing hard)
                for c in missing_feats:
                    raw[c] = np.nan

            df = raw[feat_cols].dropna()

            min_len = int(self.config.lookback_window) + int(self.config.predict_window) + 1
            if len(df) < min_len:
                print(f"[WARN] Skip {sym}: insufficient rows after cleaning (< {min_len}).")
                continue

            self.data[sym] = df

        if not self.data:
            raise RuntimeError("No symbols produced usable data frames. Aborting.")

    # -----------------------------
    # Split & Persist
    # -----------------------------
    def _to_ts(self, v) -> pd.Timestamp:
        ts = pd.Timestamp(v)
        return ts.tz_localize(None) if ts.tzinfo is not None else ts

    def prepare_dataset(self):
        print("Splitting data into train/val/test and writing .pkl files...")
        train_data: Dict[str, pd.DataFrame] = {}
        val_data: Dict[str, pd.DataFrame] = {}
        test_data: Dict[str, pd.DataFrame] = {}

        # Time bounds (naive timestamps to match df index)
        t0, t1 = map(self._to_ts, self.config.train_time_range)
        v0, v1 = map(self._to_ts, self.config.val_time_range)
        s0, s1 = map(self._to_ts, self.config.test_time_range)

        for sym, df in self.data.items():
            # boolean masks
            tm = (df.index >= t0) & (df.index <= t1)
            vm = (df.index >= v0) & (df.index <= v1)
            sm = (df.index >= s0) & (df.index <= s1)

            train_data[sym] = df.loc[tm]
            val_data[sym]   = df.loc[vm]
            test_data[sym]  = df.loc[sm]

        os.makedirs(self.config.dataset_path, exist_ok=True)
        with open(os.path.join(self.config.dataset_path, "train_data.pkl"), "wb") as f:
            pickle.dump(train_data, f)
            # also save as csv for easy inspection
            pd.concat(train_data).to_csv(os.path.join(self.config.dataset_path, "train_data.csv"))
        with open(os.path.join(self.config.dataset_path, "val_data.pkl"), "wb") as f:
            pickle.dump(val_data, f)
            pd.concat(val_data).to_csv(os.path.join(self.config.dataset_path, "val_data.csv"))
        with open(os.path.join(self.config.dataset_path, "test_data.pkl"), "wb") as f:
            pickle.dump(test_data, f)
            pd.concat(test_data).to_csv(os.path.join(self.config.dataset_path, "test_data.csv"))

        print("Done. Files saved under:", self.config.dataset_path)


def main():
    pre = MongoQlibLikePreprocessor()
    pre.load_mongo_data()
    pre.prepare_dataset()


if __name__ == "__main__":
    main()
