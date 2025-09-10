
from __future__ import annotations
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import pandas as pd
from .mongo_shell import MongoShell

@dataclass
class DataSource:
    def fetch(self) -> pd.DataFrame:
        raise NotImplementedError

class MongoKlineSource(DataSource):
    def __init__(self, mongo: MongoShell, db: str, coll: str, symbol: str, interval: str,
                 start_ms: Optional[int]=None, end_ms: Optional[int]=None,
                 time_field: str = "endtime"):
        self.mongo = mongo
        self.db = db
        self.coll = coll
        self.symbol = symbol.upper()
        self.interval = interval
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.time_field = time_field

    def fetch(self) -> pd.DataFrame:
        c = self.mongo.get_collection(self.coll, self.db)
        q: Dict[str, Any] = {"symbol": self.symbol, "interval": self.interval}
        if self.start_ms is not None or self.end_ms is not None:
            q[self.time_field] = {}
            if self.start_ms is not None: q[self.time_field]["$gte"] = int(self.start_ms)
            if self.end_ms is not None: q[self.time_field]["$lte"] = int(self.end_ms)

        cur = c.find(q, projection={"_id":0}).sort(self.time_field, 1) # 1 is ascending
        rows: List[dict] = list(cur)
        if not rows:
            raise ValueError("No kline rows matched the query.")
        df = pd.DataFrame(rows)
        # mapping = {
        #     "endtime": "ts",
        #     "close": "close",
        #     "open": "open",
        #     "high": "high",
        #     "low": "low",
        #     "volume": "volume",
        # }
        # rename_map = {k:v for k,v in mapping.items() if k in df.columns}
        # df = df.rename(columns=rename_map)
        required = ["endtime", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required kline fields: {missing}")
        
        df = df.sort_values(self.time_field).reset_index(drop=True)
        return df
