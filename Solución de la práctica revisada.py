import yfinance as yf
import pandas as pd
import requests
from datetime import date, datetime
from typing import Optional, Literal, List
import time
import re

Interval = Literal["1h","1d","5d","1wk","1mo"]

class DataExtractorAPIs:
    def __init__(self, alpha_api_key: Optional[str] = None):
        self.alpha_api_key = alpha_api_key

    # --- Yahoo por TICKER (str) ---
    def get_yf_data(
        self,
        ticker: str,
        start: Optional[date|datetime|str] = None,
        end: Optional[date|datetime|str] = None,
        interval: Interval = "1d"
    ) -> pd.DataFrame:
        df = yf.download(
            tickers=ticker,
            start=start or None,
            end=end or None,
            interval=interval,
            group_by=None,        # columnas planas
            auto_adjust=False,
            progress=False
        )
        return df

    # --- AlphaVantage por TICKER (str) ---
    def get_alphavantage_data(
        self,
        ticker: str,
        interval: Interval = "1d"
    ) -> pd.DataFrame:
        if not self.alpha_api_key:
            raise ValueError("Falta alpha_api_key para AlphaVantage.")

        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",  # esta demo usa diario
            "symbol": ticker,
            "apikey": self.alpha_api_key
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()

        key = "Time Series (Daily)"
        if key not in payload:
            raise ValueError(f"AlphaVantage no devolvió '{key}' para {ticker}: {payload}")

        df = pd.DataFrame.from_dict(payload[key], orient="index")
        # Normaliza nombres crudos
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index = pd.to_datetime(df.index, utc=True)
        # Convierte strings a numéricos
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.sort_index()

    @staticmethod
    def to_float(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce")

    def normalize_ohlcv(self, df_raw: pd.DataFrame, symbol: str, source: Literal["yahoo","alphavantage"]) -> pd.DataFrame:
        if df_raw is None or df_raw.empty:
            return pd.DataFrame(columns=["symbol","open","high","low","close","adj_close","volume"])

        df = df_raw.copy()

        # 1) Aplanar MultiIndex si existe
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join([str(x) for x in col]) for col in df.columns]

        # 2) Normalizar nombres
        df.columns = [str(c).lower().replace(" ", "_").strip() for c in df.columns]

        # 3) Eliminar prefijos del ticker (como aapl_open, msft_close, etc.)
        df.columns = [re.sub(r"^[a-z0-9]+_", "", c) for c in df.columns]

        candidates = {
            "open": ["open","o","opn", "open", "aapl_open"],
            "high": ["high","h", "High"],
            "low":  ["low","l", "Low"],
            "close":["close","c","cls", "Close"],
            "adj_close":["adj_close","adjclose","adj_close_","adj_close_price", "Adj_close"],
            "volume":["volume","v","vol", "Volume"]
        }

        def pick(colnames, options):
            for opt in options:
                if opt in colnames:
                  return opt     
            return None

        col_open  = pick(df.columns, candidates["open"])
        col_high  = pick(df.columns, candidates["high"])
        col_low   = pick(df.columns, candidates["low"])
        col_close = pick(df.columns, candidates["close"])
        col_adj   = pick(df.columns, candidates["adj_close"])
        col_vol   = pick(df.columns, candidates["volume"])

        # 3) Si no hay adj_close pero sí close, crearla igualando
        if col_adj is None and col_close is not None:
            df["adj_close"] = df[col_close]
            col_adj = "adj_close"

        # 4) Asegurar índice UTC
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
        else:
            df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")

        out = pd.DataFrame(index=df.index)
        if col_open:  out["open"]  = pd.to_numeric(df[col_open],  errors="coerce")
        if col_high:  out["high"]  = pd.to_numeric(df[col_high],  errors="coerce")
        if col_low:   out["low"]   = pd.to_numeric(df[col_low],   errors="coerce")
        if col_close: out["close"] = pd.to_numeric(df[col_close], errors="coerce")
        if col_adj:   out["adj_close"] = pd.to_numeric(df[col_adj], errors="coerce")
        if col_vol:
            out["volume"] = pd.to_numeric(df[col_vol], errors="coerce").astype("Int64")

        # 5) Rellenar las que falten para no romper el reindex
        required = ["open","high","low","close","adj_close","volume"]
        for c in required:
            if c not in out.columns:
                out[c] = pd.NA if c == "volume" else pd.NA

        out["symbol"] = symbol
        out = out.sort_index().dropna(how="all")
        out = out[["symbol","open","high","low","close","adj_close","volume"]]
        out = out[~out.index.duplicated(keep="last")]
        return out

    def multiple_tickers(
        self,
        tickers: List[str],
        source: Literal["yahoo","alphavantage"] = "yahoo",
        start: Optional[str | date | datetime] = None,
        end: Optional[str | date | datetime] = None,
        interval: Interval = "1d",
        pause_s: float = 0.7,
        out_format: Literal["long","wide"] = "long"
    ) -> pd.DataFrame:

        all_norm: List[pd.DataFrame] = []

        for ticker in tickers:
            print(f"Descargando {ticker} desde {source.upper()}...")
            if source == "yahoo":
                df_raw = self.get_yf_data(ticker, start, end, interval)
            elif source == "alphavantage":
                df_raw = self.get_alphavantage_data(ticker, interval)
                time.sleep(12)  # rate limit AV
            else:
                raise ValueError("Fuente no soportada")

            df_norm = self.normalize_ohlcv(df_raw, symbol=ticker, source=source)
            all_norm.append(df_norm)
            time.sleep(pause_s)

        if not all_norm:
            return pd.DataFrame(columns=["symbol","open","high","low","close","adj_close","volume"])

        df_long = pd.concat(all_norm, axis=0).sort_index()

        if out_format == "long":
            return df_long

        # wide
        fields = ["open","high","low","close","adj_close","volume"]
        wides = []
        for f in fields:
            sub = df_long.pivot_table(index=df_long.index, columns="symbol", values=f, aggfunc="last")
            sub.columns = pd.MultiIndex.from_product([[f], sub.columns], names=["field","symbol"])
            wides.append(sub)
        df_wide = pd.concat(wides, axis=1).sort_index()
        df_wide = df_wide.reindex(columns=pd.MultiIndex.from_product([fields, sorted(tickers)]))
        return df_wide
    

    #Como llamar a todo: 


    # 1) Yahoo en formato LONG
ext = DataExtractorAPIs()
df_long = ext.multiple_tickers(["AAPL","MSFT"], source="yahoo", start="2024-01-01", end="2024-12-31", out_format="long")
print(df_long.head())

# 2) Yahoo en formato WIDE (columnas multi-índice)
df_wide = ext.multiple_tickers(["AAPL","MSFT"], source="yahoo", start="2024-01-01", end="2024-12-31", out_format="wide")
print(df_wide.head())

# 3) AlphaVantage (requiere API key) y normalizado igual
ext_av = DataExtractorAPIs(alpha_api_key="EFRIRH9WY1QSDM9U")
df_av = ext_av.multiple_tickers(["AAPL"], source="alphavantage", out_format="long")
print(df_av.head())