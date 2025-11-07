import yfinance as yf
import pandas as pd
import requests
from datetime import date, datetime
from typing import Optional, Literal, List
import time
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterable, Tuple
import numpy as np
import matplotlib.pyplot as plt


# Cada serie de datos sebe ser un objeto: 


@dataclass
class PriceBar:
    ts: pd.Timestamp
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    adj_close: Optional[float] = None
    volume: Optional[int] = None

@dataclass
class PriceSeries:
    symbol: str
    bars: List[PriceBar] = field(default_factory=list)

    # ---------- Constructores / conversión ----------
    @classmethod
    def from_dataframe(cls, symbol: str, df: pd.DataFrame) -> "PriceSeries":
        """
        Espera un DataFrame por-símbolo con índice datetime (o columna 'date') y
        columnas: open, high, low, close, adj_close, volume.
        """
        d = df.copy()

        # Asegurar índice temporal
        if "date" in d.columns:
            d["date"] = pd.to_datetime(d["date"], errors="coerce", utc=True)
            d = d.dropna(subset=["date"]).set_index("date")
        if not isinstance(d.index, pd.DatetimeIndex):
            d.index = pd.to_datetime(d.index, errors="coerce", utc=True)
        d = d.sort_index()

        # Garantizar columnas
        cols = ["open","high","low","close","adj_close","volume"]
        for c in cols:
            if c not in d.columns:
                d[c] = pd.NA

        # Coerción segura
        for c in ["open","high","low","close","adj_close"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d["volume"] = pd.to_numeric(d["volume"], errors="coerce").astype("Int64")

        bars = [
            PriceBar(
                ts=ts,
                open=(None if pd.isna(row["open"]) else float(row["open"])),
                high=(None if pd.isna(row["high"]) else float(row["high"])),
                low=(None if pd.isna(row["low"]) else float(row["low"])),
                close=(None if pd.isna(row["close"]) else float(row["close"])),
                adj_close=(None if pd.isna(row["adj_close"]) else float(row["adj_close"])),
                volume=(None if pd.isna(row["volume"]) else int(row["volume"]))
            )
            for ts, row in d[cols].iterrows()
        ]
        return cls(symbol=symbol, bars=bars)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for b in self.bars:
            rows.append(
                {
                    "date": b.ts,
                    "symbol": self.symbol,
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "adj_close": b.adj_close,
                    "volume": b.volume,
                }
            )
        df = pd.DataFrame(rows).set_index("date").sort_index()
        # Tipos
        for c in ["open","high","low","close","adj_close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")
        return df

    # ---------- Utilidades de coherencia/diagnóstico ----------
    def n(self) -> int:
        return len(self.bars)

    def date_range(self) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        if not self.bars:
            return (None, None)
        return (self.bars[0].ts, self.bars[-1].ts)

    def nan_ratio(self, col: str = "adj_close") -> float:
        vals = [getattr(b, col) for b in self.bars]
        total = len(vals)
        if total == 0:
            return 0.0
        n_nan = sum(v is None or (isinstance(v, float) and np.isnan(v)) for v in vals)
        return 100.0 * n_nan / total

    def price_min_max(self, col: str = "adj_close") -> tuple[Optional[float], Optional[float]]:
        vals = [getattr(b, col) for b in self.bars if getattr(b, col) is not None]
        if not vals:
            return (None, None)
        return (float(np.nanmin(vals)), float(np.nanmax(vals)))

    def gaps_over(self, days: int = 5) -> List[tuple[pd.Timestamp, pd.Timestamp, pd.Timedelta]]:
        """Devuelve lista de gaps > days entre barras consecutivas."""
        res = []
        if len(self.bars) < 2:
            return res
        for prev, nxt in zip(self.bars, self.bars[1:]):
            delta = nxt.ts - prev.ts
            if delta > pd.Timedelta(days=days):
                res.append((prev.ts, nxt.ts, delta))
        return res


def make_price_series_by_symbol(df_long: pd.DataFrame) -> Dict[str, PriceSeries]:
  
    """
    Crea objetos PriceSeries por símbolo desde tu df_long (formato que devuelve multiple_tickers(out_format='long')).
    """
    if df_long is None or df_long.empty:
        return {}

    if "symbol" not in df_long.columns:
        raise ValueError("df_long debe contener la columna 'symbol'.")

    series: Dict[str, PriceSeries] = {}
    for sym, g in df_long.groupby("symbol"):
        series[sym] = PriceSeries.from_dataframe(sym, g)
    return series



# Portfolio con: MONTE CARLO + REPORT Y GRÁFICOS 



@dataclass
class Portfolio:
    series_by_symbol: Dict[str, PriceSeries]
    weights: Dict[str, float]  # pesos por símbolo (pueden no estar normalizados)

    def __post_init__(self):
        # Filtrar pesos a símbolos existentes y normalizarlos
        w = {k: float(v) for k, v in self.weights.items() if k in self.series_by_symbol}
        if not w:
            raise ValueError("Los pesos no contienen ninguno de los símbolos de la cartera.")
        s = sum(abs(v) for v in w.values())
        self.weights = {k: v / s for k, v in w.items()}  # normaliza por suma de absolutos

    # --- utilidades internas ---
    def _prices_wide(self) -> pd.DataFrame:
        """
        Devuelve un DataFrame wide con adj_close por símbolo.
        Índice: datetime; Columnas: símbolos.
        """
        frames = []
        for sym, ps in self.series_by_symbol.items():
            df = ps.to_dataframe()[["adj_close"]].rename(columns={"adj_close": sym})
            frames.append(df)
        wide = pd.concat(frames, axis=1).sort_index()
        return wide

    def _returns_wide(
        self,
        log_returns: bool = True,
        dropna: bool = True,
    ) -> pd.DataFrame:
        """
        Rendimientos diarios por símbolo, alineados.
        """
        px = self._prices_wide()
        # asegurar numérico
        px = px.apply(pd.to_numeric, errors="coerce")
        # calcular rendimientos
        if log_returns:
            ret = np.log(px).diff()
        else:
            ret = px.pct_change()
        if dropna:
            ret = ret.dropna(how="all")
        return ret

    # --- MONTE CARLO ---
    def simulate_monte_carlo(
        self,
        *,
        horizon_days: int = 252,
        n_sims: int = 1000,
        log_returns: bool = True,
        with_correlation: bool = True,
        seed: Optional[int] = None,
        start_prices: Optional[Dict[str, float]] = None,
    ) -> Dict[str, pd.DataFrame]:

        
        rng = np.random.default_rng(seed)

        # 1) Históricos y parámetros
        ret = self._returns_wide(log_returns=log_returns, dropna=True)
        if ret.empty:
            raise ValueError("No hay rendimientos suficientes para estimar parámetros.")

        mu = ret.mean().values  # vector media
        cov = ret.cov().values  # matriz covarianza
        syms = ret.columns.tolist()
        n_assets = len(syms)

        # precios iniciales
        px0 = {}
        if start_prices:
            for s in syms:
                if s in start_prices and start_prices[s] is not None:
                    px0[s] = float(start_prices[s])
        # completa con último adj_close disponible si falta alguno
        if len(px0) < n_assets:
            last_prices = self._prices_wide().ffill().iloc[-1]
            for s in syms:
                if s not in px0 or px0[s] is None:
                    px0[s] = float(last_prices[s])

        # 2) Choques aleatorios (correlacionados o independientes)
        T = horizon_days
        if with_correlation:
            # Cholesky puede fallar si cov no es PD; ajustar con jitter si es necesario
            try:
                L = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                # pequeño jitter diagonal
                eps = 1e-10
                L = np.linalg.cholesky(cov + eps * np.eye(n_assets))
        # 3) Simulación GBM por pasos (discretización lognormal)
        #   dS/S = mu*dt + sigma*sqrt(dt)*Z
        dt = 1.0 / 252.0  # diario aprox.
        # matrices de salida
        assets_paths: Dict[str, pd.DataFrame] = {}
        # inicializar arrays
        # shape: (T+1, n_sims, n_assets)
        paths = np.zeros((T + 1, n_sims, n_assets), dtype=float)
        # set inicial
        for j, s in enumerate(syms):
            paths[0, :, j] = px0[s]

        # precomputos
        vol = np.sqrt(np.diag(cov))  # sigmas
        drift = mu - 0.5 * (vol ** 2) if log_returns else mu  # drift GBM con log-retornos

        for t in range(1, T + 1):
            # Z ~ N(0, I)
            z = rng.standard_normal(size=(n_sims, n_assets))
            if with_correlation:
                z = z @ L.T  # correlacionar
            # paso multiplicativo
            if log_returns:
                step = np.exp(drift * dt + np.sqrt(dt) * z)
            else:
                step = 1.0 + (drift * dt + np.sqrt(dt) * z)
            paths[t] = paths[t - 1] * step

        # 4) Construir DataFrames por activo
        idx_steps = pd.RangeIndex(start=0, stop=T + 1, step=1, name="step")
        for j, s in enumerate(syms):
            df_s = pd.DataFrame(paths[:, :, j], index=idx_steps)
            assets_paths[s] = df_s

        # 5) Cartera agregada: valor = suma(weights * precios normalizados por px0) * 100
        #    (escala a 100 en t=0 para visualizar)
        w = np.array([self.weights.get(s, 0.0) for s in syms], dtype=float)
        norm_prices = paths / np.array([px0[s] for s in syms])  # normaliza a 1 en t=0
        portfolio_paths = (norm_prices * w.reshape(1, 1, -1)).sum(axis=2) * 100.0
        df_port = pd.DataFrame(portfolio_paths, index=idx_steps)

        return {
            "assets": assets_paths,
            "portfolio": df_port,
            "symbols": syms,
            "weights": w,
            "px0": px0,
            "params": {"mu": mu, "cov": cov, "log_returns": log_returns, "with_correlation": with_correlation, "dt": dt},
            "horizon_days": T,
            "n_sims": n_sims,
        }

    def plot_results(
        self,
        sim_result: Dict[str, pd.DataFrame],
        *,
        show_assets: bool = False,
        quantiles: List[float] = [0.05, 0.5, 0.95],
        title: Optional[str] = None,
    ) -> None:
        """
        Muestra visualmente:
        - Fan chart de la cartera (percentiles).
        - (opcional) 1-2 activos como ejemplo.
        """
        df_port = sim_result["portfolio"]
        qs = df_port.quantile(quantiles, axis=1).T  # columnas: quantiles
        plt.figure(figsize=(9, 5))
        for q in sorted(quantiles):
            plt.plot(qs.index, qs[q], label=f"q={q:.2f}")
        plt.title(title or "Simulación Monte Carlo — Cartera")
        plt.xlabel("Paso (día)")
        plt.ylabel("Valor cartera (base=100)")
        plt.legend()
        plt.grid(True)
        plt.show()

        if show_assets:
            assets: Dict[str, pd.DataFrame] = sim_result["assets"]
            syms: List[str] = sim_result["symbols"]
            # mostrar todos los activos
            for sym in syms:
                df = assets[sym]
                qs_a = df.quantile(quantiles, axis=1).T
                plt.figure(figsize=(9, 4))
                for q in sorted(quantiles):
                    plt.plot(qs_a.index, qs_a[q], label=f"q={q:.2f}")
                plt.title(f"Simulación — {sym}")
                plt.xlabel("Paso (día)")
                plt.ylabel(f"Precio simulado ({sym})")
                plt.legend()
                plt.grid(True)
                plt.show()

    def summary(self, sim_result: Dict[str, pd.DataFrame]) -> None:
        """
        Imprime estadísticas de la distribución final (t = horizon):
        para la cartera y para cada activo.
        """
        print("\n📈 Resumen Monte Carlo (valor final):\n")
        dfp = sim_result["portfolio"]
        last = dfp.iloc[-1, :]
        print("Cartera (base=100):")
        print(f"   media={last.mean():.2f}  std={last.std(ddof=1):.2f}  min={last.min():.2f}  mediana={last.median():.2f}  max={last.max():.2f}")

        for sym, df in sim_result["assets"].items():
            last_a = df.iloc[-1, :]
            print(f"{sym}:   media={last_a.mean():.2f}  std={last_a.std(ddof=1):.2f}  min={last_a.min():.2f}  mediana={last_a.median():.2f}  max={last_a.max():.2f}")



    # Informe formateado en markdown 


    def report(
        self,
        sim_result: Dict[str, pd.DataFrame],
        *,
        alpha: float = 0.05,
        include_assets: bool = True,
        annualize: bool = True,
        show: bool = True,
        title: str = "## Informe de Cartera (Monte Carlo)"
    ) -> str:
        """
        Devuelve un string en Markdown con el resumen de la simulación y avisos. Incluye:
        - alpha: nivel para VaR/CVaR (ej. 0.05)
        - include_assets: añade stats por activo
        - annualize: anualiza μ y σ de rendimientos históricos usados
        - show: si True, hace print del Markdown
        """
        # Parámetros y básicos
        dfp = sim_result["portfolio"]          # trayectorias cartera (base 100 en t=0)
        last = dfp.iloc[-1, :]                # distribución final
        params = sim_result["params"]
        syms = sim_result["symbols"]
        w = sim_result["weights"]
        px0 = sim_result["px0"]
        T = sim_result["horizon_days"]
        n_sims = sim_result["n_sims"]

        # μ y Σ de históricos
        mu = params["mu"]
        cov = params["cov"]
        log_returns = params["log_returns"]
        with_corr = params["with_correlation"]

        # métricas finales (cartera)
        mean_f = float(last.mean())
        std_f  = float(last.std(ddof=1))
        min_f  = float(last.min())
        med_f  = float(last.median())
        max_f  = float(last.max())

        # Pérdida (respecto a 100): VaR/CVaR
        pnl = last - 100.0
        var = float(np.percentile(pnl, alpha*100))  # puede ser negativo
        cvar = float(pnl[pnl <= var].mean()) if (pnl <= var).any() else float(var)

        # μ y σ por activo (históricos)
        mu_ann = (mu * 252.0) if annualize else mu
        vol = np.sqrt(np.diag(cov))
        vol_ann = (vol * np.sqrt(252.0)) if annualize else vol

        # Warnings y notas
        notes = []
        if not with_corr:
            notes.append("- ⚠️ La simulación ignora correlaciones (with_correlation=False).")
        if np.isnan(mu).any() or np.isnan(cov).any():
            notes.append("- ⚠️ Hay NaNs en μ o Σ estimados; revisa los datos históricos.")
        if (np.array(list(px0.values())) <= 0).any():
            notes.append("- ⚠️ Algún precio inicial ≤ 0; revisa precios de partida.")
        if T < 60:
            notes.append("- ℹ️ Horizonte corto: podrías aumentar `horizon_days` para escenarios más largos.")
        if n_sims < 500:
            notes.append("- ℹ️ Pocas simulaciones: aumenta `n_sims` para estabilidad en percentiles.")

        # Construir Markdown
        lines = []
        lines.append(title)
        lines.append("")
        lines.append(f"- **Horizonte:** {T} días | **Simulaciones:** {n_sims} | **Log-returns:** `{log_returns}` | **Correlaciones:** `{with_corr}`")
        lines.append(f"- **Activos:** {', '.join(syms)}")
        lines.append("- **Pesos normalizados:** " + ", ".join(f"{s}={w_i:.2f}" for s, w_i in zip(syms, w)))
        lines.append("- **Precios iniciales:** " + ", ".join(f"{s}={px0[s]:.2f}" for s in syms))
        lines.append("")
        lines.append("### Resultados — Cartera (valor base=100)")
        lines.append(f"- **Media:** {mean_f:.2f}  | **Desv. típica:** {std_f:.2f}  | **Mediana:** {med_f:.2f}")
        lines.append(f"- **Mín:** {min_f:.2f}  | **Máx:** {max_f:.2f}")
        lines.append(f"- **VaR {int(alpha*100)}%:** {var:.2f}  | **CVaR {int(alpha*100)}%:** {cvar:.2f}  (respecto a 100)")
        lines.append("")
        lines.append("### Parámetros históricos por activo" + (" (anualizados)" if annualize else ""))
        for s, m, v in zip(syms, mu_ann, vol_ann):
            lines.append(f"- **{s}:** μ={m:.4f}  σ={v:.4f}")
        if include_assets:
            lines.append("")
            lines.append("### Distribución final por activo (valor simulado)")
            for s, df in sim_result["assets"].items():
                la = df.iloc[-1, :]
                lines.append(f"- **{s}:** media={la.mean():.2f}  std={la.std(ddof=1):.2f}  min={la.min():.2f}  mediana={la.median():.2f}  max={la.max():.2f}")
        if notes:
            lines.append("")
            lines.append("### Notas / Advertencias")
            lines.extend(notes)

        md = "\n".join(lines)
        if show:
            print(md)
        return md


    # Reporte de gráficos

    def plots_report(
        self,
        sim_result: Dict[str, pd.DataFrame],
        *,
        show_assets: bool = True,
        asset_examples: int = 2,
        save: bool = True,
        outdir: str = "plots_report",
        quantiles: List[float] = [0.05, 0.5, 0.95],
        title_portfolio: str = "Monte Carlo — Cartera (fan chart)"
    ) -> List[str]:
        """
        Genera y muestra visualizaciones útiles y (opcional) las guarda en disco.
        Devuelve la lista de rutas guardadas.
        Figuras:
        1) Fan chart (percentiles) de la cartera.
        2) Histograma del valor final de la cartera.
        3) Heatmap de correlaciones históricas.
        4) Precios históricos normalizados (base 100).
        """
        import os  # import local para no tocar la cabecera
        saved = []
        os.makedirs(outdir, exist_ok=True)

        syms = sim_result["symbols"]
        dfp = sim_result["portfolio"]
        params = sim_result["params"]
        cov = params["cov"]
        # Corr a partir de cov
        stds = np.sqrt(np.diag(cov))
        denom = np.outer(stds, stds)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = cov / denom
        corr = np.clip(corr, -1, 1)
        
        # Histograma valor final cartera
        last = dfp.iloc[-1, :]
        plt.figure(figsize=(8, 4.5))
        plt.hist(last.values, bins=40)
        plt.title("Distribución valor final — Cartera")
        plt.xlabel("Valor final (base=100)"); plt.ylabel("Frecuencia")
        plt.grid(True); plt.tight_layout()
        if save:
            fn = os.path.join(outdir, "02_portfolio_final_hist.png")
            plt.savefig(fn, dpi=140); saved.append(fn)
        plt.show()

        # Heatmap correlaciones históricas
        plt.figure(figsize=(6.5, 5.5))
        im = plt.imshow(corr, vmin=-1, vmax=1)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title("Correlaciones históricas (adj_close)")
        plt.xticks(ticks=range(len(syms)), labels=syms, rotation=45, ha="right")
        plt.yticks(ticks=range(len(syms)), labels=syms)
        plt.tight_layout()
        if save:
            fn = os.path.join(outdir, "03_corr_heatmap.png")
            plt.savefig(fn, dpi=140); saved.append(fn)
        plt.show()

        # Precios históricos normalizados (base=100)
        px = self._prices_wide().ffill()
        base = px.iloc[0]
        px_norm = (px / base) * 100.0
        plt.figure(figsize=(9, 5))
        for s in syms:
            plt.plot(px_norm.index, px_norm[s], label=s)
        plt.title("Histórico — Precios normalizados (base=100)")
        plt.xlabel("Fecha"); plt.ylabel("Índice (base=100)")
        plt.grid(True); plt.legend(); plt.tight_layout()
        if save:
            fn = os.path.join(outdir, "04_prices_normalized.png")
            plt.savefig(fn, dpi=140); saved.append(fn)
        plt.show()

        

# Inicio de la extracción de datos con módulos de normalización, validación, limpieza y obtención de datos estadísticos.  


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

    # --- CoinGecko (criptomonedas) ---
    def get_coingecko_data(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        days: str | int = "max"
    ) -> pd.DataFrame:
        """
        Descarga datos de una criptomoneda desde CoinGecko
        """
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {"vs_currency": vs_currency, "days": days}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()

        prices = payload.get("prices", [])
        volumes = payload.get("total_volumes", [])

        if not prices:
            return pd.DataFrame(columns=["symbol","open","high","low","close","adj_close","volume"])

        dfp = pd.DataFrame(prices, columns=["ts_ms", "price"])
        dfv = pd.DataFrame(volumes, columns=["ts_ms", "vol"])
        df = dfp.merge(dfv, on="ts_ms", how="left")

        df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        df = df.drop(columns=["ts_ms"]).set_index("ts").sort_index()

        # Re-muestreo diario OHLC + volumen
        o = df["price"].resample("D").first()
        h = df["price"].resample("D").max()
        l = df["price"].resample("D").min()
        c = df["price"].resample("D").last()
        v = df["vol"].resample("D").sum(min_count=1)

        out = pd.DataFrame({
            "open": o, "high": h, "low": l, "close": c,
            "adj_close": c, "volume": v
        })
        out["symbol"] = coin_id.upper()
        out = out.dropna(how="all")
        return out


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


# Validación: datos de precios


        print("  ")
        print("="*50)
        print("📈 VALIDACIÓN - DATOS DE PRECIOS")
        print("="*50)

        precios_raw = {t: df for t, df in zip(tickers, all_norm)}

        for ticker, df_raw in precios_raw.items():
            print(f"\n{ticker}:")
            print(f"   Registros: {len(df_raw)}")

            # --- Asegurar columna 'date' en datetime (UTC) de forma robusta ---
            df_tmp = df_raw.copy()

            if "date" in df_tmp.columns:
                # Si existe, garantizamos tipo datetime
                df_tmp["date"] = pd.to_datetime(df_tmp["date"], errors="coerce", utc=True)
            else:
                # No existe: intentar sacarla del índice
                if isinstance(df_tmp.index, pd.DatetimeIndex):
                    df_tmp = df_tmp.reset_index()
                    # La columna creada tendrá el nombre del índice (o 'index' si no tiene)
                    # Detectamos qué columna es datetime
                    dt_cols = [c for c in df_tmp.columns if pd.api.types.is_datetime64_any_dtype(df_tmp[c])]
                    if dt_cols:
                        main_date_col = dt_cols[0]
                    else:
                        # Si aún no es datetime, convertimos la primera columna
                        main_date_col = df_tmp.columns[0]
                        df_tmp[main_date_col] = pd.to_datetime(df_tmp[main_date_col], errors="coerce", utc=True)
                    if main_date_col != "date":
                        df_tmp = df_tmp.rename(columns={main_date_col: "date"})
                else:
                    # Índice no es datetime: creamos 'date' convirtiendo el índice
                    idx_series = pd.Series(df_tmp.index)
                    df_tmp["date"] = pd.to_datetime(idx_series, errors="coerce", utc=True)

            # Drop de filas sin fecha válida
            df_tmp = df_tmp.dropna(subset=["date"]).sort_values("date")

            # --- Métricas ---
            rango_min = df_tmp["date"].min()
            rango_max = df_tmp["date"].max()
            print(f"   Rango: {rango_min.date()} → {rango_max.date()}")
            print(f"   Días: {(rango_max - rango_min).days}")

            nans = df_tmp["adj_close"].isna().sum()
            total = len(df_tmp)
            pct = (nans/total*100) if total else 0.0
            print(f"   NaNs: {nans} ({pct:.2f}%)")

            # Precios min/max (con coerción segura)
            ac = pd.to_numeric(df_tmp["adj_close"], errors="coerce")
            if ac.notna().any():
                print(f"   Precio: ${ac.min():.2f} → ${ac.max():.2f}")
            else:
                print("   Precio: N/D (todo NaN)")

            # Detectar gaps grandes (> 5 días)
            gaps = df_tmp["date"].diff()
            large_gaps = gaps[gaps > pd.Timedelta(days=5)]
            if len(large_gaps) > 0:
                print(f"   ⚠️  Gaps > 5 días: {len(large_gaps)}")
            else:
                print(f"   ✅ Sin gaps significativos")


        # Limpieza: precios → mensual


        print("  ")
        print("="*70)
        print("\n🧹 LIMPIEZA | limpiando datos de precios...\n")
        print("="*70)

        precios_mensuales_clean = {}

        for ticker, df_raw in precios_raw.items():
            # Asegurar índice temporal y resample directo SIN columna 'date'
            g = df_raw.copy()

            # Garantizar DatetimeIndex UTC
            if not isinstance(g.index, pd.DatetimeIndex):
                g.index = pd.to_datetime(g.index, errors="coerce", utc=True)
            else:
                g.index = g.index.tz_localize("UTC") if g.index.tz is None else g.index.tz_convert("UTC")

            g = g.sort_index()

            # Serie de adj_close con índice datetime
            s = pd.to_numeric(g["adj_close"], errors="coerce")

            # Resample a fin de mes (último valor del mes)
            df_monthly = s.resample("ME").last().reset_index()
            df_monthly.columns = ["month_end", "adj_close"]
            df_monthly["ticker"] = ticker

            # Manejo de NaNs: backward fill
            nans_before = df_monthly["adj_close"].isna().sum()
            df_monthly["adj_close"] = df_monthly["adj_close"].bfill()
            nans_after = df_monthly["adj_close"].isna().sum()

            precios_mensuales_clean[ticker] = df_monthly

            print(f"✅ {ticker}: {len(df_monthly)} meses | NaNs: {nans_before} → {nans_after} (después bfill)")

        # Consolidado mensual
        df_precios_clean = pd.concat(precios_mensuales_clean.values(), ignore_index=True)
        df_precios_clean = df_precios_clean.sort_values(["ticker", "month_end"]).reset_index(drop=True)

        print(f"\n✅ Precios mensuales limpios: {len(df_precios_clean)} registros ({df_precios_clean['ticker'].nunique()} tickers)")
        print("="*70)


# Datos estadísticos  
        

                # === ESTADÍSTICAS: DIARIO y MENSUAL (adj_close) ==
        print("\n📊 Estadísticos básicos de adj_close (diario y mensual limpio):\n")

        for ticker, df_raw in precios_raw.items():
            # ---- Diario (normalizado) ----
            s_daily = pd.to_numeric(df_raw["adj_close"], errors="coerce").dropna()
            if s_daily.empty:
                print(f"{ticker}: sin datos diarios válidos")
            else:
                mean_d = s_daily.mean()
                std_d  = s_daily.std(ddof=1)
                min_d  = s_daily.min()
                med_d  = s_daily.median()
                max_d  = s_daily.max()
                print(f"{ticker} — Diario:")
                print(f"   media={mean_d:.4f}  std={std_d:.4f}  min={min_d:.4f}  mediana={med_d:.4f}  max={max_d:.4f}")

            # ---- Mensual (limpio, tras resample + bfill) ----
            dfm = precios_mensuales_clean.get(ticker)
            if dfm is None or dfm.empty:
                print(f"   (Mensual limpio) sin datos")
            else:
                s_m = pd.to_numeric(dfm["adj_close"], errors="coerce").dropna()
                if s_m.empty:
                    print(f"   (Mensual limpio) sin datos numéricos")
                else:
                    mean_m = s_m.mean()
                    std_m  = s_m.std(ddof=1)
                    min_m  = s_m.min()
                    med_m  = s_m.median()
                    max_m  = s_m.max()
                    print(f"   (Mensual limpio) media={mean_m:.4f}  std={std_m:.4f}  min={min_m:.4f}  mediana={med_m:.4f}  max={max_m:.4f}")

        print("\n" + "="*70)


# Formato DataFrames

        # formato de DF long
        if not all_norm:
              return pd.DataFrame(columns=["symbol","open","high","low","close","adj_close","volume"])

        df_long = pd.concat(all_norm, axis=0).sort_index()

        if out_format == "long":
            return df_long

        # Formato de DF wide
        fields = ["open","high","low","close","adj_close","volume"]
        wides = []
        for f in fields:
            sub = df_long.pivot_table(index=df_long.index, columns="symbol", values=f, aggfunc="last")
            sub.columns = pd.MultiIndex.from_product([[f], sub.columns], names=["field","symbol"])
            wides.append(sub)
        df_wide = pd.concat(wides, axis=1).sort_index()
        df_wide = df_wide.reindex(columns=pd.MultiIndex.from_product([fields, sorted(tickers)]))
        return df_wide






