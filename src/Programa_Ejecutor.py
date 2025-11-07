#############
# Bloque principal de ejecución
#############


if __name__ == "__main__":
    # ========= Parámetros =========
    yahoo_tickers = ["AAPL", "MSFT", "SPY"]
    alpha_tickers = ["AAPL", "MSFT", "SPY"]   # opcional
    start = "2024-01-01"
    end   = "2024-12-31"

    # (Opcional) Clave de AlphaVantage para también mostrar WIDE de esa fuente
    ALPHA_API_KEY = ""  # pon tu clave o déjalo vacío para omitir AlphaVantage

    # ========= 1) Yahoo: extracción (LONG, para objetos y Monte Carlo) =========
    ext_y = DataExtractorAPIs()
    df_long_y = ext_y.multiple_tickers(
        tickers=yahoo_tickers,
        source="yahoo",
        start=start,
        end=end,
        out_format="long"   # usamos LONG internamente, no lo mostramos
    )

    # ========= 2) Mostrar SOLO el formato WIDE (derivado localmente del LONG) =========
    print("\n" + "="*70)
    print("📋 Vista WIDE de precios (Yahoo) — (open, high, low, close, adj_close, volume)")
    print("="*70)

    fields = ["open", "high", "low", "close", "adj_close", "volume"]
    wides = []
    for f in fields:
        sub = df_long_y.pivot_table(index=df_long_y.index, columns="symbol", values=f, aggfunc="last")
        sub.columns = pd.MultiIndex.from_product([[f], sub.columns], names=["field", "symbol"])
        wides.append(sub)
    df_wide_yahoo = pd.concat(wides, axis=1).sort_index()
    df_wide_yahoo = df_wide_yahoo.reindex(columns=pd.MultiIndex.from_product([fields, sorted(yahoo_tickers)]))
    print(df_wide_yahoo.head())

    # ========= 3) (Opcional) AlphaVantage: extracción y mostrar WIDE =========
    if ALPHA_API_KEY:
        print("\n" + "="*70)
        print("🔗 AlphaVantage: mostrando WIDE (diario)")
        print("="*70)
        ext_av = DataExtractorAPIs(alpha_api_key=ALPHA_API_KEY)
        df_wide_av = ext_av.multiple_tickers(
            tickers=alpha_tickers,
            source="alphavantage",
            out_format="wide"    # pedimos WIDE directamente
        )
        print(df_wide_av.head())
    else:
        print("\n(ℹ️ No se ha configurado ALPHA_API_KEY; omitiendo AlphaVantage.)")

    # ========= 4) Objetos y Cartera (desde Yahoo) =========
    series_by_symbol = make_price_series_by_symbol(df_long_y)
    pf = Portfolio(
        series_by_symbol=series_by_symbol,
        weights={"AAPL": 0.4, "MSFT": 0.35, "SPY": 0.25}  # se normalizan
    )

    # ========= 5) Simulación Monte Carlo =========
    sim = pf.simulate_monte_carlo(
        horizon_days=252,
        n_sims=2000,
        log_returns=True,
        with_correlation=True,
        seed=42
    )

    # ========= 6) MOSTRAR TODAS LAS GRÁFICAS =========
    # 6.1 Fan chart de la cartera y fan charts de los primeros activos
    pf.plot_results(
        sim_result=sim,
        show_assets=True,
        quantiles=[0.05, 0.5, 0.95],
        title="Monte Carlo — Cartera"
    )

    # 6.2 Gráficas de “report”: histograma final, heatmap correlaciones, precios normalizados
    saved_paths = pf.plots_report(
        sim_result=sim,
        show_assets=True,       # fan charts de algunos activos (si tu método los pinta)
        asset_examples=2,
        save=True,              # guarda PNGs además de mostrarlos
        outdir="plots_report",
        quantiles=[0.05, 0.5, 0.95],
        title_portfolio="Monte Carlo — Cartera"
    )
    print("\nImágenes guardadas:", saved_paths)

    # (Opcional) resumen e informe en Markdown
    pf.summary(sim)
    md = pf.report(sim, alpha=0.05, include_assets=True, annualize=True, show=True)

    print("\n✅ Ejecución completada.")