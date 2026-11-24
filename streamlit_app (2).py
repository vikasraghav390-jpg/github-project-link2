"""Community Microgrid Energy Forecasting - SELF-CONTAINED Streamlit dashboard.
Needs only: streamlit, pandas, numpy, scikit-learn (see requirements.txt). No other project files required.
Run: streamlit run streamlit_app.py"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

TEST_H, THR = 18 * 24, 40.0


# ---------------- synthetic data (90 days, 120 kWp, seed 42) ----------------
def generate(days=90, start="2026-08-01", seed=42, kwp=120):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 24, freq="h")
    n, h, d, dow = len(idx), idx.hour.values, np.arange(len(idx)) // 24, idx.dayofweek.values
    cloud_day = np.clip(rng.beta(5, 2, days), 0.3, 1.0)
    cloud = np.clip(cloud_day[d] * (1 + 0.08 * rng.standard_normal(n)), 0.1, 1.0)
    clear = np.where((h >= 6) & (h <= 18), np.sin(np.pi * (h - 6) / 12), 0.0)
    irr = clear * cloud
    temp = 30 + rng.normal(0, 2, days)[d] + 5 * np.sin(2 * np.pi * (h - 9) / 24) + rng.normal(0, 0.5, n)
    pv = kwp * 0.82 * irr * (1 - 0.004 * np.maximum(temp - 25, 0))
    g = lambda mu, s: np.exp(-0.5 * ((h - mu) / s) ** 2)
    load = 39 + 23 * g(7.7, 1.6) + 16 * g(12.5, 3.0) + 53 * g(20, 1.8)
    load = load * (1 + 0.03 * rng.normal(0, 1, days))[d] * np.where(dow >= 5, 1.10, 1.0)
    load = (load + 1.5 * np.maximum(temp - 28, 0)) * (1 + 0.05 * rng.standard_normal(n))
    return pd.DataFrame({
        "ts": idx, "hour": h, "dow": dow, "temp": temp, "irr": irr,
        "pv_kw": np.clip(pv, 0, None), "load_kw": load,
        "irr_forecast": np.clip(irr * (1 + 0.12 * rng.standard_normal(n)), 0, None),
        "temp_forecast": temp + rng.normal(0, 1.0, n)})


def make_features(df, target):
    p = {"pv_kw": "pv", "load_kw": "load"}[target]
    X = pd.DataFrame(index=df.index)
    X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * df.hour / 24), np.cos(2 * np.pi * df.hour / 24)
    X["dow"], X["weekend"] = df.dow, (df.dow >= 5).astype(int)
    for L in (24, 25, 48, 168):                      # every lag >= 24 h horizon: no leakage
        X[f"{p}_lag{L}"] = df[target].shift(L)
    X["roll24_mean"] = df[target].shift(24).rolling(24).mean()
    X["irr_fc"], X["temp_fc"] = df.irr_forecast, df.temp_forecast
    return X.dropna()


def metrics(y, p):
    mae = mean_absolute_error(y, p)
    return {"mae": mae, "rmse": mean_squared_error(y, p) ** 0.5, "nmae": 100 * mae / np.mean(y)}


def simulate(pv, load, cap=200.0, pmax=50.0, eff=0.95, soc_min=0.10, soc0=0.50, use_battery=True):
    e, soc, imp, exp = cap * soc0, [], 0.0, 0.0
    for p, l in zip(pv, load):
        net = l - p
        if use_battery and net < 0:
            c = min(-net, pmax, (cap - e) / eff); e += c * eff; net += c
        elif use_battery and net > 0:
            x = min(net, pmax, max(e - cap * soc_min, 0) * eff); e -= x / eff; net -= x
        imp += max(net, 0); exp += max(-net, 0); soc.append(100 * e / cap)
    return {"import_kwh": imp, "export_kwh": exp, "self_sufficiency": 100 * (1 - imp / float(np.sum(load))), "soc": soc}


def advice(net, soc, thr=THR):
    peaks = [v for v in net if v > thr]
    if not peaks:
        return "No peak expected. Keep the battery in normal cycling."
    return (f"Peak net load expected in {len(peaks)} h (max {max(peaks):.0f} kW > {thr:.0f} kW). "
            f"Battery SoC {soc:.0f}%: hold charge for the peak and shift flexible loads.")


@st.cache_data(show_spinner="Training forecasting models (first load only)...")
def run():
    df = generate()
    res, pred = {}, {}
    for tgt in ("pv_kw", "load_kw"):
        X = make_features(df, tgt); y = df.loc[X.index, tgt]
        Xtr, Xte, ytr, yte = X.iloc[:-TEST_H], X.iloc[-TEST_H:], y.iloc[:-TEST_H], y.iloc[-TEST_H:]
        pers = df[tgt].shift(24).loc[Xte.index]
        res[(tgt, "Persistence (t-24h)")] = metrics(yte, pers)
        pred[(tgt, "actual")], pred[(tgt, "persistence")] = yte.values, pers.values
        for name, m in {"Ridge Regression": Ridge(alpha=1.0),
                        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
                        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42)}.items():
            m.fit(Xtr, ytr); p = np.clip(m.predict(Xte), 0, None)
            res[(tgt, name)], pred[(tgt, name)] = metrics(yte, p), p
    pa, la = pred[("pv_kw", "actual")], pred[("load_kw", "actual")]
    pf, lf = pred[("pv_kw", "Ridge Regression")], pred[("load_kw", "Ridge Regression")]
    a, f = (la - pa) > THR, (lf - pf) > THR
    tp_, fp_, fn_ = int((a & f).sum()), int((~a & f).sum()), int((a & ~f).sum())
    alert = {"tp": tp_, "fp": fp_, "fn": fn_, "precision": 100 * tp_ / max(tp_ + fp_, 1), "recall": 100 * tp_ / max(tp_ + fn_, 1)}
    base, bat = simulate(pa, la, use_battery=False), simulate(pa, la)
    tp = pd.DataFrame({"ts": df.ts.iloc[-TEST_H:].values, "pv_actual": pa, "pv_forecast": pf,
                       "pv_persistence": pred[("pv_kw", "persistence")], "load_actual": la, "load_forecast": lf,
                       "load_persistence": pred[("load_kw", "persistence")], "soc": bat["soc"]})
    return res, alert, base, bat, tp


st.set_page_config(page_title="Community Microgrid Forecast", layout="wide")
st.title("Community Microgrid Energy Forecasting")
st.caption("HCLTech BTech Learner Program - Embedded Systems & IoT | Vikas Raghav")
st.warning("All results shown are from a 90-day synthetic dataset (simulation), not field measurements.")
res, alert, base, bat, tp = run()
tab1, tab2 = st.tabs(["Forecast", "Model metrics"])

with tab1:
    tp["date"] = tp.ts.dt.date
    day = st.selectbox("Test day", sorted(tp.date.unique()))
    d = tp[tp.date == day].set_index("ts")
    c1, c2 = st.columns(2)
    c1.subheader("PV generation (kW)"); c1.line_chart(d[["pv_actual", "pv_forecast", "pv_persistence"]])
    c2.subheader("Community load (kW)"); c2.line_chart(d[["load_actual", "load_forecast", "load_persistence"]])
    st.subheader("Battery SoC (simulated, %)"); st.line_chart(d[["soc"]])
    st.info(advice((d.load_forecast - d.pv_forecast).values, float(d.soc.iloc[0])))

with tab2:
    rows = [{"target": t, "model": m, **{k: round(float(v), 2) for k, v in x.items()}} for (t, m), x in res.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.metric("Peak alert precision %", f"{alert['precision']:.1f}"); c2.metric("Peak alert recall %", f"{alert['recall']:.1f}")
    st.subheader("Battery dispatch (200 kWh / 50 kW)")
    st.dataframe(pd.DataFrame([
        {"case": "No battery", "import_kwh": round(base["import_kwh"]), "export_kwh": round(base["export_kwh"]), "self_sufficiency_%": round(base["self_sufficiency"], 1)},
        {"case": "With battery", "import_kwh": round(bat["import_kwh"]), "export_kwh": round(bat["export_kwh"]), "self_sufficiency_%": round(bat["self_sufficiency"], 1)}]))
