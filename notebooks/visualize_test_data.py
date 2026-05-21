from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

try:
    import ipywidgets as widgets
    from IPython.display import display

    HAS_WIDGETS = True
except Exception:
    HAS_WIDGETS = False

pd.options.display.width = 140
pd.options.display.max_columns = 50

PROJECT_ROOT = Path.cwd().parent

SCENARIOS = {
    "with_agent_1": PROJECT_ROOT
    / "metrics"
    / "total_reward_lr01_df099_epd0999_30_20_10_0_100eps_7200steps(l_reward_ 1.5 1.2 0.7 g_reward_ 1 1.0 0.5)",
    "with_agent_2": PROJECT_ROOT
    / "metrics"
    / "total_reward_lr01_df099_epd0999_acc_in_rew_30_20_10_0_100eps_7200steps(l_reward_ 1.5 1.2 0.7 g_reward_ 1 1.0 0.5)",
}
NETWORK_FILE = "network_metrics.csv"
TLS_FILE = "tls_metrics.csv"

for name, base in SCENARIOS.items():
    net_p = base / NETWORK_FILE
    tls_p = base / TLS_FILE
    print(f"[{name}] network: {net_p.exists()} -> {net_p}")
    print(f"[{name}] tls:     {tls_p.exists()} -> {tls_p}")


def load_network_metrics(path: Path, scenario: str) -> pd.DataFrame:
    df = pd.read_csv(path / NETWORK_FILE)
    need_cols = {
        "step",
        "time",
        "active_vehicles",
        "mean_speed_network",
        "total_queue_len",
        "total_waiting_time_snapshot",
    }
    missing = need_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path / NETWORK_FILE}: {missing}")
    df = df.copy()
    df["scenario"] = scenario
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["step"] = pd.to_numeric(df["step"], errors="coerce").astype("Int64")
    df["time_rounded"] = df["time"].round().astype(int)
    return df


def load_tls_metrics(path: Path, scenario: str) -> pd.DataFrame:
    df = pd.read_csv(path / TLS_FILE)
    need_cols = {
        "step",
        "time",
        "tls_id",
        "phase_index",
        "tls_queue_len",
        "tls_waiting_time_snapshot",
        "tls_mean_speed",
    }
    missing = need_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path / TLS_FILE}: {missing}")
    df = df.copy()
    df["scenario"] = scenario
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["step"] = pd.to_numeric(df["step"], errors="coerce").astype("Int64")
    df["time_rounded"] = df["time"].round().astype(int)
    df["tls_id"] = df["tls_id"].astype(str)
    return df


def infer_step_interval(df: pd.DataFrame) -> float:
    s = df.sort_values("time_rounded")["time_rounded"].drop_duplicates().diff().dropna()
    if s.empty:
        return np.nan
    return s.mode().iloc[0] if not s.mode().empty else s.median()


def plot_network_metric(df, metric, title, ytitle):
    fig = px.line(
        df.sort_values(["scenario", "time_rounded"]),
        x="time_rounded",
        y=metric,
        color="scenario",
        markers=True,
        title=title,
    )
    fig.update_layout(
        xaxis_title="time (s)",
        yaxis_title=ytitle,
        legend_title="Scenario",
        template="plotly_white",
    )
    fig.show()


def p95(x):
    return np.percentile(x, 95)


def rel_change(base_val, var_val):
    if pd.isna(base_val) or pd.isna(var_val) or base_val == 0:
        return np.nan
    return (var_val - base_val) / base_val * 100.0


def auc_trapz(y, x):
    return np.trapz(y, x)


def plot_tls_agg_metric(df, metric, title, ytitle):
    fig = px.line(
        df.sort_values(["scenario", "time_rounded"]),
        x="time_rounded",
        y=metric,
        color="scenario",
        markers=True,
        title=title,
    )
    fig.update_layout(
        xaxis_title="time (s)",
        yaxis_title=ytitle,
        legend_title="Scenario",
        template="plotly_white",
    )
    fig.show()


def plot_tls_single(tls_id: str, tls_c: pd.DataFrame):
    df = tls_c[tls_c["tls_id"] == tls_id].sort_values(["scenario", "time_rounded"])
    title_base = f"TLS {tls_id}"
    fig1 = px.line(
        df,
        x="time_rounded",
        y="tls_queue_len",
        color="scenario",
        markers=True,
        title=title_base + " — Queue length",
    )
    fig2 = px.line(
        df,
        x="time_rounded",
        y="tls_waiting_time_snapshot",
        color="scenario",
        markers=True,
        title=title_base + " — Waiting time snapshot",
    )
    fig3 = px.line(
        df,
        x="time_rounded",
        y="tls_mean_speed",
        color="scenario",
        markers=True,
        title=title_base + " — Mean speed",
    )
    for fig in (fig1, fig2, fig3):
        fig.update_layout(
            xaxis_title="time (s)", template="plotly_white", legend_title="Scenario"
        )
        fig.show()


dfs_net = []
dfs_tls = []

for name, base in SCENARIOS.items():
    dfs_net.append(load_network_metrics(base, name))
    dfs_tls.append(load_tls_metrics(base, name))

net = pd.concat(dfs_net, ignore_index=True)
tls = pd.concat(dfs_tls, ignore_index=True)

print("Network rows:", len(net), "TLS rows:", len(tls))
print("Scenarios:", net["scenario"].unique().tolist())
print("TLS count:", tls["tls_id"].nunique())

times_with_1 = set(
    net.query("scenario == 'with_agent_1'")["time_rounded"].unique().tolist()
)
times_with_2 = set(
    net.query("scenario == 'with_agent_2'")["time_rounded"].unique().tolist()
)
common_times = sorted(times_with_1 & times_with_2)

print(
    f"Common sampled times: {len(common_times)} with_1={len(times_with_1)}, with_1={len(times_with_2)})"
)

if len(common_times) == 0:
    raise RuntimeError(
        "Нет общих временных отметок между сценариями. Проверьте STEP_INTERVAL и файлы."
    )

net_c = net[net["time_rounded"].isin(common_times)].copy()
tls_c = tls[tls["time_rounded"].isin(common_times)].copy()

print("Inferred sampling interval (s):", infer_step_interval(net_c))

plot_network_metric(net_c, "mean_speed_network", "Mean Speed (network)", "m/s")
plot_network_metric(
    net_c, "total_queue_len", "Total Queue Length (network)", "vehicles stopped"
)
plot_network_metric(
    net_c,
    "total_waiting_time_snapshot",
    "Total Waiting Time Snapshot (network)",
    "seconds",
)
plot_network_metric(net_c, "active_vehicles", "Active Vehicles (network)", "count")
