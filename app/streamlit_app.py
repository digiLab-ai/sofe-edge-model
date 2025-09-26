
from __future__ import annotations
import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Callable
import os, sys

ROOT = os.path.dirname(__file__)
REPO = os.path.dirname(ROOT)
SRC = os.path.join(REPO, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from demos_edge_app.two_point_model import edge_simulator

KEY_LIME   = "#EBF38B"
INDIGO     = "#16425B"
INDIGO_50  = "#8AA0AD"
KEPPEL     = "#16D5C2"
KEPPEL_50  = "#8AEAE1"
BLACK      = "#000000"
GREY_80    = "#333333"

try:
    import altair as alt
except Exception:
    alt = None

def apply_branding():
    if st.session_state.get("_branding_injected"):
        return
    BRAND_CSS = f"""
    <style>
      :root {{
        --brand-primary: {INDIGO};
        --brand-primary-50: {INDIGO_50};
        --brand-accent: {KEPPEL};
        --brand-accent-50: {KEPPEL_50};
        --brand-accent-2: {KEY_LIME};
        --brand-black: {BLACK};
        --brand-grey-80: {GREY_80};
      }}
      h1, h2, h3 {{ color: var(--brand-black); }}
      div.stButton > button, div.stDownloadButton > button {{
        background-color: var(--brand-primary);
        color: white;
        border: 0;
        border-radius: 12px;
      }}
      div.stButton > button:hover, div.stDownloadButton > button:hover {{
        background-color: {INDIGO_50};
      }}
      div.streamlit-expanderHeader {{
        background: linear-gradient(90deg, var(--brand-accent-50) 0%, var(--brand-accent-2) 100%);
        color: var(--brand-black);
        border-radius: 8px;
      }}
      label {{ color: var(--brand-grey-80); }}
      [data-testid="stTable"] thead th {{
        background-color: var(--brand-accent-50);
        color: var(--brand-black);
      }}
    </style>
    """
    st.markdown(BRAND_CSS, unsafe_allow_html=True)
    st.session_state["_branding_injected"] = True

UNITS = {
    "power": "MW",
    "upstream_density": "10¹⁹ m⁻³",
    "f_cond": "–",
    "f_mom": "–",
    "f_pow": "–",
    "connection_length": "m",
    "lambda_q": "m",
    "R_m": "m",
    "erosion_lifetime": "years",
}

@st.cache_data(show_spinner=False)
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def variable_fixed_controls(
    label: str,
    ranges: Dict[str, Tuple[float, float]],
    defaults: Dict[str, float],
    default_variable: Tuple[str, ...] = ("power", "upstream_density"),
):
    st.subheader(f"{label} • Variable/Fixed Controls")
    n = st.number_input(
        "Number of points (per sweep / random batch)",
        min_value=1, max_value=200_000, value=20, step=10, key=f"{label}_vf_n"
    )

    var_flags: Dict[str, bool] = {}
    var_ranges: Dict[str, Tuple[float, float]] = {}
    fixed_vals: Dict[str, float] = {}

    with st.expander("Select variables to vary and set ranges / fixed values", expanded=False):
        for k, (a, b) in ranges.items():
            cols = st.columns([1.6, 1, 1, 1.2])

            # Only 'power' and 'upstream_density' default to variable (unless user already set it in session)
            default_var = (k in default_variable)
            var_flags[k] = cols[0].checkbox(
                f"{k} is variable",
                value=default_var if f"{label}_{k}_is_var" not in st.session_state else None,
                key=f"{label}_{k}_is_var"
            )

            if var_flags[k]:
                new_a = cols[1].number_input(f"{k} min", value=float(a), key=f"{label}_{k}_a_vf")
                new_b = cols[2].number_input(f"{k} max", value=float(b), key=f"{label}_{k}_b_vf")
                if new_b < new_a:
                    st.warning(f"Adjusted {k} max to be ≥ min"); new_b = new_a
                var_ranges[k] = (new_a, new_b)
                cols[3].markdown("&nbsp;")
            else:
                # Use provided defaults (not mid-range) as the starting fixed values
                c = defaults.get(k, (a + b) / 2.0)
                fixed_vals[k] = cols[3].number_input(
                    f"{k} (fixed)", value=float(c), key=f"{label}_{k}_fixed_vf"
                )

    return int(n), var_flags, var_ranges, fixed_vals


def custom_points_editor(label: str, ranges: Dict[str, Tuple[float, float]], defaults: Dict[str, float]) -> pd.DataFrame:
    st.subheader(f"{label} • Custom Points")
    st.caption("Add/remove rows to choose exact input points where the model will be evaluated.")
    init = {k: [defaults.get(k, (a + b) / 2)] for k, (a, b) in ranges.items()}
    return st.data_editor(pd.DataFrame(init), num_rows="dynamic", use_container_width=True, key=f"{label}_custom_editor")

def upload_inputs(label: str, ranges: Dict[str, Tuple[float, float]]):
    st.subheader(f"{label} • Upload Inputs")
    st.caption(f"Expected columns: {list(ranges.keys())}")
    file = st.file_uploader(f"Upload inputs CSV for {label}", type=["csv"], key=f"{label}_inputs_uploader")
    if file is None: return None
    try:
        df = pd.read_csv(file)
        missing = [c for c in ranges.keys() if c not in df.columns]
        if missing: st.error(f"Missing columns: {missing}"); return None
        return df
    except Exception as e:
        st.error(f"Failed to read CSV: {e}"); return None

def preview_and_download(key_prefix: str, default_prefix: str):
    X = st.session_state.get(f"{key_prefix}_inputs")
    y = st.session_state.get(f"{key_prefix}_outputs")
    combined = st.session_state.get(f"{key_prefix}_combined")
    if isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
        with st.expander("Preview, Plot & Download", expanded=True):
            st.subheader("Preview: Inputs"); st.dataframe(X, use_container_width=True)
            st.subheader("Preview: Outputs"); st.dataframe(y, use_container_width=True)
            st.subheader("Plot (scatter)")
            colx, coly = st.columns(2)
            with colx: x_var = st.selectbox("X axis (input variable)", list(X.columns), key=f"{key_prefix}_plot_x")
            with coly: y_var = st.selectbox("Y axis (output variable)", list(y.columns), key=f"{key_prefix}_plot_y")
            plot_df = pd.DataFrame({x_var: X[x_var].to_numpy(), y_var: y[y_var].to_numpy()})
            try:
                chart = (
                    alt.Chart(plot_df)
                    .mark_circle(size=64, color=INDIGO)
                    .encode(
                        x=alt.X(x_var, title=f"{x_var} [{UNITS.get(x_var, '')}]"),
                        y=alt.Y(y_var, title=f"{y_var} [{UNITS.get(y_var, '')}]"),
                        tooltip=[x_var, y_var],
                    )
                    .interactive()
                    .properties(width=400, height=400)
                )
                st.altair_chart(chart, use_container_width=False)

                # chart = (alt.Chart(plot_df).mark_circle(size=64, color=INDIGO).encode(x=alt.X(x_var), y=alt.Y(y_var), tooltip=[x_var, y_var]).interactive())
                # st.altair_chart(chart, use_container_width=True)
            except Exception:
                st.scatter_chart(plot_df, x=x_var, y=y_var, use_container_width=True)
            st.markdown("### Download")
            default_name = f"{default_prefix}.csv"
            st.text_input("Dataset CSV filename. Press Enter to set.", value=default_name, key=f"{key_prefix}_dl_name")
            current_name = st.session_state.get(f"{key_prefix}_dl_name", default_name)
            st.download_button("Download dataset CSV", data=to_csv_bytes(combined), file_name=current_name, mime="text/csv", key=f"{key_prefix}_dl_btn")

def compute_erosion_lifetime_df(X: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, row in X.iterrows():
        params = {k: float(row[k]) for k in X.columns}
        # Unit conversions: UI uses power [MW], upstream_density [10^19 m^-3]
        if "power" in params:
            params["power"] = params["power"] * 1e6  # MW -> W
        if "upstream_density" in params:
            params["upstream_density"] = params["upstream_density"] * 1e19  # (1e19 m^-3 units) -> m^-3
        val = edge_simulator(**params)
        # try:
        #     val = edge_simulator(**params)
        # except TypeError:
        #     import inspect
        #     sig = inspect.signature(edge_simulator)
        #     allowed = {k: params[k] for k in params.keys() if k in sig.parameters}
        #     val = edge_simulator(**allowed)
        out.append(val)
    return pd.DataFrame({"erosion_lifetime": np.array(out, dtype=float)})

def run_tab(
    *,
    label: str,
    key_prefix: str,
    ranges: Dict[str, Tuple[float, float]],
    rng: np.random.Generator,
    default_prefix: str,
    fixed_defaults: Dict[str, float] | None = None,
    default_variable: Tuple[str, ...] = ("power", "upstream_density"),
):
    apply_branding()
    # Use caller-specified fixed defaults if provided; else fall back to midpoints
    if fixed_defaults is None:
        defaults = {k: (a + b) / 2 for k, (a, b) in ranges.items()}
    else:
        defaults = {k: fixed_defaults.get(k, (a + b) / 2) for k, (a, b) in ranges.items()}

    st.divider(); st.markdown("### Input Source")
    input_mode = st.radio(
        "Choose how to provide inputs",
        ["Auto sampling", "Custom points", "Upload inputs"],
        key=f"{label}_input_mode", horizontal=True
    )
    X = None
    if input_mode == "Auto sampling":
        n, var_flags, var_ranges, fixed_vals = variable_fixed_controls(
            label, ranges, defaults, default_variable=default_variable
        )
        sampling_type = st.selectbox("Sampling type", ["Random (uniform)", "Structured grid"], index=0, key=f"{label}_sampling_type")
        variable_keys = [k for k, flag in var_flags.items() if flag]
        if len(variable_keys) == 0:
            X = pd.DataFrame({k: [fixed_vals[k]] for k in ranges.keys()})
        elif len(variable_keys) == 1:
            sweep_key = variable_keys[0]; a, b = var_ranges[sweep_key]
            x = rng.uniform(a, b, size=n) if sampling_type == "Random (uniform)" else np.linspace(a, b, n)
            data = {k: (x if k == sweep_key else np.full(n, fixed_vals[k])) for k in ranges.keys()}; X = pd.DataFrame(data)
        else:
            if sampling_type == "Random (uniform)":
                data = {k: (rng.uniform(*var_ranges[k], size=n) if var_flags[k] else np.full(n, fixed_vals[k])) for k in ranges.keys()}
                X = pd.DataFrame(data)
            else:
                d = len(variable_keys); steps = max(2, int(round(n ** (1 / d)))); total = steps ** d
                if total > 100_000: st.error("Structured grid too large (>100,000 points). Reduce requested points."); X=None
                else:
                    axes = [np.linspace(*var_ranges[k], steps) for k in variable_keys]
                    grids = np.meshgrid(*axes, indexing="xy"); flat = [g.reshape(-1) for g in grids]
                    grid_df = pd.DataFrame({k: v for k, v in zip(variable_keys, flat)})
                    for k in ranges.keys():
                        if not var_flags[k]: grid_df[k] = fixed_vals[k]
                    X = grid_df[[*ranges.keys()]]
        if st.button(f"Generate {label} Data", key=f"{label}_go_vf"): pass
    elif input_mode == "Custom points":
        edited = custom_points_editor(label, ranges, defaults)
        if st.button(f"Generate {label} Data from Custom Points", key=f"{label}_go_custom"): X = edited.copy()
    else:
        up = upload_inputs(label, ranges)
        if st.button(f"Generate {label} Data from Uploaded Inputs", key=f"{label}_go_upload"):
            if up is None: st.error("Please upload a valid inputs CSV with the expected columns.")
            else: X = up.copy()
    if isinstance(X, pd.DataFrame):
        y = compute_erosion_lifetime_df(X)
        combined = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        st.session_state[f"{key_prefix}_inputs"] = X; st.session_state[f"{key_prefix}_outputs"] = y; st.session_state[f"{key_prefix}_combined"] = combined
        st.success(f"Generated {label} dataset.")
    preview_and_download(key_prefix, default_prefix)

st.set_page_config(page_title="Two Point Model", layout="wide")
st.title("Two-Point Model")

# Image banner
img_path = os.path.join(REPO, "assets", "plasma.gif")
col1, col2 = st.columns([1,1])  # two equal halves
with col1:
    if os.path.exists(img_path): st.image(img_path, 
                                    #   caption="Running experiment...", 
                                      use_container_width=True)
with col2:
    st.empty()  # leave blank or add text

st.caption("Choose inputs, evaluate erosion lifetime, preview, plot, and download a CSV.")
with st.sidebar:
    st.header("Inputs configuration")
    default_names = ["power", "upstream_density", "f_cond", "f_mom", "f_pow", "connection_length", "lambda_q", "R_m"]
    ranges: Dict[str, Tuple[float, float]] = {
        "power": (0.1, 500.0),                 # MW
        "upstream_density": (0.1, 100.0),      # 1e19 m^-3
        "f_cond": (0.5, 1.0),                  # –
        "f_mom": (0.5, 1.0),                   # –
        "f_pow": (0.0, 0.95),                  # –
        "connection_length": (10, 200),        # m
        "lambda_q": (0.001, 0.1),              # m
        "R_m": (1, 20),                        # m
    }

    # Choose which inputs are active (column order preserved)
    n_inputs = st.slider("Number of input variables (first N are active)",
                         1, len(default_names), len(default_names), 1)
    active_keys = default_names[:n_inputs]
    ranges = {k: ranges[k] for k in active_keys}

    # Your chosen *fixed-value* defaults (these are just examples — edit as you like)
    fixed_defaults = {
        "power": 50.0,               # MW 
        "upstream_density": 10.0,     # 1e19 m^-3
        "f_cond": 0.9,
        "f_mom": 0.9,
        "f_pow": 0.4,
        "connection_length": 50.0,
        "lambda_q": 0.005,
        "R_m": 5.,
    }

# Run with your defaults and only power + upstream_density variable by default
rng = np.random.default_rng(42)
run_tab(
    label="Erosion Lifetime",
    key_prefix="edge",
    ranges=ranges,
    rng=rng,
    default_prefix="erosion_lifetime",
    fixed_defaults=fixed_defaults,
    default_variable=("power", "upstream_density"),
)


with st.expander("Parameters & Units (summary)", expanded=False):
    units_df = pd.DataFrame([
        {"Parameter": "power", "Units": "MW", "Description": "Plasma power."},
        {"Parameter": "upstream_density", "Units": "10^19 m^-3", "Description": "Upstream electron number density."},
        {"Parameter": "f_cond", "Units": "–", "Description": "Fraction of heat flux density carried by conduction."},
        {"Parameter": "f_mom", "Units": "–", "Description": "Fraction of momentum dissipated from upstream to downstream."},
        {"Parameter": "f_pow", "Units": "–", "Description": "Fraction of power dissipated from upstream to downstream."},
        {"Parameter": "connection_length", "Units": "m", "Description": "Magnetic connection length from outer midplane to the target."},
        {"Parameter": "lambda_q", "Units": "m", "Description": "The scrape off layer width at the outer midplane."},
        {"Parameter": "R_m", "Units": "m", "Description": "The major radius at the outer midplane."},
        {"Parameter": "erosion_lifetime", "Units": "years", "Description": "The erosion lifetime of plasma facing materials."},
    ])
    st.table(units_df)
