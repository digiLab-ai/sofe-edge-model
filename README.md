# demos-edge-app

Streamlit app to generate datasets for a **two-point edge model**.
It wraps your `edge_simulator(...)` in `src/demos_edge_app/two_point_model.py` to compute **erosion_lifetime** for batches of inputs.

## Run
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Inputs (defaults)
- power [W] — 1e5 … 5e7
- upstream_density [m^-3] — 1e18 … 1e21
- parallel_connection_length [m] — 1e2 … 2e4
- f_c [–] — 0 … 1
- f_mom [–] — 0 … 1
- f_power [–] — 0 … 1

**Output:** erosion_lifetime [years]
