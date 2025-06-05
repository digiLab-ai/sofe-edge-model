# streamlit_app.py
import streamlit as st
import traceback
import time
from numpy import log10
from models.two_point_model import edge_simulator

st.title("Tokamak Erosion Lifetime Simulator")

MEGA = 1.e6
TIME_DELAY = 5  # s
# Log-scale input
log10_p_in = st.number_input(
    "Log₁₀(Input Power [W])", 
    value=5.0,
    step=0.1,
    min_value=5.0,
    max_value=10.0,
    format="%.2f"
)

# Convert and show actual power (linear scale)
p_in = 10 ** log10_p_in
st.markdown(f"**Input Power:** {p_in / MEGA:.3f} MW")


# Reserve a spot for the GIF or result
placeholder = st.empty()

if st.button("Run Experiment"):
    try:
        # Show GIF temporarily while processing
        with placeholder.container():
            st.image("assets/plasma.gif", caption="Running experiment...", use_container_width=True)

        # Optional: artificial delay to show the gif for effect
        time.sleep(TIME_DELAY)

        # Run your model
        erosion_lifetime = edge_simulator(power=p_in)

        # Replace GIF with result
        placeholder.empty()
        # st.success(f"Erosion Lifetime:\n {erosion_lifetime:.2f} years\n Log₁₀(Erosion Lifetime [years])={log10(erosion_lifetime):.2f}")
        st.success(
            f"""Erosion Lifetime: {erosion_lifetime:.3f} years""")
        # Log₁₀(Erosion Lifetime [years]): {log10(erosion_lifetime):.2f}""")

    except Exception as e:
        placeholder.empty()
        st.error(f"Error: {e}")
        traceback.print_exc()
