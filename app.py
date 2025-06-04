# streamlit_app.py
import streamlit as st
import traceback
import time
from models.two_point_model import edge_simulator

st.title("Tokamak Erosion Lifetime Simulator")
p_in = st.number_input(
    "Input Power (MW)", 
    value=1.0,
    step=10.0,
    min_value=0.1,
    max_value=10000.0)

MEGA = 1.e6
TIME_DELAY = 5  # s

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
        erosion_lifetime = edge_simulator(power=p_in / MEGA)

        # Replace GIF with result
        placeholder.empty()
        st.success(f"Erosion Lifetime: {erosion_lifetime:.2f} years")

    except Exception as e:
        placeholder.empty()
        st.error(f"Error: {e}")
        traceback.print_exc()
