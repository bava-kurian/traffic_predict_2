import streamlit as st
import joblib
import numpy as np
import os
from datetime import datetime
import time
from streamlit_autorefresh import st_autorefresh

MODEL_PATH = os.path.join("train", "xgb_traffic_predictor.pkl")
MAX_SIGNAL_DURATION = 90  # seconds

# Load model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("🚦 Traffic Junction Signal Simulator")
st.write("Simulate real-time traffic signal adjustments using an ML model.")

# UI widgets
col1, col2 = st.columns(2)
with col1:
    hour = st.slider("Hour of Day", 0, 23, datetime.now().hour)
    day = st.slider("Day", 1, 31, datetime.now().day)
    vehicles = st.slider("Vehicles Detected", 0, 100, 10)
with col2:
    day_of_week = st.selectbox("Day of Week", list(enumerate(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])), format_func=lambda x: x[1])[0]
    month = st.selectbox("Month", list(range(1, 13)), format_func=lambda x: datetime(2000, x, 1).strftime('%B'))
    emergency = st.checkbox("Emergency Vehicle Detected")

# Prediction logic
if emergency:
    pred_count = vehicles
    signal_time = MAX_SIGNAL_DURATION
    emergency_override = True
else:
    input_arr = np.array([[hour, day, day_of_week, month, vehicles]])
    pred = model.predict(input_arr)[0]
    pred_count = int(round(pred))
    signal_time = min(MAX_SIGNAL_DURATION, max(10, int(pred_count * 2)))
    emergency_override = False

# Store recent signal durations in session state
def init_session():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
init_session()

# Update history
if st.button("Simulate Signal"):
    st.session_state['history'].append({
        'time': datetime.now().strftime('%H:%M:%S'),
        'signal_duration': signal_time
    })

# Display results
st.subheader("Results")
st.metric("Predicted Vehicle Count", pred_count)
st.metric("Signal Duration (sec)", signal_time)
if emergency_override:
    st.warning("Emergency override: Signal set to max duration!")

# Optional: Time series graph
if st.session_state['history']:
    st.subheader("Recent Signal Durations")
    import pandas as pd
    df = pd.DataFrame(st.session_state['history'])
    st.line_chart(df.set_index('time'))

def traffic_light_display(state):
    color_map = {
        "red": "#FF0000",
        "yellow": "#FFFF00",
        "green": "#00FF00"
    }
    cols = st.columns(3)
    for i, light in enumerate(["red", "yellow", "green"]):
        if state == light:
            cols[i].markdown(f"<div style='width:60px;height:60px;border-radius:50%;background:{color_map[light]};margin:auto;'></div>", unsafe_allow_html=True)
        else:
            cols[i].markdown(f"<div style='width:60px;height:60px;border-radius:50%;background:#333;margin:auto;'></div>", unsafe_allow_html=True)

# --- Traffic Light Simulation State ---
def init_light_simulation():
    if 'light_state' not in st.session_state:
        st.session_state['light_state'] = 'green'
    if 'light_time_left' not in st.session_state:
        st.session_state['light_time_left'] = signal_time
    if 'simulation_running' not in st.session_state:
        st.session_state['simulation_running'] = False

if st.session_state['history']:
    st.subheader("Traffic Light Simulation")
    init_light_simulation()

    # Start/Stop buttons
    col_sim1, col_sim2 = st.columns(2)
    if col_sim1.button("Start Simulation"):
        st.session_state['simulation_running'] = True
    if col_sim2.button("Reset Simulation"):
        st.session_state['light_state'] = 'green'
        st.session_state['light_time_left'] = signal_time
        st.session_state['simulation_running'] = False

    # Auto-refresh if running
    if st.session_state['simulation_running']:
        st_autorefresh(interval=1000, key="light_sim_refresh")

        # Advance the simulation
        if st.session_state['light_state'] == 'green':
            if st.session_state['light_time_left'] > 6:
                st.session_state['light_time_left'] -= 1
            else:
                st.session_state['light_state'] = 'yellow'
                st.session_state['light_time_left'] = 5
        elif st.session_state['light_state'] == 'yellow':
            if st.session_state['light_time_left'] > 1:
                st.session_state['light_time_left'] -= 1
            else:
                st.session_state['light_state'] = 'red'
                st.session_state['light_time_left'] = 5
        elif st.session_state['light_state'] == 'red':
            if st.session_state['light_time_left'] > 1:
                st.session_state['light_time_left'] -= 1
            else:
                st.session_state['light_state'] = 'green'
                st.session_state['light_time_left'] = signal_time

    # Display the current light
    traffic_light_display(st.session_state['light_state'])
    st.write(f"Time left: {st.session_state['light_time_left']} seconds") 