import streamlit as st
import joblib
import numpy as np
import os
from datetime import datetime
import time
from streamlit_autorefresh import st_autorefresh

MODEL_PATH = os.path.join("train", "xgb_traffic_predictor.pkl")
MAX_SIGNAL_DURATION = 90  # seconds
DIRECTIONS = ["North", "East", "South", "West"]

# Load model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("🚦 Traffic Junction Signal Simulator (4 Directions)")
st.write("Simulate real-time traffic signal adjustments for a 4-way intersection using an ML model.")

# UI widgets for time/date
col_time1, col_time2 = st.columns(2)
with col_time1:
    hour = st.slider("Hour of Day", 0, 23, datetime.now().hour)
    day = st.slider("Day", 1, 31, datetime.now().day)
with col_time2:
    day_of_week = st.selectbox("Day of Week", list(enumerate(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])), format_func=lambda x: x[1])[0]
    month = st.selectbox("Month", list(range(1, 13)), format_func=lambda x: datetime(2000, x, 1).strftime('%B'))

# UI widgets for vehicle counts per direction
st.subheader("Vehicle Counts Per Direction")
vehicle_counts = {}
for direction in DIRECTIONS:
    vehicle_counts[direction] = st.slider(f"{direction} Vehicles", 0, 100, 10)

# Emergency vehicle detection per direction (optional, can be extended)
emergency = st.checkbox("Emergency Vehicle Detected (any direction)")

# Prediction logic for each direction
green_times = {}
pred_counts = {}
if emergency:
    for direction in DIRECTIONS:
        pred_counts[direction] = vehicle_counts[direction]
        green_times[direction] = MAX_SIGNAL_DURATION
    emergency_override = True
else:
    for direction in DIRECTIONS:
        input_arr = np.array([[hour, day, day_of_week, month, vehicle_counts[direction]]])
        pred = model.predict(input_arr)[0]
        pred_counts[direction] = int(round(pred))
        green_times[direction] = min(MAX_SIGNAL_DURATION, max(10, int(pred_counts[direction] * 2)))
    emergency_override = False

# Store recent signal durations in session state
def init_session():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'cycle_idx' not in st.session_state:
        st.session_state['cycle_idx'] = 0
init_session()

# Update history
if st.button("Simulate Signal Cycle"):
    now = datetime.now().strftime('%H:%M:%S')
    for direction in DIRECTIONS:
        st.session_state['history'].append({
            'time': now,
            'direction': direction,
            'signal_duration': green_times[direction]
        })

# Display results
st.subheader("Predicted Results Per Direction")
for direction in DIRECTIONS:
    st.metric(f"{direction} - Predicted Vehicle Count", pred_counts[direction])
    st.metric(f"{direction} - Signal Duration (sec)", green_times[direction])
if emergency_override:
    st.warning("Emergency override: All signals set to max duration!")

# Optional: Time series graph
if st.session_state['history']:
    st.subheader("Recent Signal Durations (All Directions)")
    import pandas as pd
    df = pd.DataFrame(st.session_state['history'])
    st.line_chart(df.pivot(index='time', columns='direction', values='signal_duration'))

def traffic_light_display(state, active_direction):
    color_map = {
        "red": "#FF0000",
        "yellow": "#FFFF00",
        "green": "#00FF00"
    }
    cols = st.columns(4)
    for i, direction in enumerate(DIRECTIONS):
        if direction == active_direction:
            color = color_map[state]
        else:
            color = color_map["red"]
        cols[i].markdown(f"<div style='width:60px;height:60px;border-radius:50%;background:{color};margin:auto;'></div>", unsafe_allow_html=True)
        cols[i].markdown(f"<div style='text-align:center'>{direction}</div>", unsafe_allow_html=True)
        

# --- Traffic Light Simulation State ---
def init_light_simulation():
    if 'light_state' not in st.session_state:
        st.session_state['light_state'] = 'green'
    if 'light_time_left' not in st.session_state:
        st.session_state['light_time_left'] = green_times[DIRECTIONS[0]]
    if 'simulation_running' not in st.session_state:
        st.session_state['simulation_running'] = False
    if 'active_direction' not in st.session_state:
        st.session_state['active_direction'] = DIRECTIONS[0]
    if 'cycle_idx' not in st.session_state:
        st.session_state['cycle_idx'] = 0

if st.session_state['history']:
    st.subheader("Traffic Light Simulation (4 Directions)")
    init_light_simulation()

    # Start/Stop buttons
    col_sim1, col_sim2 = st.columns(2)
    if col_sim1.button("Start Simulation"):
        st.session_state['simulation_running'] = True
    if col_sim2.button("Reset Simulation"):
        st.session_state['light_state'] = 'green'
        st.session_state['light_time_left'] = green_times[DIRECTIONS[0]]
        st.session_state['simulation_running'] = False
        st.session_state['active_direction'] = DIRECTIONS[0]
        st.session_state['cycle_idx'] = 0

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
                # Move to next direction
                st.session_state['cycle_idx'] = (st.session_state['cycle_idx'] + 1) % 4
                next_dir = DIRECTIONS[st.session_state['cycle_idx']]
                st.session_state['active_direction'] = next_dir
                st.session_state['light_state'] = 'green'
                st.session_state['light_time_left'] = green_times[next_dir]

    # Display the current light
    traffic_light_display(st.session_state['light_state'], st.session_state['active_direction'])
    st.write(f"Active Direction: {st.session_state['active_direction']}")
    st.write(f"Time left: {st.session_state['light_time_left']} seconds") 