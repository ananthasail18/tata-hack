import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="NASA C-MAPSS FD001 Visualizer",
    page_icon="✈️",
    layout="wide"
)

# Custom aero-dark styling
st.markdown("""
<style>
    .reportview-container { background: #0b0f19; }
    .metric-card {
        background: #151d30;
        border: 1px solid #23314f;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    col_names = ['unit', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
    df = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None, names=col_names)
    max_cycles = df.groupby('unit')['cycle'].transform('max')
    df['RUL'] = max_cycles - df['cycle']
    return df

df = load_data()

sensor_meta = {
    'sensor_2': {'name': 'T24', 'desc': 'LPC Outlet Temp', 'unit': '°R', 'dir': 'up'},
    'sensor_3': {'name': 'T30', 'desc': 'HPC Outlet Temp', 'unit': '°R', 'dir': 'up'},
    'sensor_4': {'name': 'T50', 'desc': 'LPT Outlet Temp (EGT)', 'unit': '°R', 'dir': 'up'},
    'sensor_7': {'name': 'P30', 'desc': 'HPC Outlet Total Pressure', 'unit': 'psia', 'dir': 'down'},
    'sensor_8': {'name': 'Nf', 'desc': 'Physical Fan Speed', 'unit': 'rpm', 'dir': 'up'},
    'sensor_9': {'name': 'Nc', 'desc': 'Physical Core Speed', 'unit': 'rpm', 'dir': 'up'},
    'sensor_11': {'name': 'Ps30', 'desc': 'HPC Static Pressure', 'unit': 'psia', 'dir': 'up'},
    'sensor_12': {'name': 'phi', 'desc': 'Fuel Flow to Ps30 Ratio', 'unit': 'pps/psia', 'dir': 'down'},
    'sensor_13': {'name': 'NRf', 'desc': 'Corrected Fan Speed', 'unit': 'rpm', 'dir': 'up'},
    'sensor_14': {'name': 'NRc', 'desc': 'Corrected Core Speed', 'unit': 'rpm', 'dir': 'up'},
    'sensor_15': {'name': 'BPR', 'desc': 'Bypass Ratio', 'unit': '-', 'dir': 'up'},
    'sensor_17': {'name': 'htBleed', 'desc': 'Bleed Enthalpy', 'unit': '-', 'dir': 'up'},
    'sensor_20': {'name': 'W31', 'desc': 'HPT Coolant Bleed', 'unit': 'lbm/s', 'dir': 'down'},
    'sensor_21': {'name': 'W32', 'desc': 'LPT Coolant Bleed', 'unit': 'lbm/s', 'dir': 'down'}
}
active_sensors = list(sensor_meta.keys())

st.title("🚀 NASA C-MAPSS FD001 Turbofan Degradation Visualizer")
st.caption("Multivariate Run-to-Failure Analysis • Sea Level Cruise • High-Pressure Compressor (HPC) Degradation Fault Mode")

# Top KPI metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Fleet Engines", f"{df['unit'].nunique()} Units")
c2.metric("Total Snapshots", f"{len(df):,} Cycles")
c3.metric("Average Lifespan", f"{df.groupby('unit')['cycle'].max().mean():.1f} Cycles")
c4.metric("Lifespan Range", f"{df.groupby('unit')['cycle'].max().min()} - {df.groupby('unit')['cycle'].max().max()} Cycles")

tab1, tab2, tab3 = st.tabs([
    "📈 Single Engine Drill-Down",
    "🌐 Fleet Trajectories (Aligned by RUL)",
    "⚡ Health Index / Normalized View"
])

with tab1:
    col_sel, col_smooth = st.columns([2, 2])
    with col_sel:
        selected_unit = st.selectbox("Select Engine Unit", options=sorted(df['unit'].unique()), index=0)
    with col_smooth:
        smooth_window = st.slider("Moving Average Smoothing Window", min_value=1, max_value=20, value=5)

    u_df = df[df['unit'] == selected_unit].sort_values('cycle')
    max_c = u_df['cycle'].max()
    st.info(f"Unit {selected_unit} operated for **{max_c} cycles** before catastrophic HPC failure.")

    # 14 Active Sensors Grid (7 rows x 2 cols)
    fig_grid = make_subplots(rows=7, cols=2, subplot_titles=[
        f"{s}: {sensor_meta[s]['name']} ({sensor_meta[s]['desc']})" for s in active_sensors
    ], vertical_spacing=0.06)

    for i, s in enumerate(active_sensors):
        row = (i // 2) + 1
        col = (i % 2) + 1
        raw_vals = u_df[s]
        smooth_vals = raw_vals.rolling(smooth_window, min_periods=1).mean()
        line_color = '#f43f5e' if sensor_meta[s]['dir'] == 'up' else '#38bdf8'

        fig_grid.add_trace(go.Scatter(
            x=u_df['cycle'], y=raw_vals, mode='markers',
            marker=dict(size=3, opacity=0.35, color=line_color),
            showlegend=False, hoverinfo='x+y'
        ), row=row, col=col)

        fig_grid.add_trace(go.Scatter(
            x=u_df['cycle'], y=smooth_vals, mode='lines',
            line=dict(width=2, color=line_color),
            showlegend=False, hoverinfo='x+y'
        ), row=row, col=col)

    fig_grid.update_layout(height=1600, template='plotly_dark', margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig_grid, use_container_width=True)

with tab2:
    col_s, col_dom = st.columns([3, 2])
    with col_s:
        chosen_sensor = st.selectbox("Select Sensor Parameter", options=active_sensors,
            format_func=lambda s: f"{s}: {sensor_meta[s]['name']} - {sensor_meta[s]['desc']} ({sensor_meta[s]['unit']})")
    with col_dom:
        time_mode = st.radio("X-Axis Domain", ["Remaining Useful Life (RUL countdown to 0)", "Operational Cycle (0 to failure)"])

    meta = sensor_meta[chosen_sensor]
    fig_fleet = go.Figure()

    sample_units = [1, 5, 10, 20, 30, 39, 50, 60, 69, 80, 90, 100]
    for u in sample_units:
        sub = df[df['unit'] == u].sort_values('cycle')
        x_val = sub['RUL'] if "RUL" in time_mode else sub['cycle']
        fig_fleet.add_trace(go.Scatter(
            x=x_val, y=sub[chosen_sensor], mode='lines',
            line=dict(width=1, color='rgba(148, 163, 184, 0.25)'),
            name=f"Unit {u}", hoverinfo='name+x+y', showlegend=False
        ))

    # Fleet mean
    if "RUL" in time_mode:
        fleet_mean = df.groupby('RUL')[chosen_sensor].mean().loc[lambda x: x.index <= 250]
        fig_fleet.add_trace(go.Scatter(
            x=fleet_mean.index, y=fleet_mean.values, mode='lines',
            line=dict(width=3.5, color='#38bdf8'),
            name="Fleet Mean (All 100 Units)"
        ))
        fig_fleet.update_xaxes(autorange="reversed", title="Remaining Useful Life (Cycles to Failure)")
        fig_fleet.add_vline(x=60, line_width=2, line_dash="dash", line_color="#f59e0b",
                            annotation_text="Degradation Knee Point (~60 RUL)", annotation_position="top left")
    else:
        fig_fleet.update_xaxes(title="Operational Cycle")

    fig_fleet.update_layout(
        height=550, template='plotly_dark',
        title=f"Fleet Trajectories for {chosen_sensor} ({meta['name']} - {meta['desc']})",
        yaxis_title=f"{meta['name']} ({meta['unit']})"
    )
    st.plotly_chart(fig_fleet, use_container_width=True)

with tab3:
    norm_u = st.selectbox("Select Engine for Normalized Health View", options=sorted(df['unit'].unique()), key="norm_u")
    invert_drop = st.checkbox("Align Direction (Invert sensors that decrease so 100% = Failure State)", value=True)

    u_norm = df[df['unit'] == norm_u].sort_values('cycle')
    fig_norm = go.Figure()

    key_sensors = ['sensor_4', 'sensor_11', 'sensor_12', 'sensor_7', 'sensor_15', 'sensor_21']
    for s in key_sensors:
        vals = u_norm[s].rolling(5, min_periods=1).mean()
        v_min, v_max = vals.min(), vals.max()
        normed = (vals - v_min) / (v_max - v_min if v_max > v_min else 1)
        name = f"{s} ({sensor_meta[s]['name']})"
        if invert_drop and sensor_meta[s]['dir'] == 'down':
            normed = 1 - normed
            name += " [Inverted]"

        fig_norm.add_trace(go.Scatter(
            x=u_norm['cycle'], y=normed * 100, mode='lines',
            line=dict(width=2), name=name
        ))

    fig_norm.update_layout(
        height=550, template='plotly_dark',
        title=f"Normalized Subsystem Health Trajectory (Unit {norm_u} - Lifespan: {u_norm['cycle'].max()} Cycles)",
        xaxis_title="Operational Cycle",
        yaxis_title="Degradation Severity Level (%) [0% = Normal, 100% = Failure]"
    )
    st.plotly_chart(fig_norm, use_container_width=True)
