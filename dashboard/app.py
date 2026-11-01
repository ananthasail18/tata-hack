"""
NASA C-MAPSS Predictive Maintenance Operations Center
Streamlit Interactive Dashboard (5 Pages)
Teliport Season 4, Case Study 1: Connected Fleet Predictive Maintenance
"""

from datetime import datetime
import json
import os
import sqlite3
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.data.ingestion import load_cmapss_data, load_test_rul
from src.pipeline import run_fleet_inference
from src.maintenance.work_orders import WorkOrderManager
from src.mlops.registry import ModelRegistry
from src.mlops.monitoring import PipelineMonitor
from src.mlops.retraining import run_retraining_workflow

# Streamlit Page Config
st.set_page_config(
    page_title="C-MAPSS Predictive Maintenance Center",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Cockpit Theme)
st.markdown("""
<style>
    .main { background-color: #0b0f19; }
    .stMetric {
        background-color: #151d30;
        border: 1px solid #23314f;
        padding: 10px 14px;
        border-radius: 8px;
    }
    .badge-critical {
        background-color: rgba(244, 63, 94, 0.2);
        color: #f43f5e;
        border: 1px solid #f43f5e;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-high {
        background-color: rgba(249, 115, 22, 0.2);
        color: #f97316;
        border: 1px solid #f97316;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-medium {
        background-color: rgba(234, 179, 8, 0.2);
        color: #eab308;
        border: 1px solid #eab308;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-low {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_cached_raw_data():
    train_df = load_cmapss_data(data_dir="data/raw", subset="FD001", split="train")
    test_df = load_cmapss_data(data_dir="data/raw", subset="FD001", split="test")
    test_ruls = load_test_rul(data_dir="data/raw", subset="FD001")
    return train_df, test_df, test_ruls

@st.cache_data
def get_cached_inference():
    _, test_df, _ = get_cached_raw_data()
    return run_fleet_inference(test_df)

train_raw, test_raw, test_ruls = get_cached_raw_data()
wo_mgr = WorkOrderManager()
registry = ModelRegistry()
champion_meta = registry.get_champion()

# Sidebar Navigation
st.sidebar.title("✈️ Fleet Maintenance")
st.sidebar.caption("NASA C-MAPSS FD001 • Teliport Season 4")

champ_ver = champion_meta["version"] if champion_meta else "v1.0.0"
st.sidebar.info(f"**Active Model:** {champ_ver} ({champion_meta['model_type'] if champion_meta else 'XGBoost'})")

page = st.sidebar.radio(
    "Navigation",
    [
        "1. Fleet Overview",
        "2. Engine Health Deep-Dive",
        "3. Sensor Analysis & Physics",
        "4. Maintenance Work Orders",
        "5. MLOps & Monitoring"
    ]
)

inference_res = get_cached_inference()
fleet_df = inference_res["fleet_summary_df"]
annotated_df = inference_res["annotated_cycles_df"]

# ==============================================================================
# PAGE 1: FLEET OVERVIEW
# ==============================================================================
if page == "1. Fleet Overview":
    st.title("🌐 Connected Fleet Overview")
    st.markdown("Real-time operational status across all 100 monitored turbofan engines.")

    counts = fleet_df["severity"].value_counts().to_dict()
    n_crit = counts.get("CRITICAL", 0)
    n_high = counts.get("HIGH", 0)
    n_med = counts.get("MEDIUM", 0)
    n_low = counts.get("LOW", 0)

    active_wos = wo_mgr.list_work_orders(status="OPEN") + wo_mgr.list_work_orders(status="IN_REVIEW")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Monitored Fleet", f"{len(fleet_df)} Units")
    col2.metric("Critical (Action Now)", f"{n_crit}", delta=f"-{n_crit}" if n_crit > 0 else "0", delta_color="inverse")
    col3.metric("High Priority", f"{n_high}")
    col4.metric("Medium (Inspection)", f"{n_med}")
    col5.metric("Nominal / Low", f"{n_low}")
    col6.metric("Active Work Orders", f"{len(active_wos)}")

    st.markdown("---")

    c_left, c_right = st.columns([1, 1])

    with c_left:
        # Severity Donut Chart
        sev_data = pd.DataFrame({
            "Severity": ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            "Count": [n_crit, n_high, n_med, n_low],
            "Color": ["#f43f5e", "#f97316", "#eab308", "#10b981"]
        })
        fig_donut = px.pie(
            sev_data, names="Severity", values="Count", color="Severity",
            color_discrete_map={"CRITICAL": "#f43f5e", "HIGH": "#f97316", "MEDIUM": "#eab308", "LOW": "#10b981"},
            hole=0.45, title="Fleet Health Severity Distribution"
        )
        fig_donut.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_donut, use_container_width=True)

    with c_right:
        # RUL Histogram
        fig_hist = px.histogram(
            fleet_df, x="predicted_rul", nbins=20,
            title="Distribution of Forecasted Remaining Useful Life (RUL)",
            color_discrete_sequence=["#38bdf8"]
        )
        fig_hist.add_vline(x=15, line_dash="dash", line_color="#f43f5e", annotation_text="Critical (<=15)")
        fig_hist.add_vline(x=40, line_dash="dash", line_color="#f97316", annotation_text="High (<=40)")
        fig_hist.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20), xaxis_title="Predicted RUL (Cycles)")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("📋 Fleet Engine Telemetry and Triage Roster")
    
    # Filter controls
    sev_filter = st.multiselect("Filter by Severity", options=["CRITICAL", "HIGH", "MEDIUM", "LOW"], default=["CRITICAL", "HIGH", "MEDIUM", "LOW"])
    filtered_roster = fleet_df[fleet_df["severity"].isin(sev_filter)].copy()
    
    st.dataframe(
        filtered_roster[[
            "unit", "current_cycle", "predicted_rul", "anomaly_score", "anomaly_status", "severity", "recommended_action"
        ]].sort_values(by="predicted_rul"),
        use_container_width=True,
        hide_index=True
    )

# ==============================================================================
# PAGE 2: ENGINE HEALTH DEEP-DIVE
# ==============================================================================
elif page == "2. Engine Health Deep-Dive":
    st.title("🔍 Engine Health Diagnostic Deep-Dive")
    
    # Engine Selection
    selected_unit = st.selectbox(
        "Select Engine ID to Inspect",
        options=sorted(fleet_df["unit"].unique()),
        index=0
    )

    engine_summary = fleet_df[fleet_df["unit"] == selected_unit].iloc[0]
    engine_cycles = annotated_df[annotated_df["unit"] == selected_unit].sort_values("cycle")
    engine_raw = test_raw[test_raw["unit"] == selected_unit].sort_values("cycle")

    # Engine Status Ribbon
    sev = engine_summary["severity"]
    sev_color = "#f43f5e" if sev == "CRITICAL" else ("#f97316" if sev == "HIGH" else ("#eab308" if sev == "MEDIUM" else "#10b981"))
    
    st.markdown(f"""
    <div style="background-color: #151d30; border-left: 5px solid {sev_color}; padding: 14px 18px; border-radius: 6px; margin-bottom: 20px;">
        <span style="font-size: 20px; font-weight: bold; color: white;">Unit {selected_unit} Diagnostic Report</span>
        <span style="float: right; font-weight: bold; color: {sev_color}; font-size: 16px;">SEVERITY: {sev}</span>
        <br/>
        <span style="color: #94a3b8; font-size: 14px;"><strong>Recommended Action:</strong> {engine_summary['recommended_action']}</span><br/>
        <span style="color: #94a3b8; font-size: 13px;"><strong>Diagnostic Justification:</strong> {engine_summary['reason']}</span>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Observed Cycles", f"{engine_summary['current_cycle']}")
    m2.metric("Forecasted RUL", f"{engine_summary['predicted_rul']:.1f} cycles")
    m3.metric("Anomaly Score", f"{engine_summary['anomaly_score']:.2f}")
    m4.metric("Anomaly Status", f"{engine_summary['anomaly_status']}")

    st.markdown("### Telemetry Trajectories & Anomaly Progression")

    # Dual Plot: RUL and Anomaly Score over Cycle
    fig_diag = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["Forecasted Remaining Useful Life (RUL)", "Isolation Forest Anomaly Score"])
    fig_diag.add_trace(go.Scatter(x=engine_cycles["cycle"], y=engine_cycles["predicted_rul"], mode="lines", name="Predicted RUL", line=dict(color="#38bdf8", width=2.5)), row=1, col=1)
    fig_diag.add_hline(y=15, line_dash="dash", line_color="#f43f5e", row=1, col=1)
    fig_diag.add_hline(y=40, line_dash="dash", line_color="#f97316", row=1, col=1)

    fig_diag.add_trace(go.Scatter(x=engine_cycles["cycle"], y=engine_cycles["anomaly_score"], mode="lines", name="Anomaly Score", line=dict(color="#f43f5e", width=2)), row=2, col=1)
    fig_diag.add_hline(y=0.55, line_dash="dash", line_color="#f59e0b", annotation_text="Anomaly Threshold (0.55)", row=2, col=1)

    fig_diag.update_layout(template="plotly_dark", height=450, margin=dict(l=40, r=20, t=40, b=30))
    st.plotly_chart(fig_diag, use_container_width=True)

    st.markdown("### Critical HPC Degradation Sensor Trajectories")
    
    sens_tabs = st.tabs(["S4: T50 LPT Temp", "S11: Ps30 Static Pressure", "S7: P30 Total Pressure", "S15: Bypass Ratio", "S3: T30 HPC Temp"])
    
    sensor_map = {
        "S4: T50 LPT Temp": ("sensor_4", "T50 LPT Outlet Temperature (°R)", 1425.0, "#f43f5e"),
        "S11: Ps30 Static Pressure": ("sensor_11", "Ps30 HPC Outlet Static Pressure (psia)", 48.0, "#f43f5e"),
        "S7: P30 Total Pressure": ("sensor_7", "P30 HPC Outlet Total Pressure (psia)", 551.5, "#38bdf8"),
        "S15: Bypass Ratio": ("sensor_15", "Bypass Ratio (BPR)", 8.50, "#f43f5e"),
        "S3: T30 HPC Temp": ("sensor_3", "T30 HPC Outlet Temperature (°R)", 1600.0, "#f43f5e")
    }

    for tab, (s_col, s_name, s_thresh, s_color) in zip(sens_tabs, sensor_map.values()):
        with tab:
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(x=engine_raw["cycle"], y=engine_raw[s_col], mode="markers+lines", name="Raw Telemetry", line=dict(color=s_color, width=1.5)))
            fig_s.add_hline(y=s_thresh, line_dash="dash", line_color="orange", annotation_text=f"Degradation Limit ({s_thresh})")
            fig_s.update_layout(template="plotly_dark", height=320, title=f"Unit {selected_unit} — {s_name}", xaxis_title="Cycle", yaxis_title=s_name)
            st.plotly_chart(fig_s, use_container_width=True)

# ==============================================================================
# PAGE 3: SENSOR ANALYSIS & PHYSICS
# ==============================================================================
elif page == "3. Sensor Analysis & Physics":
    st.title("🔬 Sensor Physics & Degradation Mechanics")
    st.markdown("""
    In subset **FD001**, the sole fault mode is **High-Pressure Compressor (HPC) aerodynamic degradation**.
    Compressor blade erosion and tip clearance widening reduce compression efficiency ($\Delta \eta_{HPC}$) 
    and flow capacity ($\Delta C_{HPC}$).
    """)

    c_c1, c_c2 = st.columns([1, 1])

    with c_c1:
        st.subheader("Sensor Correlation with Remaining Life")
        active_sensors = [
            "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9",
            "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
            "sensor_17", "sensor_20", "sensor_21"
        ]
        train_rul_df = train_raw.copy()
        train_rul_df["RUL"] = train_rul_df.groupby("unit")["cycle"].transform("max") - train_rul_df["cycle"]
        corrs = train_rul_df[active_sensors + ["RUL"]].corr()["RUL"].drop("RUL").sort_values()

        fig_corr = px.bar(
            x=corrs.values, y=corrs.index, orientation="h",
            color=corrs.values, color_continuous_scale="RdBu_r",
            labels={"x": "Pearson Correlation with RUL", "y": "Sensor"}
        )
        fig_corr.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig_corr, use_container_width=True)

    with c_c2:
        st.subheader("Physical Degradation Signatures")
        st.markdown("""
        * **Temperatures ($T_{50}, T_{30}, T_{24}$)**: **DRIFT UP ($\uparrow$)**  
          Compensating fuel flow injects more thermal energy into combustor and turbine stages to maintain target sea-level thrust.
        * **Total Pressure ($P_{30}$)**: **DRIFTS DOWN ($\downarrow$)**  
          Worn HPC blades fail to generate required dynamic compression ratio.
        * **Static Pressure ($Ps_{30}$)**: **DRIFTS UP ($\uparrow$)**  
          Core flow restriction lowers axial velocity, converting dynamic pressure into static backpressure.
        * **Bypass Ratio ($BPR$)**: **DRIFTS UP ($\uparrow$)**  
          Core swallow capacity is restricted, causing fan air to divert around the core into bypass.
        * **Coolant Bleed ($W_{31}, W_{32}$)**: **DRIFTS DOWN ($\downarrow$)**  
          Reduced core pressure gradient starves turbine cooling air passages.
        """)

# ==============================================================================
# PAGE 4: MAINTENANCE WORK ORDERS
# ==============================================================================
elif page == "4. Maintenance Work Orders":
    st.title("🛠️ Maintenance Work Order Management")
    st.markdown("Closed-loop work-order tracking with SQLite persistence and approval workflows.")

    all_wos = wo_mgr.list_work_orders()
    
    st.metric("Total Generated Work Orders", len(all_wos))

    if not all_wos:
        st.info("No work orders generated yet. Run inference on the fleet to generate orders.")
    else:
        wo_df = pd.DataFrame(all_wos)
        
        # Filter controls
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            stat_filter = st.multiselect("Filter by Status", options=["OPEN", "IN_REVIEW", "APPROVED", "COMPLETED", "CANCELLED"], default=["OPEN", "IN_REVIEW", "APPROVED"])
        with f_col2:
            sev_filter = st.multiselect("Filter by Severity", options=["CRITICAL", "HIGH", "MEDIUM"], default=["CRITICAL", "HIGH", "MEDIUM"])

        disp_df = wo_df[wo_df["status"].isin(stat_filter) & wo_df["severity"].isin(sev_filter)]
        st.dataframe(disp_df[["work_order_id", "engine_id", "severity", "predicted_rul", "status", "recommended_action", "created_at"]], use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Review & Update Work Order Status")
        
        selected_wo_id = st.selectbox("Select Work Order to Review", options=wo_df["work_order_id"].tolist())
        selected_wo = wo_mgr.get_work_order(selected_wo_id)

        if selected_wo:
            r1, r2 = st.columns([2, 1])
            with r1:
                st.markdown(f"**Work Order ID:** `{selected_wo['work_order_id']}`")
                st.markdown(f"**Target Engine:** Unit {selected_wo['engine_id']} | **Severity:** `{selected_wo['severity']}`")
                st.markdown(f"**Current Status:** `{selected_wo['status']}`")
                st.markdown(f"**Recommended Directive:** {selected_wo['recommended_action']}")
                st.markdown(f"**Engineering Justification:** {selected_wo['reason']}")
            with r2:
                new_status = st.selectbox("Update Status", options=["OPEN", "IN_REVIEW", "APPROVED", "COMPLETED", "CANCELLED"], index=["OPEN", "IN_REVIEW", "APPROVED", "COMPLETED", "CANCELLED"].index(selected_wo["status"]))
                reviewer = st.text_input("Reviewer Name", value=selected_wo.get("reviewed_by") or "Lead_Maintenance_Engineer")
                notes = st.text_area("Review Notes", value=selected_wo.get("reviewer_notes") or "")
                
                if st.button("Submit Status Update"):
                    wo_mgr.update_status(selected_wo_id, new_status, reviewed_by=reviewer, reviewer_notes=notes)
                    st.success(f"Work Order {selected_wo_id} updated to {new_status}!")
                    st.rerun()

# ==============================================================================
# PAGE 5: MLOPS & MONITORING
# ==============================================================================
elif page == "5. MLOps & Monitoring":
    st.title("⚙️ MLOps Lifecycle & Telemetry Drift Monitoring")

    st.subheader("Model Registry & Champion Status")
    all_models = registry.list_models()
    
    if all_models:
        reg_df = pd.DataFrame([
            {
                "Version": m["version"],
                "Model Type": m["model_type"],
                "Status": m["status"],
                "Registered At": m["registered_at"],
                "Test RMSE": m.get("metrics", {}).get("test_rmse"),
                "Test NASA Score": m.get("metrics", {}).get("test_nasa")
            }
            for m in all_models
        ])
        st.dataframe(reg_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Data Drift Detection (Kolmogorov-Smirnov Test)")
    monitor = PipelineMonitor(baseline_df=train_raw)
    drift_report = monitor.compute_data_drift(test_raw)
    
    d1, d2 = st.columns(2)
    d1.metric("Drifted Sensors", f"{drift_report.get('drifted_sensors_count', 0)} of {drift_report.get('total_sensors_checked', 14)}")
    d2.metric("Drift Severity Rate", f"{drift_report.get('drift_percentage', 0.0)}%")
    
    if st.checkbox("Show Sensor Drift Details"):
        st.json(drift_report.get("sensor_details", {}))

    st.markdown("---")
    st.subheader("Simulate Incoming Data & Retraining Flow")
    st.markdown("""
    Trigger automated retraining flow: simulates newly arrived run-to-failure engine logs,
    retrains candidate XGBoost model, checks validation acceptance gates, and conditionally promotes to Champion.
    """)

    if st.button("🚀 Trigger Automated Continuous Retraining Workflow"):
        with st.spinner("Retraining candidate model and checking acceptance gates..."):
            retrain_res = run_retraining_workflow()
            st.success(f"Retraining Workflow Completed! Candidate Version: {retrain_res['candidate_version']}")
            st.json(retrain_res)
            st.rerun()
