import os
import json
import numpy as np
import pandas as pd
import plotly.offline as po

def generate_interactive_dashboard():
    print("Loading train_FD001.txt...")
    col_names = ['unit', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
    df = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None, names=col_names)

    # Compute RUL for each engine
    max_cycles = df.groupby('unit')['cycle'].transform('max')
    df['RUL'] = max_cycles - df['cycle']

    active_sensors = [
        'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9',
        'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
        'sensor_17', 'sensor_20', 'sensor_21'
    ]

    sensor_meta = {
        'sensor_1': {'name': 'T2', 'desc': 'Total temperature at fan inlet', 'unit': '°R', 'active': False, 'system': 'Fan', 'dir': 'none', 'expl': 'Ambient/inlet air temp; constant in FD001 (sea level condition).'},
        'sensor_2': {'name': 'T24', 'desc': 'Total temperature at LPC outlet', 'unit': '°R', 'active': True, 'system': 'LPC', 'dir': 'up', 'expl': 'Increases as compressor efficiency drops, requiring more work and heat.'},
        'sensor_3': {'name': 'T30', 'desc': 'Total temperature at HPC outlet', 'unit': '°R', 'active': True, 'system': 'HPC', 'dir': 'up', 'expl': 'Direct indicator of HPC degradation; temperature rises due to aerodynamic losses.'},
        'sensor_4': {'name': 'T50', 'desc': 'Total temperature at LPT outlet', 'unit': '°R', 'active': True, 'system': 'LPT', 'dir': 'up', 'expl': 'Exhaust gas temp increases significantly as engine burns more fuel to maintain thrust.'},
        'sensor_5': {'name': 'P2', 'desc': 'Pressure at fan inlet', 'unit': 'psia', 'active': False, 'system': 'Fan', 'dir': 'none', 'expl': 'Constant ambient pressure at sea level in FD001.'},
        'sensor_6': {'name': 'P15', 'desc': 'Total pressure in bypass-duct', 'unit': 'psia', 'active': False, 'system': 'Bypass', 'dir': 'none', 'expl': 'Essentially invariant in FD001 (very low variance ~0.001).'},
        'sensor_7': {'name': 'P30', 'desc': 'Total pressure at HPC outlet', 'unit': 'psia', 'active': True, 'system': 'HPC', 'dir': 'down', 'expl': 'HPC blade wear reduces compression capability; pressure drops over life.'},
        'sensor_8': {'name': 'Nf', 'desc': 'Physical fan speed', 'unit': 'rpm', 'active': True, 'system': 'Fan', 'dir': 'up', 'expl': 'Controller increases fan speed to compensate for loss of core efficiency.'},
        'sensor_9': {'name': 'Nc', 'desc': 'Physical core speed', 'unit': 'rpm', 'active': True, 'system': 'Core', 'dir': 'up', 'expl': 'Core rotor speed rises to deliver requested thrust despite HPC blade wear.'},
        'sensor_10': {'name': 'epr', 'desc': 'Engine pressure ratio (P50/P2)', 'unit': '-', 'active': False, 'system': 'Turbine', 'dir': 'none', 'expl': 'Engine pressure ratio remains constant in single sea-level operating setting.'},
        'sensor_11': {'name': 'Ps30', 'desc': 'Static pressure at HPC outlet', 'unit': 'psia', 'active': True, 'system': 'HPC', 'dir': 'up', 'expl': 'Static pressure increases as flow velocity drops and backpressure builds at HPC outlet.'},
        'sensor_12': {'name': 'phi', 'desc': 'Ratio of fuel flow to Ps30', 'unit': 'pps/psia', 'active': True, 'system': 'Fuel', 'dir': 'down', 'expl': 'Fuel-to-static-pressure ratio drops as Ps30 increases and fuel metering compensates.'},
        'sensor_13': {'name': 'NRf', 'desc': 'Corrected fan speed', 'unit': 'rpm', 'active': True, 'system': 'Fan', 'dir': 'up', 'expl': 'Temperature-corrected fan speed drifts upward with degradation.'},
        'sensor_14': {'name': 'NRc', 'desc': 'Corrected core speed', 'unit': 'rpm', 'active': True, 'system': 'Core', 'dir': 'up', 'expl': 'Corrected core speed drifts upward to maintain commanded engine thrust.'},
        'sensor_15': {'name': 'BPR', 'desc': 'Bypass Ratio', 'unit': '-', 'active': True, 'system': 'Bypass', 'dir': 'up', 'expl': 'Core flow capacity drops, causing higher proportion of air to divert through bypass.'},
        'sensor_16': {'name': 'farB', 'desc': 'Burner fuel-air ratio', 'unit': '-', 'active': False, 'system': 'Combustor', 'dir': 'none', 'expl': 'Constant setpoint in FD001 (0.03).'},
        'sensor_17': {'name': 'htBleed', 'desc': 'Bleed Enthalpy', 'unit': '-', 'active': True, 'system': 'Bleed', 'dir': 'up', 'expl': 'Enthalpy of compressor bleed air rises due to higher core stage temperatures.'},
        'sensor_18': {'name': 'Nf_dmd', 'desc': 'Demanded fan speed', 'unit': 'rpm', 'active': False, 'system': 'Controller', 'dir': 'none', 'expl': 'Commanded fan speed setpoint is constant (2388 rpm) in sea-level cruise.'},
        'sensor_19': {'name': 'PCNfR_dmd', 'desc': 'Demanded corrected fan speed', 'unit': 'rpm', 'active': False, 'system': 'Controller', 'dir': 'none', 'expl': 'Commanded corrected fan speed is constant (100.0 rpm) in FD001.'},
        'sensor_20': {'name': 'W31', 'desc': 'HPT coolant bleed', 'unit': 'lbm/s', 'active': True, 'system': 'Cooling', 'dir': 'down', 'expl': 'High-pressure turbine cooling bleed flow decreases as supply pressures degrade.'},
        'sensor_21': {'name': 'W32', 'desc': 'LPT coolant bleed', 'unit': 'lbm/s', 'active': True, 'system': 'Cooling', 'dir': 'down', 'expl': 'Low-pressure turbine cooling bleed flow drops as engine ages.'}
    }

    # Compute correlation with RUL
    corrs = df[active_sensors + ['RUL']].corr()['RUL'].drop('RUL')
    for s in active_sensors:
        sensor_meta[s]['corr'] = round(float(corrs[s]), 4)

    # Per-engine raw data (smooth is computed on the fly in JS to save ~3MB)
    engines_data = {}
    for u, grp in df.groupby('unit'):
        grp_sorted = grp.sort_values('cycle')
        engine_dict = {
            'cycle': grp_sorted['cycle'].tolist(),
            'rul': grp_sorted['RUL'].tolist(),
            'max_cycle': int(grp_sorted['cycle'].max()),
            'sensors': {}
        }
        for s in active_sensors:
            engine_dict['sensors'][s] = grp_sorted[s].round(3).tolist()
        engines_data[int(u)] = engine_dict

    # Fleet statistics across RUL (0 to 250)
    fleet_stats = {}
    max_rul_eval = 250
    rul_df = df[df['RUL'] <= max_rul_eval]
    for s in active_sensors:
        grp_rul = rul_df.groupby('RUL')[s].agg(['mean', 'std', 'count', 'min', 'max'])
        fleet_stats[s] = {
            'rul': grp_rul.index.tolist(),
            'mean': grp_rul['mean'].round(3).tolist(),
            'std': grp_rul['std'].round(3).tolist(),
            'min': grp_rul['min'].round(3).tolist(),
            'max': grp_rul['max'].round(3).tolist()
        }

    # Sample of engines for fleet overlay traces (25 diverse engines)
    sample_engine_ids = [1, 5, 10, 15, 20, 25, 30, 35, 39, 40, 45, 50, 55, 60, 65, 69, 70, 75, 80, 85, 90, 95, 98, 99, 100]
    sample_fleet_traces = {}
    for s in active_sensors:
        sample_fleet_traces[s] = []
        for u in sample_engine_ids:
            u_df = df[df['unit'] == u].sort_values('RUL', ascending=False)
            sample_fleet_traces[s].append({
                'unit': int(u),
                'rul': u_df['RUL'].tolist(),
                'cycle': u_df['cycle'].tolist(),
                'val': u_df[s].round(3).tolist()
            })

    # Summary metrics
    lifespans = df.groupby('unit')['cycle'].max()
    dataset_summary = {
        'total_units': int(len(lifespans)),
        'total_cycles': int(len(df)),
        'mean_life': round(float(lifespans.mean()), 1),
        'min_life': int(lifespans.min()),
        'min_unit': int(lifespans.idxmin()),
        'max_life': int(lifespans.max()),
        'max_unit': int(lifespans.idxmax()),
        'active_sensors_count': len(active_sensors),
        'invariant_sensors_count': 7
    }

    # Correlation bar chart data
    sorted_corrs = sorted([
        {'sensor': s, 'name': sensor_meta[s]['name'], 'corr': sensor_meta[s]['corr'], 'desc': sensor_meta[s]['desc']}
        for s in active_sensors
    ], key=lambda x: x['corr'])

    print("Extracting offline Plotly.js bundle...")
    plotly_js = po.get_plotlyjs()

    # Build HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NASA C-MAPSS FD001 Turbofan Time Series Visualizer</title>
    <style>
        :root {{
            --bg-main: #0b0f19;
            --bg-card: #151d30;
            --bg-card-hover: #1c263f;
            --border-color: #23314f;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent-cyan: #38bdf8;
            --accent-blue: #3b82f6;
            --accent-rose: #f43f5e;
            --accent-amber: #f59e0b;
            --accent-emerald: #10b981;
            --accent-purple: #a855f7;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }}

        body {{
            background-color: var(--bg-main);
            color: var(--text-primary);
            padding: 20px;
            min-height: 100vh;
        }}

        .dashboard-container {{
            max-width: 1560px;
            margin: 0 auto;
        }}

        /* Header */
        header {{
            background: linear-gradient(135deg, #151d30 0%, #0e1526 100%);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        }}

        .header-title h1 {{
            font-size: 24px;
            font-weight: 700;
            letter-spacing: -0.5px;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .header-title p {{
            font-size: 14px;
            color: var(--text-secondary);
            margin-top: 6px;
        }}

        .badge {{
            display: inline-block;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            padding: 3px 8px;
            border-radius: 999px;
            background: rgba(56, 189, 248, 0.15);
            color: var(--accent-cyan);
            border: 1px solid rgba(56, 189, 248, 0.3);
        }}

        /* KPI stats row */
        .kpi-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 14px;
            margin-bottom: 20px;
        }}

        .kpi-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 16px;
            position: relative;
            overflow: hidden;
        }}

        .kpi-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: var(--accent-cyan);
        }}

        .kpi-card.rose::before {{ background: var(--accent-rose); }}
        .kpi-card.amber::before {{ background: var(--accent-amber); }}
        .kpi-card.emerald::before {{ background: var(--accent-emerald); }}
        .kpi-card.purple::before {{ background: var(--accent-purple); }}

        .kpi-label {{
            font-size: 12px;
            color: var(--text-muted);
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.5px;
        }}

        .kpi-val {{
            font-size: 22px;
            font-weight: 700;
            color: #ffffff;
            margin: 6px 0 2px 0;
        }}

        .kpi-sub {{
            font-size: 12px;
            color: var(--text-secondary);
        }}

        /* Navigation Tabs */
        .tab-bar {{
            display: flex;
            gap: 8px;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 20px;
            overflow-x: auto;
            padding-bottom: 4px;
        }}

        .tab-btn {{
            background: transparent;
            border: 1px solid transparent;
            color: var(--text-secondary);
            font-size: 14px;
            font-weight: 600;
            padding: 10px 18px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
            white-space: nowrap;
        }}

        .tab-btn:hover {{
            color: var(--text-primary);
            background: rgba(255, 255, 255, 0.04);
        }}

        .tab-btn.active {{
            background: rgba(56, 189, 248, 0.12);
            color: var(--accent-cyan);
            border-color: rgba(56, 189, 248, 0.4);
        }}

        /* Tab Content */
        .tab-pane {{
            display: none;
        }}
        .tab-pane.active {{
            display: block;
        }}

        /* Controls card */
        .controls-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 18px;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
        }}

        .controls-group {{
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
        }}

        .control-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .control-label {{
            font-size: 13px;
            font-weight: 600;
            color: var(--text-secondary);
        }}

        select, input[type="text"] {{
            background: #0d1322;
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 7px 12px;
            border-radius: 6px;
            font-size: 13px;
            outline: none;
        }}

        select:focus {{
            border-color: var(--accent-cyan);
        }}

        .quick-btn {{
            background: #1c263f;
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            font-size: 12px;
            font-weight: 600;
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .quick-btn:hover {{
            background: var(--border-color);
            color: #ffffff;
        }}

        .quick-btn.active {{
            background: var(--accent-blue);
            color: white;
            border-color: var(--accent-cyan);
        }}

        .toggle-switch {{
            display: inline-flex;
            align-items: center;
            cursor: pointer;
            gap: 8px;
            font-size: 13px;
            color: var(--text-secondary);
        }}

        /* Grid for sensor subplots */
        .sensor-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
            gap: 16px;
            margin-bottom: 20px;
        }}

        .chart-box {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 14px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        }}

        .chart-header {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            padding: 4px 6px 10px 6px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            margin-bottom: 8px;
        }}

        .chart-title {{
            font-size: 14px;
            font-weight: 600;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .chart-subtitle {{
            font-size: 12px;
            color: var(--text-muted);
        }}

        .chart-container {{
            width: 100%;
            height: 270px;
            min-height: 270px;
        }}

        .full-chart-container {{
            width: 100%;
            height: 520px;
            min-height: 520px;
        }}

        /* Table styles */
        .table-container {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            overflow-x: auto;
            margin-bottom: 24px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            text-align: left;
        }}

        th {{
            background: #111827;
            padding: 12px 16px;
            color: var(--text-secondary);
            font-weight: 600;
            border-bottom: 1px solid var(--border-color);
            white-space: nowrap;
        }}

        td {{
            padding: 12px 16px;
            border-bottom: 1px solid rgba(35, 49, 79, 0.5);
            color: var(--text-primary);
        }}

        tr:hover td {{
            background: rgba(255, 255, 255, 0.02);
        }}

        .trend-up {{
            color: var(--accent-rose);
            font-weight: 600;
        }}

        .trend-down {{
            color: var(--accent-cyan);
            font-weight: 600;
        }}

        .trend-flat {{
            color: var(--text-muted);
        }}

        /* Callout alert boxes */
        .info-callout {{
            background: rgba(56, 189, 248, 0.08);
            border-left: 4px solid var(--accent-cyan);
            border-radius: 0 8px 8px 0;
            padding: 14px 18px;
            margin-bottom: 20px;
            font-size: 13px;
            line-height: 1.6;
            color: var(--text-secondary);
        }}

        .info-callout strong {{
            color: var(--accent-cyan);
        }}

        /* Responsive adjustments */
        @media (max-width: 768px) {{
            .sensor-grid {{
                grid-template-columns: 1fr;
            }}
            .controls-card {{
                flex-direction: column;
                align-items: stretch;
            }}
        }}
    </style>

    <!-- Embedded Offline Plotly.js (100% Self-Contained, zero network requests) -->
    <script>
{plotly_js}
    </script>
</head>
<body>

<div class="dashboard-container">
    <!-- Header -->
    <header>
        <div class="header-title">
            <h1>
                🚀 NASA C-MAPSS FD001 Turbofan Engine Telemetry
                <span class="badge">Run-to-Failure Analysis</span>
            </h1>
            <p>Multivariate Time Series Degradation Analysis &bull; Sea Level Cruise &bull; HPC Degradation Fault Mode</p>
        </div>
        <div style="text-align: right;">
            <span class="badge" style="background: rgba(16, 185, 129, 0.15); color: var(--accent-emerald); border-color: rgba(16, 185, 129, 0.3);">
                Dataset: train_FD001.txt (PHM 2008 Benchmark)
            </span>
        </div>
    </header>

    <!-- KPI Summary Row -->
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-label">Fleet Engines</div>
            <div class="kpi-val">{dataset_summary['total_units']} Units</div>
            <div class="kpi-sub">All run until catastrophic failure</div>
        </div>
        <div class="kpi-card emerald">
            <div class="kpi-label">Total Time Snapshots</div>
            <div class="kpi-val">{dataset_summary['total_cycles']:,} Cycles</div>
            <div class="kpi-sub">Multivariate sensor snapshots</div>
        </div>
        <div class="kpi-card amber">
            <div class="kpi-label">Mean Lifespan</div>
            <div class="kpi-val">{dataset_summary['mean_life']} Cycles</div>
            <div class="kpi-sub">Median: {lifespans.median():.0f} &bull; Std: {lifespans.std():.1f}</div>
        </div>
        <div class="kpi-card rose">
            <div class="kpi-label">Lifespan Range</div>
            <div class="kpi-val">{dataset_summary['min_life']} - {dataset_summary['max_life']} Cycles</div>
            <div class="kpi-sub">Min: Unit {dataset_summary['min_unit']} &bull; Max: Unit {dataset_summary['max_unit']}</div>
        </div>
        <div class="kpi-card purple">
            <div class="kpi-label">Active Sensors</div>
            <div class="kpi-val">14 of 21</div>
            <div class="kpi-sub">7 sensors are flat / invariant in FD001</div>
        </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="tab-bar">
        <button class="tab-btn active" onclick="switchTab('single')">📈 Single Engine Run-to-Failure</button>
        <button class="tab-btn" onclick="switchTab('fleet')">🌐 Fleet Trajectories (Aligned by RUL)</button>
        <button class="tab-btn" onclick="switchTab('normalized')">⚡ Multi-Sensor Degradation (Health Index)</button>
        <button class="tab-btn" onclick="switchTab('sensors')">📚 Sensor Reference & Correlation</button>
    </div>

    <!-- TAB 1: SINGLE ENGINE RUN-TO-FAILURE -->
    <div id="pane-single" class="tab-pane active">
        <div class="controls-card">
            <div class="controls-group">
                <div class="control-item">
                    <span class="control-label">Select Engine:</span>
                    <select id="single-engine-select" onchange="onSingleEngineChange(this.value)">
                        <!-- Populated by JS -->
                    </select>
                </div>
                <div class="control-item">
                    <span class="control-label">Quick Jump:</span>
                    <button class="quick-btn active" id="btn-u1" onclick="selectEngine(1)">Unit 1 (Standard: 192 cyc)</button>
                    <button class="quick-btn" id="btn-u39" onclick="selectEngine(39)">Unit 39 (Shortest: 128 cyc)</button>
                    <button class="quick-btn" id="btn-u69" onclick="selectEngine(69)">Unit 69 (Longest: 362 cyc)</button>
                    <button class="quick-btn" id="btn-u100" onclick="selectEngine(100)">Unit 100 (Final: 200 cyc)</button>
                </div>
            </div>

            <div class="controls-group">
                <label class="toggle-switch">
                    <input type="checkbox" id="single-smooth-toggle" checked onchange="updateSingleCharts()">
                    <span>Show 5-Cycle Smoothed Trendline</span>
                </label>
                <label class="toggle-switch">
                    <input type="checkbox" id="single-raw-toggle" checked onchange="updateSingleCharts()">
                    <span>Show Raw Data Points</span>
                </label>
            </div>
        </div>

        <div class="info-callout" id="single-engine-info">
            <!-- Dynamic info about current engine -->
        </div>

        <!-- 14 Active Sensors Grid -->
        <div class="sensor-grid" id="single-sensor-grid">
            <!-- Populated by JS -->
        </div>
    </div>

    <!-- TAB 2: FLEET TRAJECTORIES (ALIGNED BY RUL) -->
    <div id="pane-fleet" class="tab-pane">
        <div class="controls-card">
            <div class="controls-group">
                <div class="control-item">
                    <span class="control-label">Select Sensor Telemetry:</span>
                    <select id="fleet-sensor-select" onchange="renderFleetChart()">
                        <!-- Populated by JS -->
                    </select>
                </div>
                <div class="control-item">
                    <span class="control-label">Time Domain:</span>
                    <select id="fleet-x-domain" onchange="renderFleetChart()">
                        <option value="rul" selected>Remaining Useful Life (RUL countdown: Failure at 0)</option>
                        <option value="cycle">Operational Cycle (0 to Max Lifespan)</option>
                    </select>
                </div>
            </div>

            <div class="controls-group">
                <label class="toggle-switch">
                    <input type="checkbox" id="fleet-show-individual" checked onchange="renderFleetChart()">
                    <span>Show Engine Sample Traces (25 Units)</span>
                </label>
                <label class="toggle-switch">
                    <input type="checkbox" id="fleet-show-mean" checked onchange="renderFleetChart()">
                    <span>Show Fleet Mean</span>
                </label>
                <label class="toggle-switch">
                    <input type="checkbox" id="fleet-show-band" checked onchange="renderFleetChart()">
                    <span>Show ±1σ Variance Band</span>
                </label>
            </div>
        </div>

        <div class="info-callout">
            <strong>Degradation Alignment Phenomenon:</strong> When engines are aligned by <em>Remaining Useful Life (RUL)</em>, individual manufacturing variations and operational noise fade into a clear, unified physical degradation profile. Notice the <strong>knee point around RUL ≈ 50–70 cycles</strong>, where HPC aerodynamic wear sharply escalates until system failure at RUL = 0.
        </div>

        <div class="chart-box" style="margin-bottom: 20px;">
            <div class="chart-header">
                <div>
                    <span class="chart-title" id="fleet-chart-title">Fleet Degradation Profile</span>
                    <span class="chart-subtitle" id="fleet-chart-subtitle">Across 100 run-to-failure engines</span>
                </div>
            </div>
            <div id="fleet-plot-container" class="full-chart-container"></div>
        </div>
    </div>

    <!-- TAB 3: NORMALIZED MULTI-SENSOR HEALTH INDEX -->
    <div id="pane-normalized" class="tab-pane">
        <div class="controls-card">
            <div class="controls-group">
                <div class="control-item">
                    <span class="control-label">Select Engine:</span>
                    <select id="norm-engine-select" onchange="renderNormalizedChart()">
                        <!-- Populated by JS -->
                    </select>
                </div>
                <div class="control-item">
                    <span class="control-label">Sensor Grouping:</span>
                    <select id="norm-sensor-group" onchange="renderNormalizedChart()">
                        <option value="top_degrading" selected>Top Degrading Sensors (T50, Ps30, phi, P30, BPR, W32)</option>
                        <option value="temperatures">Temperatures Only (T24, T30, T50)</option>
                        <option value="pressures">Pressures & Speeds (P30, Ps30, Nf, Nc, NRc)</option>
                        <option value="all_active">All 14 Active Sensors</option>
                    </select>
                </div>
            </div>

            <div class="controls-group">
                <label class="toggle-switch">
                    <input type="checkbox" id="norm-align-direction" checked onchange="renderNormalizedChart()">
                    <span>Invert Dropping Sensors (Unified "Degradation Index" 0% -> 100%)</span>
                </label>
            </div>
        </div>

        <div class="info-callout">
            <strong>Unified Degradation Index:</strong> By min-max normalizing sensor signals to the [0%, 100%] interval, we can observe how multiple subsystems (compressor discharge, turbine exhaust, fan speed, and bypass) drift concurrently as mechanical health deteriorates.
        </div>

        <div class="chart-box">
            <div class="chart-header">
                <div>
                    <span class="chart-title" id="norm-chart-title">Normalized Sensor Health Divergence</span>
                    <span class="chart-subtitle">Observation of concurrent subsystem degradation</span>
                </div>
            </div>
            <div id="norm-plot-container" class="full-chart-container"></div>
        </div>
    </div>

    <!-- TAB 4: SENSOR REFERENCE & CORRELATION -->
    <div id="pane-sensors" class="tab-pane">
        <div class="chart-box" style="margin-bottom: 24px;">
            <div class="chart-header">
                <div>
                    <span class="chart-title">Sensor Correlation with Remaining Useful Life (RUL)</span>
                    <span class="chart-subtitle">Pearson correlation coefficient (r) across all 20,631 operational cycles</span>
                </div>
            </div>
            <div id="corr-plot-container" style="width: 100%; height: 420px; min-height: 420px;"></div>
        </div>

        <h3 style="font-size: 16px; margin-bottom: 12px; color: #ffffff;">Complete 21-Sensor Specification & Physical Mechanism</h3>
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Sensor ID</th>
                        <th>Symbol</th>
                        <th>Subsystem</th>
                        <th>Description</th>
                        <th>Units</th>
                        <th>FD001 Status</th>
                        <th>Degradation Trend</th>
                        <th>Corr (RUL)</th>
                        <th>Physical Mechanism / Explanation</th>
                    </tr>
                </thead>
                <tbody id="sensor-table-body">
                    <!-- Populated by JS -->
                </tbody>
            </table>
        </div>
    </div>
</div>

<script>
// Embedded Data Structures
const ENGINES_DATA = {json.dumps(engines_data)};
const FLEET_STATS = {json.dumps(fleet_stats)};
const SAMPLE_TRACES = {json.dumps(sample_fleet_traces)};
const SENSOR_META = {json.dumps(sensor_meta)};
const ACTIVE_SENSORS = {json.dumps(active_sensors)};
const CORR_DATA = {json.dumps(sorted_corrs)};

// Utility: moving average smoothing
function movingAverage(arr, windowSize = 5) {{
    const result = [];
    for (let i = 0; i < arr.length; i++) {{
        const start = Math.max(0, i - windowSize + 1);
        let sum = 0;
        let count = 0;
        for (let j = start; j <= i; j++) {{
            sum += arr[j];
            count++;
        }}
        result.push(Number((sum / count).toFixed(3)));
    }}
    return result;
}}

let currentSingleUnit = 1;

// Tab switcher
function switchTab(tabId) {{
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));

    const btn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.getAttribute('onclick').includes(tabId));
    if (btn) btn.classList.add('active');

    const pane = document.getElementById('pane-' + tabId);
    if (pane) pane.classList.add('active');

    // Re-render / resize plot when tab becomes visible
    setTimeout(() => {{
        if (tabId === 'fleet') {{
            renderFleetChart();
        }} else if (tabId === 'normalized') {{
            renderNormalizedChart();
        }} else if (tabId === 'sensors') {{
            renderCorrChart();
        }} else if (tabId === 'single') {{
            window.dispatchEvent(new Event('resize'));
        }}
    }}, 50);
}}

// Initialize Dropdowns
function initControls() {{
    const singleSelect = document.getElementById('single-engine-select');
    const normSelect = document.getElementById('norm-engine-select');
    singleSelect.innerHTML = '';
    normSelect.innerHTML = '';

    for (let u = 1; u <= 100; u++) {{
        const maxC = ENGINES_DATA[u].max_cycle;
        const opt = document.createElement('option');
        opt.value = u;
        opt.textContent = `Unit ${{u}} (Lifespan: ${{maxC}} cycles)`;
        singleSelect.appendChild(opt);

        const opt2 = document.createElement('option');
        opt2.value = u;
        opt2.textContent = `Unit ${{u}} (Lifespan: ${{maxC}} cycles)`;
        normSelect.appendChild(opt2);
    }}

    const fleetSensorSelect = document.getElementById('fleet-sensor-select');
    fleetSensorSelect.innerHTML = '';
    ACTIVE_SENSORS.forEach(s => {{
        const meta = SENSOR_META[s];
        const opt = document.createElement('option');
        opt.value = s;
        opt.textContent = `${{s}}: ${{meta.name}} - ${{meta.desc}} (${{meta.unit}}) [corr: ${{meta.corr}}]`;
        if (s === 'sensor_4') opt.selected = true; // Default to T50
        fleetSensorSelect.appendChild(opt);
    }});
}}

// Select engine via quick button
function selectEngine(unitId) {{
    document.querySelectorAll('.quick-btn').forEach(b => b.classList.remove('active'));
    const btn = document.getElementById('btn-u' + unitId);
    if (btn) btn.classList.add('active');

    document.getElementById('single-engine-select').value = unitId;
    currentSingleUnit = unitId;
    renderSingleEngineGrid();
}}

function onSingleEngineChange(val) {{
    currentSingleUnit = parseInt(val);
    document.querySelectorAll('.quick-btn').forEach(b => b.classList.remove('active'));
    const btn = document.getElementById('btn-u' + currentSingleUnit);
    if (btn) btn.classList.add('active');
    renderSingleEngineGrid();
}}

// Render 14-sensor grid for single engine
function renderSingleEngineGrid() {{
    const engine = ENGINES_DATA[currentSingleUnit];
    const maxCycle = engine.max_cycle;

    const infoDiv = document.getElementById('single-engine-info');
    infoDiv.innerHTML = `
        <strong>Unit ${{currentSingleUnit}} Profile:</strong> Total operational lifespan is <strong>${{maxCycle}} cycles</strong> before catastrophic failure.
        Early baseline operational health is represented in cycles 1–30. HPC degradation symptoms become pronounced beyond cycle ${{Math.max(1, maxCycle - 60)}}.
    `;

    const grid = document.getElementById('single-sensor-grid');
    grid.innerHTML = '';

    const showSmooth = document.getElementById('single-smooth-toggle').checked;
    const showRaw = document.getElementById('single-raw-toggle').checked;

    ACTIVE_SENSORS.forEach(s => {{
        const meta = SENSOR_META[s];
        const box = document.createElement('div');
        box.className = 'chart-box';

        const chartId = `chart-single-${{s}}`;
        box.innerHTML = `
            <div class="chart-header">
                <div>
                    <span class="chart-title">
                        ${{s}} &bull; ${{meta.name}}
                        <span style="font-size:11px; padding: 2px 6px; border-radius: 4px; background: rgba(255,255,255,0.06);">
                            ${{meta.unit}}
                        </span>
                    </span>
                    <span class="chart-subtitle">${{meta.desc}}</span>
                </div>
                <div style="font-size:12px; font-weight:600; color: ${{meta.dir === 'up' ? 'var(--accent-rose)' : 'var(--accent-cyan)'}}">
                    ${{meta.dir === 'up' ? '▲ Drift Up' : '▼ Drift Down'}}
                </div>
            </div>
            <div id="${{chartId}}" class="chart-container"></div>
        `;
        grid.appendChild(box);

        const traces = [];
        const rawVals = engine.sensors[s];
        const smoothVals = movingAverage(rawVals, 5);

        if (showRaw) {{
            traces.push({{
                x: engine.cycle,
                y: rawVals,
                mode: 'markers',
                name: 'Raw',
                marker: {{
                    size: 3.5,
                    color: meta.dir === 'up' ? '#f43f5e' : '#38bdf8',
                    opacity: 0.4
                }},
                hoverinfo: 'x+y'
            }});
        }}

        if (showSmooth) {{
            traces.push({{
                x: engine.cycle,
                y: smoothVals,
                mode: 'lines',
                name: '5-Cycle Smooth',
                line: {{
                    color: meta.dir === 'up' ? '#fb7185' : '#38bdf8',
                    width: 2.2
                }},
                hoverinfo: 'x+y'
            }});
        }}

        // Failure threshold marker
        const lastVal = smoothVals[smoothVals.length - 1];
        traces.push({{
            x: [maxCycle],
            y: [lastVal],
            mode: 'markers',
            name: 'Failure',
            marker: {{
                size: 9,
                color: '#f43f5e',
                symbol: 'x'
            }}
        }});

        const layout = {{
            height: 270,
            margin: {{ l: 45, r: 15, t: 15, b: 35 }},
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: {{ color: '#94a3b8', size: 10 }},
            xaxis: {{
                title: 'Operational Cycle',
                gridcolor: '#1e293b',
                zeroline: false,
                range: [0, maxCycle + 5]
            }},
            yaxis: {{
                gridcolor: '#1e293b',
                zeroline: false,
                tickformat: '.2f'
            }},
            showlegend: false,
            shapes: [
                {{
                    type: 'line',
                    x0: maxCycle,
                    x1: maxCycle,
                    y0: 0,
                    y1: 1,
                    yref: 'paper',
                    line: {{
                        color: 'rgba(244, 63, 94, 0.4)',
                        width: 1.5,
                        dash: 'dot'
                    }}
                }}
            ]
        }};

        Plotly.newPlot(chartId, traces, layout, {{ responsive: true, displayModeBar: false }});
    }});
}}

function updateSingleCharts() {{
    renderSingleEngineGrid();
}}

// TAB 2: Fleet Chart
function renderFleetChart() {{
    const s = document.getElementById('fleet-sensor-select').value;
    const xDomain = document.getElementById('fleet-x-domain').value;
    const showIndiv = document.getElementById('fleet-show-individual').checked;
    const showMean = document.getElementById('fleet-show-mean').checked;
    const showBand = document.getElementById('fleet-show-band').checked;

    const meta = SENSOR_META[s];
    const fleet = FLEET_STATS[s];
    const traces = [];

    document.getElementById('fleet-chart-title').innerHTML = `Fleet Telemetry: ${{s}} (${{meta.name}} - ${{meta.desc}})`;
    document.getElementById('fleet-chart-subtitle').textContent = `Plotted against ${{xDomain === 'rul' ? 'Remaining Useful Life (Cycles to Failure)' : 'Operational Cycle'}} across 100 engines`;

    if (xDomain === 'rul') {{
        // Plot against RUL
        if (showIndiv && SAMPLE_TRACES[s]) {{
            SAMPLE_TRACES[s].forEach(tr => {{
                traces.push({{
                    x: tr.rul,
                    y: tr.val,
                    mode: 'lines',
                    line: {{ color: 'rgba(148, 163, 184, 0.15)', width: 1 }},
                    hoverinfo: 'skip',
                    showlegend: false
                }});
            }});
        }}

        if (showBand) {{
            const upper = fleet.mean.map((m, i) => m + fleet.std[i]);
            const lower = fleet.mean.map((m, i) => m - fleet.std[i]);
            traces.push({{
                x: fleet.rul.concat(fleet.rul.slice().reverse()),
                y: upper.concat(lower.slice().reverse()),
                fill: 'toself',
                fillcolor: 'rgba(56, 189, 248, 0.12)',
                line: {{ color: 'transparent' }},
                name: '±1σ Variance Band',
                hoverinfo: 'skip'
            }});
        }}

        if (showMean) {{
            traces.push({{
                x: fleet.rul,
                y: fleet.mean,
                mode: 'lines',
                line: {{ color: '#38bdf8', width: 3 }},
                name: 'Fleet Mean (100 Engines)',
                hoverinfo: 'x+y'
            }});
        }}
    }} else {{
        // Plot against Cycle
        if (showIndiv && SAMPLE_TRACES[s]) {{
            SAMPLE_TRACES[s].forEach(tr => {{
                traces.push({{
                    x: tr.cycle,
                    y: tr.val,
                    mode: 'lines',
                    line: {{ color: 'rgba(148, 163, 184, 0.25)', width: 1.2 }},
                    name: `Unit ${{tr.unit}}`,
                    hoverinfo: 'name+x+y'
                }});
            }});
        }}
    }}

    const layout = {{
        height: 520,
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: {{ color: '#94a3b8', size: 12 }},
        margin: {{ l: 60, r: 30, t: 20, b: 60 }},
        xaxis: {{
            title: xDomain === 'rul' ? 'Remaining Useful Life (Cycles to Failure &bull; RUL = 0 is Failure)' : 'Operational Cycle',
            gridcolor: '#1e293b',
            autorange: xDomain === 'rul' ? 'reversed' : true
        }},
        yaxis: {{
            title: `${{meta.name}} (${{meta.unit}})`,
            gridcolor: '#1e293b'
        }},
        legend: {{
            orientation: 'h',
            y: 1.08,
            x: 0.5,
            xanchor: 'center',
            font: {{ color: '#f1f5f9' }}
        }},
        shapes: xDomain === 'rul' ? [
            {{
                type: 'line',
                x0: 60,
                x1: 60,
                y0: 0,
                y1: 1,
                yref: 'paper',
                line: {{ color: '#f59e0b', width: 2, dash: 'dash' }}
            }},
            {{
                type: 'line',
                x0: 0,
                x1: 0,
                y0: 0,
                y1: 1,
                yref: 'paper',
                line: {{ color: '#f43f5e', width: 2.5 }}
            }}
        ] : [],
        annotations: xDomain === 'rul' ? [
            {{
                x: 60,
                y: 0.95,
                yref: 'paper',
                text: 'Degradation Knee Point (~60 RUL)',
                showarrow: true,
                arrowhead: 2,
                arrowcolor: '#f59e0b',
                font: {{ color: '#f59e0b', size: 11 }}
            }},
            {{
                x: 0,
                y: 0.05,
                yref: 'paper',
                text: 'System Failure (RUL=0)',
                showarrow: true,
                arrowhead: 2,
                arrowcolor: '#f43f5e',
                font: {{ color: '#f43f5e', size: 11 }}
            }}
        ] : []
    }};

    Plotly.newPlot('fleet-plot-container', traces, layout, {{ responsive: true }});
}}

// TAB 3: Normalized Multi-Sensor Chart
function renderNormalizedChart() {{
    const unitId = parseInt(document.getElementById('norm-engine-select').value);
    const group = document.getElementById('norm-sensor-group').value;
    const alignDir = document.getElementById('norm-align-direction').checked;

    const engine = ENGINES_DATA[unitId];
    const maxCycle = engine.max_cycle;

    let targetSensors = [];
    if (group === 'top_degrading') {{
        targetSensors = ['sensor_4', 'sensor_11', 'sensor_12', 'sensor_7', 'sensor_15', 'sensor_21'];
    }} else if (group === 'temperatures') {{
        targetSensors = ['sensor_2', 'sensor_3', 'sensor_4'];
    }} else if (group === 'pressures') {{
        targetSensors = ['sensor_7', 'sensor_11', 'sensor_8', 'sensor_9', 'sensor_14'];
    }} else {{
        targetSensors = ACTIVE_SENSORS;
    }}

    const colors = [
        '#f43f5e', '#38bdf8', '#10b981', '#f59e0b', '#a855f7',
        '#ec4899', '#14b8a6', '#6366f1', '#fb923c', '#84cc16',
        '#06b6d4', '#eab308', '#d946ef', '#64748b'
    ];

    const traces = [];
    targetSensors.forEach((s, idx) => {{
        const meta = SENSOR_META[s];
        const smooth = movingAverage(engine.sensors[s], 5);
        const minVal = Math.min(...smooth);
        const maxVal = Math.max(...smooth);
        const range = maxVal - minVal || 1;

        let normalized = smooth.map(v => (v - minVal) / range);

        // If alignDir is true and sensor naturally drifts down, invert so 100% = failure
        let displayName = `${{s}} (${{meta.name}})`;
        if (alignDir && meta.dir === 'down') {{
            normalized = normalized.map(v => 1 - v);
            displayName += ' [Inverted]';
        }}

        traces.push({{
            x: engine.cycle,
            y: normalized.map(v => Math.round(v * 1000) / 10),
            mode: 'lines',
            line: {{ color: colors[idx % colors.length], width: 2.2 }},
            name: displayName
        }});
    }});

    const layout = {{
        height: 520,
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: {{ color: '#94a3b8', size: 12 }},
        margin: {{ l: 60, r: 30, t: 20, b: 60 }},
        xaxis: {{
            title: `Operational Cycle (Unit ${{unitId}} &bull; Failure at Cycle ${{maxCycle}})`,
            gridcolor: '#1e293b'
        }},
        yaxis: {{
            title: alignDir ? 'Degradation Severity Level (%) [0 = Baseline, 100% = Failure]' : 'Normalized Telemetry Range (%)',
            gridcolor: '#1e293b',
            range: [-5, 105]
        }},
        legend: {{
            orientation: 'h',
            y: 1.1,
            x: 0.5,
            xanchor: 'center',
            font: {{ color: '#f1f5f9' }}
        }},
        shapes: [
            {{
                type: 'line',
                x0: maxCycle,
                x1: maxCycle,
                y0: 0,
                y1: 1,
                yref: 'paper',
                line: {{ color: '#f43f5e', width: 2, dash: 'dot' }}
            }}
        ]
    }};

    Plotly.newPlot('norm-plot-container', traces, layout, {{ responsive: true }});
}}

// TAB 4: Correlation Chart and Reference Table
function renderCorrChart() {{
    const yLabels = CORR_DATA.map(d => `${{d.sensor}} (${{d.name}})`);
    const xVals = CORR_DATA.map(d => d.corr);
    const colors = xVals.map(v => v < 0 ? '#f43f5e' : '#38bdf8');

    const trace = {{
        x: xVals,
        y: yLabels,
        type: 'bar',
        orientation: 'h',
        marker: {{
            color: colors,
            line: {{ color: '#0f172a', width: 1 }}
        }},
        text: xVals.map(v => v.toFixed(3)),
        textposition: 'outside',
        hoverinfo: 'y+x'
    }};

    const layout = {{
        height: 420,
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: {{ color: '#94a3b8', size: 11 }},
        margin: {{ l: 140, r: 50, t: 20, b: 50 }},
        xaxis: {{
            title: 'Pearson Correlation with RUL (Negative = Rises with Age | Positive = Drops with Age)',
            gridcolor: '#1e293b',
            zerolinecolor: '#ffffff',
            zerolinewidth: 1.5,
            range: [-0.85, 0.85]
        }},
        yaxis: {{
            gridcolor: '#1e293b'
        }}
    }};

    Plotly.newPlot('corr-plot-container', [trace], layout, {{ responsive: true }});
}}

function populateSensorTable() {{
    const tbody = document.getElementById('sensor-table-body');
    tbody.innerHTML = '';

    for (let i = 1; i <= 21; i++) {{
        const s = `sensor_${{i}}`;
        const meta = SENSOR_META[s];
        const tr = document.createElement('tr');

        let trendClass = 'trend-flat';
        let trendIcon = '— Invariant';
        if (meta.dir === 'up') {{
            trendClass = 'trend-up';
            trendIcon = '▲ Increases (worsening)';
        }} else if (meta.dir === 'down') {{
            trendClass = 'trend-down';
            trendIcon = '▼ Decreases (worsening)';
        }}

        tr.innerHTML = `
            <td><strong>${{s}}</strong></td>
            <td style="color: #ffffff; font-weight: 600;">${{meta.name}}</td>
            <td><span class="badge" style="border-radius: 4px; padding: 2px 6px;">${{meta.system}}</span></td>
            <td>${{meta.desc}}</td>
            <td><code>${{meta.unit}}</code></td>
            <td>${{meta.active ? '<span style="color: var(--accent-emerald);">● Active Degrading</span>' : '<span style="color: var(--text-muted);">○ Invariant</span>'}}</td>
            <td class="${{trendClass}}">${{trendIcon}}</td>
            <td><strong>${{meta.corr !== undefined ? meta.corr.toFixed(3) : '0.000'}}</strong></td>
            <td style="color: var(--text-secondary);">${{meta.expl}}</td>
        `;
        tbody.appendChild(tr);
    }}
}}

// Window onload initialization
window.addEventListener('DOMContentLoaded', () => {{
    initControls();
    populateSensorTable();
    selectEngine(1);
}});
</script>

</body>
</html>
"""

    output_path = 'cmapss_fd001_dashboard.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Generated 100% self-contained dashboard: {output_path} ({os.path.getsize(output_path) / (1024 * 1024):.2f} MB)")

if __name__ == '__main__':
    generate_interactive_dashboard()
