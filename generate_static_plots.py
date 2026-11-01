import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(3, 2, figsize=(16, 14), dpi=150)
fig.suptitle('NASA C-MAPSS FD001 Turbofan Engine Degradation Time Series', fontsize=18, fontweight='bold', y=0.98)

col_names = ['unit', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
df = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None, names=col_names)
max_cycles = df.groupby('unit')['cycle'].transform('max')
df['RUL'] = max_cycles - df['cycle']

sensor_labels = {
    'sensor_2': 'T24 - LPC Outlet Temp (°R)',
    'sensor_3': 'T30 - HPC Outlet Temp (°R)',
    'sensor_4': 'T50 - LPT Outlet Temp (°R)',
    'sensor_7': 'P30 - HPC Outlet Pressure (psia)',
    'sensor_8': 'Nf - Physical Fan Speed (rpm)',
    'sensor_9': 'Nc - Physical Core Speed (rpm)',
    'sensor_11': 'Ps30 - HPC Static Pressure (psia)',
    'sensor_12': 'phi - Fuel Flow / Ps30 (pps/psia)',
    'sensor_13': 'NRf - Corrected Fan Speed (rpm)',
    'sensor_14': 'NRc - Corrected Core Speed (rpm)',
    'sensor_15': 'BPR - Bypass Ratio',
    'sensor_17': 'htBleed - Bleed Enthalpy',
    'sensor_20': 'W31 - HPT Coolant Bleed (lbm/s)',
    'sensor_21': 'W32 - LPT Coolant Bleed (lbm/s)'
}

# 1. Single Engine Run-to-Failure (Unit 1, 4 selected sensors)
ax1 = axes[0, 0]
u1 = df[df['unit'] == 1]
ax1.plot(u1['cycle'], u1['sensor_4'], color='#d62728', lw=1.5, label='sensor_4 (T50 LPT Temp)')
ax1.set_title('Unit 1: T50 LPT Outlet Temperature vs Cycle (Failure at 192)', fontweight='semibold')
ax1.set_xlabel('Operational Cycle')
ax1.set_ylabel('Temperature (°R)')
ax1.axvline(192, color='black', linestyle='--', alpha=0.7, label='Failure Point (Cycle 192)')
ax1.legend(loc='upper left')

# 2. Single Engine (Unit 1: HPC Outlet Pressure P30)
ax2 = axes[0, 1]
ax2.plot(u1['cycle'], u1['sensor_7'], color='#1f77b4', lw=1.5, label='sensor_7 (P30 HPC Pressure)')
ax2.set_title('Unit 1: P30 HPC Outlet Pressure vs Cycle', fontweight='semibold')
ax2.set_xlabel('Operational Cycle')
ax2.set_ylabel('Pressure (psia)')
ax2.axvline(192, color='black', linestyle='--', alpha=0.7, label='Failure Point (Cycle 192)')
ax2.legend(loc='lower left')

# 3. Fleet Comparison: T50 vs Remaining Useful Life (RUL)
ax3 = axes[1, 0]
sample_units = [1, 2, 10, 20, 34, 39, 45, 55, 69, 80, 90, 100]
for u in sample_units:
    udf = df[df['unit'] == u]
    ax3.plot(udf['RUL'], udf['sensor_4'], color='steelblue', alpha=0.35, lw=1)
# Compute and plot fleet mean
mean_rul = df.groupby('RUL')['sensor_4'].mean().loc[lambda x: x.index <= 200]
ax3.plot(mean_rul.index, mean_rul.values, color='crimson', lw=2.5, label='Fleet Mean (All 100 Engines)')
ax3.invert_xaxis()
ax3.set_title('Fleet Trajectories: T50 LPT Temp vs RUL (Reversed: Failure at RUL=0)', fontweight='semibold')
ax3.set_xlabel('Remaining Useful Life (Cycles to Failure)')
ax3.set_ylabel('T50 (°R)')
ax3.axvline(50, color='orange', linestyle=':', lw=2, label='Degradation Knee Point (~50 RUL)')
ax3.legend(loc='upper right')

# 4. Fleet Comparison: Ps30 vs Remaining Useful Life (RUL)
ax4 = axes[1, 1]
for u in sample_units:
    udf = df[df['unit'] == u]
    ax4.plot(udf['RUL'], udf['sensor_11'], color='darkseagreen', alpha=0.35, lw=1)
mean_rul_ps30 = df.groupby('RUL')['sensor_11'].mean().loc[lambda x: x.index <= 200]
ax4.plot(mean_rul_ps30.index, mean_rul_ps30.values, color='forestgreen', lw=2.5, label='Fleet Mean (All 100 Engines)')
ax4.invert_xaxis()
ax4.set_title('Fleet Trajectories: Ps30 HPC Static Pressure vs RUL', fontweight='semibold')
ax4.set_xlabel('Remaining Useful Life (Cycles to Failure)')
ax4.set_ylabel('Ps30 (psia)')
ax4.axvline(50, color='orange', linestyle=':', lw=2, label='Degradation Knee Point (~50 RUL)')
ax4.legend(loc='lower right')

# 5. Lifespan Distribution across all 100 engines
ax5 = axes[2, 0]
lifespans = df.groupby('unit')['cycle'].max()
sns.histplot(lifespans, bins=15, kde=True, ax=ax5, color='#4575b4', edgecolor='black')
ax5.axvline(lifespans.mean(), color='red', linestyle='--', lw=2, label=f'Mean Life: {lifespans.mean():.1f} cycles')
ax5.axvline(lifespans.min(), color='purple', linestyle=':', lw=1.5, label=f'Min Life: {lifespans.min()} (Unit 39)')
ax5.axvline(lifespans.max(), color='green', linestyle=':', lw=1.5, label=f'Max Life: {lifespans.max()} (Unit 69)')
ax5.set_title('Distribution of Engine Lifespans (100 Engines)', fontweight='semibold')
ax5.set_xlabel('Max Operating Cycles to Failure')
ax5.set_ylabel('Engine Count')
ax5.legend()

# 6. Sensor Correlation with RUL
ax6 = axes[2, 1]
active_sensors = list(sensor_labels.keys())
corrs = df[active_sensors + ['RUL']].corr()['RUL'].drop('RUL').sort_values()
labels_sorted = [f"{s} ({sensor_labels[s].split(' - ')[0]})" for s in corrs.index]
colors = ['#d73027' if c < 0 else '#4575b4' for c in corrs.values]
ax6.barh(labels_sorted, corrs.values, color=colors, edgecolor='black', alpha=0.85)
ax6.set_title('Sensor Correlation with Remaining Useful Life (RUL)', fontweight='semibold')
ax6.set_xlabel('Pearson Correlation Coefficient (r)')
ax6.axvline(0, color='black', lw=1)
ax6.text(-0.65, 1, 'Increases as engine fails\n(Negative r)', color='#a50026', fontweight='bold', fontsize=9)
ax6.text(0.15, 12, 'Decreases as engine fails\n(Positive r)', color='#313695', fontweight='bold', fontsize=9)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('cmapss_fd001_overview.png', dpi=150)
print('Static overview plot saved as cmapss_fd001_overview.png')
