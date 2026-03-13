# analyze GPS stations and smooth the data
# Mousumi Roy April 2025
# 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import glob
import os
import numpy as np
from datetime import datetime

# Define the column names based on the file header
columns = ['time[yrs]','velocity[m/yr]']
# Load the data
# Create a dictionary to store dataframes
dataframes = {}

# Directory containing your files
directory_path = './Data_fig2d'
# directory_path = './Allstationgrps_smoothed'

# Loop over all .csv files
for filepath in glob.glob(os.path.join(directory_path, '*.csv')):
    filename = os.path.basename(filepath)
    var_name = filename[:4]  # first 4 letters
    df = pd.read_csv(filepath, sep=',', names=columns, skiprows=1)
    dataframes[var_name] = df

# outave = pd.read_csv('out_mean.csv',sep=',', names=columns, skiprows=1)
outave = pd.read_csv('./Data_fig2d/out_stations_mean_NRWY_newgroups.csv',sep=',', names=columns, skiprows=1)
outave_nrwy = pd.read_csv('./Data_fig2d/out_stations_mean_newgroups.csv',sep=',', names=columns, skiprows=1)
# inave  = pd.read_csv('in_mean_full.csv',sep=',', names=columns, skiprows=1)
inave  = pd.read_csv('in_stations_mean_newgroups.csv',sep=',', names=columns, skiprows=1)
hlave  = pd.read_csv('hl_stations_mean.csv',sep=',', names=columns, skiprows=1)
# --- Load earthquake data ---
eq_df = pd.read_csv('ys_eq.csv',sep=',')#, names=['time'], skiprows=1)


# inner = [ 'LKWY', 'HVWY','WLWY', 'P801','P709', 'OFW2']
# outer = ['NRWY','P711', 'P680','P686', 'P710', 'P714', 'MAWY','P360']

HLgroup = ['P361','P676', 'P456', 'P712', 'P460', 'P457', 'P461', 'P721', 'P720', 'P718', 'P710']
inner = ['LKWY', 'P801', 'WLWY', 'P709', 'HVWY', 'OFW2']
outer = ['NRWY', 'P711', 'MAWY', 'P686', 'P714', 'P360']

# Shades of blue (not using)
blue_colors = [
    '#000080',  # Navy
    '#0000CD',  # Medium Blue
    '#1E90FF',  # Dodger Blue
    '#4682B4',  # Steel Blue
    '#5DADE2',  # Light Blue 1
    '#87CEFA',  # Light Sky Blue
    '#ADD8E6',  # Light Blue 2
    '#B0E0E6'   # Powder Blue
]
# # Shades of red for outer stations (not using)
red_orange_colors = [
    '#8B0000',  # Dark Red
    '#B22222',  # Firebrick
    '#DC143C',  # Crimson
    '#FF4500',  # Orange Red
    '#FF6347',  # Tomato
    '#FF7F50',  # Coral
    '#FFA07A',  # Light Salmon
    '#FFD700'   # Gold (warm yellow-orange)
]

# Shades of purple for inner stations
purple_colors = [
    'blueviolet',
    'rebeccapurple',
    'mediumpurple',
    'darkorchid',
    'purple',
    'plum',
    'thistle',
    'm',
    'orchid'
]

# Shades of cyan for outer stations
cyan_colors = [
    'gray',
    'cadetblue',
    'dodgerblue',
    'deepskyblue',
    'steelblue',
    'turquoise',
    'aquamarine',
    'mediumturquoise',
    'royalblue',
    'darkblue'
]

greys = [plt.cm.Greys(i) for i in np.linspace(0.9, 0.2, 8)]

# Distinct symbols (Matplotlib markers)
markers = ['o',  # Circle
           's',  # Square
           '^',  # Triangle Up
           'v',  # Triangle Down
           'D',  # Diamond
           'P',  # Plus filled (Pentagon)
           'X',  # X filled (Cross)
           '*']  # Star

lstyles = [
    '-',       # Solid
    '--',      # Dashed
    '-.',      # Dash-dot
    '-',       # Solid
    '--',      # Dashed
    '-.',      # Dash-dot
    '-',       # Solid
    '--',      # Dashed
]

#InSAR and Leveling constraints from Dzurisin
pol1 = [[2003.7180941464765, 81.3953993913284],
[2006.7113948065296, 81.60942255246827],
[2007.7118493340186, 69.55950357693372],
[2008.8449270779813, 59.378878305205326],
[2008.7829730050196, 38.791747361764365],
[2007.6852891190072, 39.34607327773607],
[2006.7517094186, 49.79052211375044],
[2005.8869609896842, 49.805936524248054],
[2003.7210584561876, 68.02636259436386]
]
pol2 = [[2008.724753962294, 1.3596300541480701],
[2008.7290818544723, -18.159163669420167],
[2006.7051697561362, -40.31560017390618],
[2003.8115291885697, -39.99664044899407],
[2003.742045768942, -26.626417928145116],
[2006.7017904430656, -25.07489822536658],
[2006.7327971226434, -14.915023121615732],
[2007.6626417928146, -8.514485593454793],
[2007.66032963124, 1.913363108177549],
]
pol3=[[1995.7139243508163, 30.20117781905853],
[1996.778230109482, 30.182206236907632],
[1997.61197185882, 20.006916722659184],
[1998.648590964784, -5.14525117584283],
[2000.5774080075887, -4.110114224734204],
[2001.5410458084662, -0.11659618196908639],
[2002.869234417612, 9.752776570096046],
[2003.7338642741395, 10.27212363147703],
[2003.738310738706, -9.78143156396979],
[2002.8094146476426, -20.460060867159385],
[2001.6119520967552, -19.90395636536104],
[2000.5477056242837, -20.152365519149427],
[1998.8161930358485, -11.030591676218322],
[1997.6199754950399, -16.08948262914508],
[1996.7848701632347, 0.23556381170705265],
[1995.7516896565355, 9.879649025730217],
]
pol4 = [[2000.5695822299515, 31.184142919252196],
[2001.569918185052, 19.668985415596225],
[2002.8349077111577, 14.566222678945508],
[2003.7019683016483, 4.1229595668155525],
[2003.7392000316195, -13.792142603059162],
[2002.8411920477452, -13.77613533061934],
[2002.8394727481127, -6.022093988379893],
[2001.5422315323506, -5.464210900754907],
[2001.607624204577, -0.38516264179281734],
[2000.5099403185645, 0.1691632741788851],
[2000.6409035216, 9.525117584285226],
[2000.6386506462197, 19.68558554997827],
[1997.777913916446, 21.608236828583856],
[1997.77572032726, 31.501324058337616]
]
# --- Plot all on one figure ---
fig, axs = plt.subplots(3, 1, figsize=(14.8, 10),sharex=True)#,layout="constrained")

axs[0].add_patch(Polygon(pol1,closed=True, facecolor='palegreen', alpha=0.5))
axs[0].add_patch(Polygon(pol3,closed=True, facecolor='palegreen', alpha=0.5))
axs[1].add_patch(Polygon(pol2,closed=True, facecolor='khaki', alpha=0.5))
axs[1].add_patch(Polygon(pol4,closed=True, facecolor='khaki', alpha=0.5))
# read and plot average of inner stations
axs[0].plot(
        inave['time[yrs]'].values, 1.e3*inave['velocity[m/yr]'].values,
        label=r'$\langle V\rangle^{inner}$',
        color='indigo',
        linewidth=15.0, 
        alpha=0.5, 
        linestyle="dashed"
    )

# read and plot average of outer stations
axs[1].plot(
        outave_nrwy['time[yrs]'].values, 1.e3*outave_nrwy['velocity[m/yr]'].values,
        label=r'$\langle V\rangle^{anti}$',
        color='teal',
        linewidth=15.0, 
        alpha=0.55, 
        linestyle = '--'
    )

# read and plot average of outer stations including NRWY
axs[1].plot(
        outave['time[yrs]'].values, 1.e3*outave['velocity[m/yr]'].values,
        label=r'$\langle V\rangle^{anti_{incNRWY}}$',
        color='gray',
        linewidth=8.0, 
        alpha=0.5, 
        linestyle="dashed"
    )

# read and plot average of HL stations
axs[2].plot(
        hlave['time[yrs]'].values, 1.e3*hlave['velocity[m/yr]'].values,
        label=r'$\langle V\rangle^{RG}$',
        color='k',
        linewidth=15.0, 
        alpha=0.55, 
        linestyle = '--'
    )

for i, station in enumerate(inner):
    df = dataframes.get(station)
    if df is None or df.empty:
        continue

    time = df['time[yrs]'].values
    vel = df['velocity[m/yr]'].values
    
    color = purple_colors[i % len(purple_colors)]
    marker = markers[i % len(markers)]
    lstyle = lstyles[i % len(lstyles)]
    axs[0].plot(
        time, 1.e3*vel,
        label=station,
        color=color,
        marker=marker,
        markevery=5,  # show symbol every 10 points
        markersize=8,
        linestyle='--',
        #alpha=0.8
        linewidth=2.5, 
        alpha=0.8, 
        markeredgecolor='black', 
        markeredgewidth=0.5
    )


for i, station in enumerate(outer):
    df = dataframes.get(station)
    if df is None or df.empty:
        continue

    time = df['time[yrs]'].values
    vel = 1.e3*df['velocity[m/yr]'].values
    
    color = cyan_colors[i % len(cyan_colors)]
    marker = markers[i % len(markers)]
    lstyle = lstyles[i % len(lstyles)]
    axs[1].plot(
        time, vel,
        label=station,
        color=color,
        marker=marker,
        markevery=5,  # show symbol every 10 points
        markersize=8,
        linestyle='--',
        #alpha=0.8
        linewidth=2.5, 
        alpha=0.8, 
        markeredgecolor='black', 
        markeredgewidth=0.5
    )

for i, station in enumerate(HLgroup):
    df = dataframes.get(station)
    if df is None or df.empty:
        continue

    time = df['time[yrs]'].values
    vel = 1.e3*df['velocity[m/yr]'].values
    
    color = greys[i % len(greys)]
    marker = markers[i % len(markers)]
    lstyle = lstyles[i % len(lstyles)]
    axs[2].plot(
        time, vel,
        label=station,
        color=color,
        marker=marker,
        markevery=5,  # show symbol every 10 points
        markersize=8,
        linestyle='--',
        #alpha=0.8
        linewidth=2.5, 
        alpha=0.8, 
        markeredgecolor='black', 
        markeredgewidth=0.5
    )

axs[0].set_ylim(-30, 70)
axs[0].set_xlim(1995, 2025)
#axs[0].set_xlabel('Years',fontsize=16)
axs[0].set_ylabel('Velocity (mm/yr)',fontsize=16)
axs[0].tick_params(axis='both', labelsize=14)

#axs[0].legend(title='Station', handlelength=4,handleheight=1.5,fontsize=12, loc='upper left', bbox_to_anchor=(1.1, 1.1))
axs[0].legend(handlelength=4,handleheight=1.5,fontsize=12, loc='upper left', bbox_to_anchor=(1.1, 1.1))
axs[1].legend(handlelength=4,handleheight=1.5,fontsize=12, loc='upper left', bbox_to_anchor=(1.1, 1.1))
axs[2].legend(handlelength=4,handleheight=1.5,fontsize=12, loc='upper left', bbox_to_anchor=(1.1, 1.1))

#ax.set_title('Smoothed First Derivative: Inner (Blue) & Outer (Red) Stations')
axs[0].grid(True,which='major', color='gray', linewidth=0.8)
axs[1].grid(True,which='major', color='gray', linewidth=0.8)
axs[2].grid(True,which='major', color='gray', linewidth=0.8)
# Minor ticks and gridlines
axs[0].minorticks_on()
axs[0].grid(True, which='minor', color='lightgray', linestyle='--', linewidth=0.5)
axs[1].minorticks_on()
axs[1].grid(True, which='minor', color='lightgray', linestyle='--', linewidth=0.5)
axs[2].minorticks_on()
axs[2].grid(True, which='minor', color='lightgray', linestyle='--', linewidth=0.5)

#axs[1].set_ylim(-25, 10)
axs[1].set_ylim(-10, 7.5)
axs[2].set_ylim(-5, 8)
axs[2].set_xlabel('Years',fontsize=16)
axs[2].set_ylabel('Velocity (mm/yr)',fontsize=16)
axs[1].set_ylabel('Velocity (mm/yr)',fontsize=16)
axs[2].tick_params(axis='both', labelsize=14)

eq_df['year'] = pd.to_numeric(eq_df['time'].str[:4].values)
eq_df['mo'] = pd.to_numeric(eq_df['time'].str[5:7].values)
eq_df['decyr'] = eq_df['year'].values + eq_df['mo'].values/12


# --- Define time bins ---
bin_width = 0.5  # in years
time_start = eq_df['decyr'].min()
time_end = eq_df['decyr'].max()
bins = np.arange(time_start, time_end + bin_width, bin_width)
bin_centers = (bins[:-1] + bins[1:]) / 2

# --- Histogram: count number of quakes per time bin ---
hist_counts, _ = np.histogram(eq_df['decyr'], bins=bins)
ax2 = axs[1].twinx()
ax2.bar(
    bin_centers, hist_counts,
    width=bin_width * 0.9,
    color='gray',
    alpha=0.7,
    label='Earthquake Count'
)

ax2.set_ylim(0, 125)
ax2.set_ylabel('# Earthquakes', fontsize=14)
ax2.tick_params(axis='y')#, labelcolor='gray')

plt.tight_layout(rect=[0, 0, 0.95, 0.95])  # Leave space for suptitle
plt.subplots_adjust(wspace=None, hspace=None)
#fig.subplots_adjust(right=0.55) 

plt.show()

plot_filename = f'Obs_All_groups.png'
fig.savefig(plot_filename)#,bbox_extra_artists=(ax.legend), bbox_inches='tight')
#Note that the bbox_extra_artists must be an iterable)  # high-quality figure
plt.close()  # close figure to save memory

############################################################################################
# new figure for looking at HLgroup average removed from all --- first make the subtracted arrays

outmerge = pd.merge(hlave, outave_nrwy, on='time[yrs]', suffixes=('_hl', '_out'))
outmerge['velocity_diff'] = outmerge['velocity[m/yr]_out'] - outmerge['velocity[m/yr]_hl'] 
outwithNRWYmerge = pd.merge(hlave, outave, on='time[yrs]', suffixes=('_hl', '_out_nonrwy'))
outwithNRWYmerge['velocity_diff'] = outwithNRWYmerge['velocity[m/yr]_out_nonrwy'] - outwithNRWYmerge['velocity[m/yr]_hl'] 
inmerge = pd.merge(hlave, inave, on='time[yrs]', suffixes=('_hl', '_in'))
inmerge['velocity_diff'] = inmerge['velocity[m/yr]_in'] - inmerge['velocity[m/yr]_hl'] 

model_inner_df = pd.read_csv(directory_path+'/model_inner.csv', sep=',')
model_outer_df = pd.read_csv(directory_path+'/model_outer.csv', sep=',')


fig, axs = plt.subplots(2, 1, figsize=(18.8, 6),sharex=True)#,layout="constrained")
# fig, axs = plt.subplots(2, 1, figsize=(14.8, 6),sharex=True)#,layout="constrained")


axs[0].add_patch(Polygon(pol1,closed=True, facecolor='palegreen', alpha=0.5))
axs[0].add_patch(Polygon(pol3,closed=True, facecolor='palegreen', alpha=0.5))
axs[1].add_patch(Polygon(pol2,closed=True, facecolor='khaki', alpha=0.5))
axs[1].add_patch(Polygon(pol4,closed=True, facecolor='khaki', alpha=0.5))
# read and plot average of inner stations
axs[0].plot(
        inave['time[yrs]'].values, 1.e3*inave['velocity[m/yr]'].values,
        label=r'$\langle V\rangle^{inner}$',
        color='indigo',
        linewidth=20.0,
        # linewidth=10.0,  
        alpha=0.5, 
    )

axs[0].plot(
        inmerge['time[yrs]'].values, 1.e3*inmerge['velocity_diff'].values,
        label=r'$\langle V\rangle^{inner}$ - $\langle V\rangle^{RG}$',
        color='black',
        linewidth=10.0,
        # linewidth=5.0, 
        alpha=0.5, 
        linestyle="dashed"
    )

# plot the model average
axs[0].plot(
    model_inner_df['time[yrs]'].values, model_inner_df['velocity[mm/yr]'].values,
    label=r'2-source',
    color='indigo',
    linewidth=10,
    # linewidth=5,
    alpha=0.7
)

# read and plot average of outer stations
axs[1].plot(
        outave_nrwy['time[yrs]'].values, 1.e3*outave_nrwy['velocity[m/yr]'].values,
        label=r'$\langle V\rangle^{anti}$',
        color='teal',
        linewidth=20.0,
        # linewidth=10.0,  
        alpha=0.55, 
    )

# read and plot average of outer stations including NRWY
# axs[1].plot(
#         outave['time[yrs]'].values, 1.e3*outave['velocity[m/yr]'].values,
#         label=r'$\langle V\rangle^{anti}$ with NRWY',
#         color='gray',
#         linewidth=8.0, 
#         alpha=0.5, 
#     )

axs[1].plot(
        outmerge['time[yrs]'].values, 1.e3*outmerge['velocity_diff'].values,
        label=r'$\langle V\rangle^{anti}$ - $\langle V\rangle^{RG}$',
        color='black',
        # linewidth=5.0, 
        linewidth=10.0,
        alpha=0.5, 
        linestyle="dashed"
    )

# plot the model outer average (unscaled and scaled)
axs[1].plot(
    model_outer_df['time[yrs]'].values, model_outer_df['velocity[mm/yr]'].values,
    label=r'2-source',
    color='teal',
    # linewidth=5,
    linewidth=10,
    alpha=0.7
)

# axs[1].plot(
#     model_outer_df['time[yrs]'].values, model_outer_df['velocity[mm/yr]'].values*5,
#     label=r'1-source $\times$ 5',
#     color='teal',
#     linewidth=10,
#     linestyle="dashed",
#     alpha=0.7
# )

# --- Histogram: count number of quakes per time bin ---
hist_counts, _ = np.histogram(eq_df['decyr'], bins=bins)
ax2 = axs[1].twinx()
ax2.bar(
    bin_centers, hist_counts,
    width=bin_width * 0.9,
    color='gray',
    alpha=0.7,
    label='Earthquake Count'
)

ax2.set_ylim(0, 125)
ax2.set_ylabel('# Earthquakes', fontsize=14)
ax2.tick_params(axis='y')#, labelcolor='gray')
# axs[0].set_ylim(-25, 50)
axs[0].set_ylim(-40, 65)
axs[0].set_xlim(1995, 2025)
#axs[0].set_xlabel('Years',fontsize=16)
axs[0].set_ylabel('Velocity (mm/yr)',fontsize=16)
axs[0].tick_params(axis='both', labelsize=14)

axs[0].legend(handlelength=4,handleheight=1.5,fontsize=12, loc='upper left', bbox_to_anchor=(1.1, 1.1))
axs[1].legend(handlelength=4,handleheight=1.5,fontsize=12, loc='upper left', bbox_to_anchor=(1.1, 1.1))

axs[0].grid(True,which='major', color='gray', linewidth=0.8)
axs[1].grid(True,which='major', color='gray', linewidth=0.8)
# Minor ticks and gridlines
axs[0].minorticks_on()
axs[0].grid(True, which='minor', color='lightgray', linestyle='--', linewidth=0.5)
axs[1].minorticks_on()
axs[1].grid(True, which='minor', color='lightgray', linestyle='--', linewidth=0.5)

axs[1].set_ylim(-10, 8)
# axs[1].set_ylim(-15, 8)
axs[1].set_xlabel('Years',fontsize=16)
axs[1].set_ylabel('Velocity (mm/yr)',fontsize=16)
axs[1].tick_params(axis='both', labelsize=14)

axs[0].grid(True,which='major', color='gray', linewidth=0.4)
axs[0].grid(True, which='minor', color='lightgray', linestyle='--', linewidth=0.2)

axs[1].grid(True,which='major', color='gray', linewidth=0.4)
axs[1].grid(True, which='minor', color='lightgray', linestyle='--', linewidth=0.2)

plt.tight_layout(rect=[0, 0, 0.95, 0.95])  # Leave space for suptitle
plt.subplots_adjust(wspace=None, hspace=None)

plt.show()

filename = f'Obs_no_RG.png'
fig.savefig(filename)#,bbox_extra_artists=(ax.legend), bbox_inches='tight')
#Note that the bbox_extra_artists must be an iterable)  # high-quality figure
plt.close()  # close figure to save memory

