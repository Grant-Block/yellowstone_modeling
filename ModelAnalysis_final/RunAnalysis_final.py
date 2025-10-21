import AnalyzeModel_final
import numpy as np

# GPS point locations calculated with the function distance_to_stations() in Analyze_GPS_final
#in
HVWY = (-2.74349, 9.103)
OFW2 = (0.2212, -15.942896)
LKWY = (6.04745, 14.27542)
P801 = (15.4316, -2.8647)
P709 = (24.3568, 11.8082)
WLWY = (3.9242, 24.52507)
#out
MAWY = (-37.80379, 18.96315)
P711 = (-16.03244, -8.1370)
P686 =  (5.18471, -44.492599)
P714 = (-33.35845, 11.85429)
P680 = (-21.359, -23.483)
P360 = (-10.675, -57.811)
P676 = (-34.408, -34.124)
P720 = (-21.78, 39.0465)
P456 = (-47.665, -16.974)
P361 = (-30.266, -44.706)


################################# model object declarations #############################################################
 
Yellowstone_Run201 = AnalyzeModel_final.Analysis("Yellowstone_Run201", r_x=0.5*13, r_y=1.0*27.5) # best fit model

Yellowstone_Run162 = AnalyzeModel_final.Analysis("Yellowstone_Run162", r_x=0.5*13, r_y=1.0*27.5) # uniform maxwell models used to look at depth variation (Fig. A4)
Yellowstone_Run163 = AnalyzeModel_final.Analysis("Yellowstone_Run163", r_x=0.5*13, r_y=1.0*27.5)
Yellowstone_Run164 = AnalyzeModel_final.Analysis("Yellowstone_Run164", r_x=0.5*13, r_y=1.0*27.5)

Yellowstone_Run192_run2 = AnalyzeModel_final.Analysis("Yellowstone_Run192_run2", r_x=0.5*13, r_y=1.0*27.5) # best fit uniform maxwell model

Yellowstone_Run2 = AnalyzeModel_final.Analysis("Yellowstone_Run2", r_x=0.5*13, r_y=0.5*27.5) # models used for scaling plot A7
Yellowstone_Run3 = AnalyzeModel_final.Analysis("Yellowstone_Run3", r_x=0.5*13, r_y=0.5*27.5)
Yellowstone_Run4 = AnalyzeModel_final.Analysis("Yellowstone_Run4", r_x=0.5*13, r_y=0.5*27.5)
Yellowstone_Run5 = AnalyzeModel_final.Analysis("Yellowstone_Run5", r_x=0.25*13, r_y=0.25*27.5)
Yellowstone_Run9 = AnalyzeModel_final.Analysis("Yellowstone_Run9", r_x=0.5*13, r_y=0.5*27.5)
Yellowstone_Run10 = AnalyzeModel_final.Analysis("Yellowstone_Run10", r_x=0.5*13, r_y=0.5*27.5)
Yellowstone_Run11 = AnalyzeModel_final.Analysis("Yellowstone_Run11", r_x=0.5*13, r_y=0.5*27.5)
Yellowstone_Run12 = AnalyzeModel_final.Analysis("Yellowstone_Run12", r_x=0.25*13, r_y=0.25*27.5)
Yellowstone_Run114 = AnalyzeModel_final.Analysis("Yellowstone_Run114", r_x=0.5*13, r_y=1.0*27.5, y_off=30) 
Yellowstone_Run116 = AnalyzeModel_final.Analysis("Yellowstone_Run116", r_x=0.5*13, r_y=1.0*27.5, y_off=30)
Yellowstone_Run118 = AnalyzeModel_final.Analysis("Yellowstone_Run118", r_x=0.5*13, r_y=1.0*27.5, y_off=30)
Yellowstone_Run119 = AnalyzeModel_final.Analysis("Yellowstone_Run119", r_x=0.5*13, r_y=1.0*27.5, y_off=30)
Yellowstone_Run120 = AnalyzeModel_final.Analysis("Yellowstone_Run120", r_x=0.5*13, r_y=1.0*27.5, y_off=30)
Yellowstone_Run121 = AnalyzeModel_final.Analysis("Yellowstone_Run121", r_x=0.5*13, r_y=1.0*27.5, y_off=30)
Yellowstone_Run122 = AnalyzeModel_final.Analysis("Yellowstone_Run122", r_x=0.5*13, r_y=1.0*27.5, y_off=30)
Yellowstone_Run123 = AnalyzeModel_final.Analysis("Yellowstone_Run123", r_x=0.5*13, r_y=1.0*27.5, y_off=30)
Yellowstone_Run124 = AnalyzeModel_final.Analysis("Yellowstone_Run124", r_x=0.5*13, r_y=1.0*27.5, y_off=30)
Yellowstone_Run125 = AnalyzeModel_final.Analysis("Yellowstone_Run125", r_x=0.5*13, r_y=1.0*27.5, y_off=30)
Yellowstone_Run126 = AnalyzeModel_final.Analysis("Yellowstone_Run126", r_x=0.5*13, r_y=1.0*27.5, y_off=30)
Yellowstone_Run127 = AnalyzeModel_final.Analysis("Yellowstone_Run127", r_x=0.5*13, r_y=1.0*27.5, y_off=30)
Yellowstone_Run159 = AnalyzeModel_final.Analysis("Yellowstone_Run159", r_x=0.5*13, r_y=1.0*27.5)
Yellowstone_Run165 = AnalyzeModel_final.Analysis("Yellowstone_Run165", r_x=0.5*13, r_y=1.0*27.5)
Yellowstone_Run160 = AnalyzeModel_final.Analysis("Yellowstone_Run160", r_x=0.5*13, r_y=1.0*27.5)
Yellowstone_Run166 = AnalyzeModel_final.Analysis("Yellowstone_Run166", r_x=0.5*13, r_y=1.0*27.5)
Yellowstone_Run161 = AnalyzeModel_final.Analysis("Yellowstone_Run161", r_x=0.5*13, r_y=1.0*27.5)
Yellowstone_Run167 = AnalyzeModel_final.Analysis("Yellowstone_Run167", r_x=0.5*13, r_y=1.0*27.5)
Yellowstone_Run169 = AnalyzeModel_final.Analysis("Yellowstone_Run169", r_x=0.5*13, r_y=1.0*27.5)
Yellowstone_Run172 = AnalyzeModel_final.Analysis("Yellowstone_Run172", r_x=0.5*13, r_y=1.0*27.5)
Yellowstone_Run170 = AnalyzeModel_final.Analysis("Yellowstone_Run170", r_x=0.5*13, r_y=1.0*27.5)
Yellowstone_Run171 = AnalyzeModel_final.Analysis("Yellowstone_Run171", r_x=0.5*13, r_y=1.0*27.5)
Yellowstone_Run173 = AnalyzeModel_final.Analysis("Yellowstone_Run173", r_x=0.5*13, r_y=1.0*27.5)
Yellowstone_Run174 = AnalyzeModel_final.Analysis("Yellowstone_Run174", r_x=0.5*13, r_y=1.0*27.5)
Yellowstone_Run179 = AnalyzeModel_final.Analysis("Yellowstone_Run179", r_x=0.5*13, r_y=1.0*27.53)
Yellowstone_Run183_run2 = AnalyzeModel_final.Analysis("Yellowstone_Run183_run2", r_x=0.5*13, r_y=1.0*27.5, mesh_width=100e3)
Yellowstone_Run184 = AnalyzeModel_final.Analysis("Yellowstone_Run184", r_x=0.5*13, r_y=1.0*27.5, mesh_width=100e3)
Yellowstone_Run185 = AnalyzeModel_final.Analysis("Yellowstone_Run185", r_x=0.5*13, r_y=1.0*27.5, mesh_width=100e3)
Yellowstone_Run186 = AnalyzeModel_final.Analysis("Yellowstone_Run186", r_x=0.5*13, r_y=1.0*27.5, mesh_width=100e3)
Yellowstone_Run189 = AnalyzeModel_final.Analysis("Yellowstone_Run189", r_x=0.5*13, r_y=1.0*27.5, mesh_width=100e3)
Yellowstone_Run190 = AnalyzeModel_final.Analysis("Yellowstone_Run190", r_x=0.5*13, r_y=1.0*27.5, mesh_width=100e3)
Yellowstone_Run191 = AnalyzeModel_final.Analysis("Yellowstone_Run191", r_x=0.5*13, r_y=1.0*27.5, mesh_width=100e3)

Yellowstone_Run181_run2 = AnalyzeModel_final.Analysis("Yellowstone_Run181_run2", r_x=0.5*13, r_y=1.0*27.5, mesh_width=100e3) # models used in mesh resolution figure
Yellowstone_Run181_run3 = AnalyzeModel_final.Analysis("Yellowstone_Run181_run3", r_x=0.5*13, r_y=1.0*27.5, mesh_width=100e3)
Yellowstone_Run181_run4 = AnalyzeModel_final.Analysis("Yellowstone_Run181_run4", r_x=0.5*13, r_y=1.0*27.5)

################################# plot function calls ##############################################################################################

# Function call to generate data files used in Fig. 3 (use makeObsFig_MRnew.py to generate the figure) and calculate model metrics
Yellowstone_Run201.plot_point_station_avg(np.arange(500, 539, 1), [LKWY, P801, WLWY, HVWY, OFW2, P709], [MAWY, P686, P714, P360], 
                                         'in_stations_mean_newgroups.csv', 'out_stations_mean_newgroups.csv', RG_file='hl_stations_mean.csv', 
                                         shift_time=1986, mult_factor=[1, 5], plot_unscaled=True, pressure_func_file=None, write_files=True,
                                         calc_metrics=True)

# Function call to make the left panels of Fig. 4 a-c
# model_time = 2006-1986+500
# Yellowstone_Run201.read_plot_groundsurf(model_time, mean_time_steps=[2005.5-1986+500, 2006.5-1986+500], parameter='vz', som_width=False, source=True, CR=True, CR_x=32, CR_y=55,
#                                         CR_inner=[22, 40], inner_CR_disp=[0, 0],
#                                         x_lim=[-75,75], y_lim=[-100,100], CR_disp=[-5.0, -0.75], profile=False, save=True, 
#                                         in_points_list=[LKWY, P801, WLWY, P709, HVWY, OFW2], 
#                                         out_points_list=[P711, MAWY, P686, P714, P360], 
#                                         plot_caldera=False, log_scale=False)

# Function call to make profile panels of Fig. 4 a-c
# Yellowstone_Run201.plot_profiles_mean_time([2005.5-1986+500, 2006.5-1986+500], theta=None, shift_time=1986, lim=[-60,60], 
#                                            plot_points=['MAWY.csv', 'P714.csv', 'P711.csv', 'HVWY.csv', 'LKWY.csv', 'P801.csv', 'P709.csv'],
#                                            station_locs = [MAWY, P714, P711, HVWY, LKWY, P801, P709])
# Yellowstone_Run201.plot_profiles_mean_time([2005.5-1986+500, 2006.5-1986+500], theta=np.pi/2., shift_time=1986, lim=[-60,60], 
#                                            plot_points=['P360.csv', 'P686.csv', 'OFW2.csv', 'HVWY.csv', 'LKWY.csv', 'WLWY.csv'], 
#                                            station_locs=[P360, P686, OFW2, HVWY, LKWY, WLWY])

# Function calls to generate Fig. A4
# Yellowstone_Run162.plot_cc_surface_time(np.arange(520, 531, 1), (32, 55), CR_shift=[-5.0, -0.75], save=True,
#                                         in_points_list=[LKWY, P801, WLWY, P709, HVWY, OFW2], 
#                                          out_points_list=[P711, MAWY, P686, P714, P360])
# Yellowstone_Run163.plot_cc_surface_time(np.arange(520, 531, 1), (32, 55), CR_shift=[-5.0, -0.75], save=True,
#                                         in_points_list=[LKWY, P801, WLWY, P709, HVWY, OFW2], 
#                                          out_points_list=[P711, MAWY, P686, P714, P360])
# Yellowstone_Run164.plot_cc_surface_time(np.arange(520, 531, 1), (32, 55), CR_shift=[-5.0, -0.75], save=True,
#                                         in_points_list=[LKWY, P801, WLWY, P709, HVWY, OFW2], 
#                                          out_points_list=[P711, MAWY, P686, P714, P360])

# Function call to make figure A6
# Yellowstone_Run192_run2.plot_point_station_avg(np.arange(500, 539, 1), [LKWY, P801, WLWY, HVWY, OFW2, P709], [MAWY, P686, P714, P360], 
#                                          'inner_mean_smoothed.csv', 'out_stations_mean_newgroups.csv', RG_file='hl_stations_mean.csv',
#                                          shift_time=1986, mult_factor=[1, 5], plot_unscaled=True, pressure_func_file=None,
#                                          calc_metrics=True)

# Function call to make figure A7
# AnalyzeModel_final.plot_nondim_profiles([[Yellowstone_Run118, Yellowstone_Run114, Yellowstone_Run119, Yellowstone_Run120],
#                                     [Yellowstone_Run121, Yellowstone_Run122, Yellowstone_Run123, Yellowstone_Run124],
#                                     [Yellowstone_Run125, Yellowstone_Run116, Yellowstone_Run126, Yellowstone_Run127]], 530,
#                                     add_models_list=[[Yellowstone_Run159, Yellowstone_Run162, Yellowstone_Run165], [Yellowstone_Run160, Yellowstone_Run163, Yellowstone_Run166], 
#                                          [Yellowstone_Run161, Yellowstone_Run164, Yellowstone_Run167], [Yellowstone_Run169, Yellowstone_Run172],
#                                          [Yellowstone_Run170, Yellowstone_Run173], [Yellowstone_Run171, Yellowstone_Run174],
#                                          [Yellowstone_Run2, Yellowstone_Run9], [Yellowstone_Run3, Yellowstone_Run10], [Yellowstone_Run4, Yellowstone_Run11],
#                                          [Yellowstone_Run5, Yellowstone_Run12]],
#                                     burgers_models_list=[[(Yellowstone_Run190, 9.462e15), (Yellowstone_Run189, 3.4694e16), (Yellowstone_Run185, 6.9388e16), 
#                                                           (Yellowstone_Run183_run2, 9.462e16), (Yellowstone_Run179, 1.0842e17), (Yellowstone_Run186, 6.9388e17)],
#                                                           [(Yellowstone_Run184, 6.9388e16), (Yellowstone_Run191, 6.9388e16)]], 
#                                     add_models_times=[520, 520, 520, 520, 520, 520, 511, 511, 511, 511], 
#                                     add_linestyles=['solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid'],
#                                     add_colors=['orange', 'green', 'purple', 'orange', 'green', 'blue', 'gray', 'gray', 'gray', 'gray'],
#                                     add_line_colors=['black', 'black', 'black', 'black', 'black', 'black', 'gray', 'gray', 'gray', 'gray'],
#                                     add_symbols=["^", "^", '^', "D", "D", 'D', '^', '^', '^', '^'], 
#                                     add_labels=[r"$d_s$=8.0 km",
#                                                 r"$d_s$=6.0 km",
#                                                 r"$d_s$=4.0 km",
#                                                 r"$d_s$=8.0 km, nested CR",
#                                                 r"$d_s$=6.0 km, nested CR",
#                                                 r"$d_s$=5.0 km, nested CR",
#                                                 r"Asymetric sawtooth time series, $d_s=6.5$ km", 
#                                                 r"Asymetric sawtooth time series, $d_s=7.25$ km",
#                                                 r"Asymetric sawtooth time series, $d_s=5.75$ km",
#                                                 r"Asymetric sawtooth time series, $d_s=6.5$ km"],
#                                     burgers_linestyles=['dashed', 'dashed'],
#                                     burgers_colors=['hotpink', 'cornflowerblue'],
#                                     burgers_symbols=['^', 'D'],
#                                     burgers_labels=[r"Uniform Burgers CR, $d_s$=4.0 km", r"Nested Burgers CR, $d_s$=5.0 km"])

# Function call to make panels a, c in Figure A8
# AnalyzeModel_final.mesh_resolution_analysis([Yellowstone_Run181_run2, Yellowstone_Run181_run3, Yellowstone_Run181_run4], [10, 5, 4.136], 520, 
#                                       ratio_list=[253.1, 227.9, 435.2]) 

