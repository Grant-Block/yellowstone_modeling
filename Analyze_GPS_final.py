import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import os
from enum import Enum
import pygmt
from pathlib import Path

# Helper direction Enum
class Direction(Enum):
    EASTING = 1
    NORTHING = 2
    UP = 3

# Global time cutoff variables for P801
P801_pre_quake = 2011.5
P801_post_quake = 2014


################## Class: ReadGPS ###################################################

# class to read in yellowstone GPS files and extract relevant data

class ReadGPS:

    def __init__(self, path):

        self.path = path
        self.read_data()

    # read the GPS data file and store
    # data as class variables

    def read_data(self):

        df = pd.read_csv(self.path, delim_whitespace=True)

        # get name of station
        self.name = df['site'][0]

        # get lat and long of station from first entries in 
        # data set
        self.lat = df['_latitude(deg)'][0]
        self.long = df['_longitude(deg)'][0]

        # get data east, north and up and times
        self.dec_years = df['yyyy.yyyy']
        self.easting = df['__east(m)']
        self.northing = df['_north(m)']
        self.up = df['____up(m)']

        # get errors
        self.sig_e = df['sig_e(m)']
        self.sig_n = df['sig_n(m)']
        self.sig_u = df['sig_u(m)']


    # Smooth station data with a moving window average.
    # Takes a direction (default is UP), window size (default is 2 yrs), and step size (default is 0.2 yrs), and whether to return smoothed errors
    # Returns an array of times [yrs] of the same length as the smoothed data, and an array of smoothed data [m]
    # Optionally returns an array of smoothed displacement errors [m]
    def smooth_component(self, dir=Direction.UP, window=2, step=0.2, return_errors=False):

        # choose what data and error component to use based on direction
        if dir == Direction.EASTING:
            component = self.easting
            error = self.sig_e
        elif dir == Direction.NORTHING:
            component = self.northing
            error = self.sig_n
        else:
            component = self.up
            error = self.sig_u
        
        smoothed_component = []
        smoothed_error = []
        smoothed_times = []

        # start and end times are the nearest times divisible by step at the start and end of the time series
        start_time = math.ceil(self.dec_years[0]*(1/step)) * step
        end_time = math.floor(self.dec_years[len(self.dec_years)-1]*(1/step)) * step

        # iterate through each step
        for t in np.arange(start_time+(window/2), end_time-(window/2)+step, step):
            t_avg_idx = (self.dec_years >= t-(window/2)) & (self.dec_years <= t+(window/2)) # gets index of all times in window
            comp_avg = 0
            err_avg = 0
            w_sum = 0

            # iterate through all times in window
            for i in range(len(component[t_avg_idx])):
                ti = np.asarray(self.dec_years)[t_avg_idx][i] # time
                ui = np.asarray(component)[t_avg_idx][i] # displacement
                ei = np.asarray(error)[t_avg_idx][i] # error
                
                weight = 1.0-0.9*(np.abs(t-ti)/(window/2)) # weight is dependent on how far ti is from the center of the window
                comp_avg += ui*weight
                err_avg += (ei*weight)**2
                w_sum += weight

            # weighted mean of displacement and error with time centered in the window
            smoothed_component.append(comp_avg/w_sum)
            smoothed_error.append(np.sqrt(err_avg)/w_sum)
            smoothed_times.append(t)


        if return_errors:
            return np.asarray(smoothed_times), np.asarray(smoothed_component), np.asarray(smoothed_error)
        else:
            return np.asarray(smoothed_times), np.asarray(smoothed_component)


    # Smooths station data and calculates its velocity and optionally the velocity errors
    # Takes direction for which component to use, default is Direction.UP and an option to return the velocity errors
    # Returns an array of times [yrs] of the same length of the velocities (and smoothed data) and velocity array [m/yr] 
    # and optionally velocity error array [m/yr]
    def calculate_velocity(self, dir=Direction.UP, return_errors=False):

        # get smoothed displacement component and error
        if dir == Direction.EASTING:
            smoothed_times, smoothed_comp, smoothed_err = self.smooth_component(dir=Direction.EASTING, return_errors=True)
        elif dir == Direction.NORTHING:
            smoothed_times, smoothed_comp, smoothed_err = self.smooth_component(dir=Direction.NORTHING, return_errors=True)
        else:
            smoothed_times, smoothed_comp, smoothed_err = self.smooth_component(return_errors=True)
        
        
        if self.name == 'P801': # If the station is P801 calculate the velocity before and after EQ jump and just concatenate the two arrays
            vel1 = np.gradient(smoothed_comp[smoothed_times <= P801_pre_quake], smoothed_times[smoothed_times <= P801_pre_quake])
            vel2 = np.gradient(smoothed_comp[smoothed_times > P801_post_quake], smoothed_times[smoothed_times > P801_post_quake])
            vel = np.concatenate((vel1, vel2))
            smoothed_times_err = smoothed_times
            smoothed_times = np.concatenate((smoothed_times[smoothed_times <= P801_pre_quake], smoothed_times[smoothed_times > P801_post_quake]))
        else:
            vel =  np.gradient(smoothed_comp, smoothed_times) # take the gradient of the smoothed data for the velocity

        # calculate velocity error
        if return_errors:
            vel_err = []
            dt = smoothed_times[1]-smoothed_times[0] # get dt

            # central difference of errors
            if self.name == 'P801':
                for i in range(len(smoothed_err[smoothed_times_err <= P801_pre_quake])):
                    if i == 0:
                        vel_err.append(np.sqrt(smoothed_err[0]**2 + smoothed_err[1]**2)/dt)
                    elif i == len(smoothed_err[smoothed_times_err <= P801_pre_quake])-1:
                        vel_err.append(np.sqrt(smoothed_err[i]**2 + smoothed_err[i-1]**2)/dt)
                    else:
                        vel_err.append(np.sqrt(smoothed_err[i-1]**2 + smoothed_err[i+1]**2)/(2*dt))

                for i in range(len(smoothed_err[smoothed_times_err > P801_post_quake])):
                    if i == 0:
                        vel_err.append(np.sqrt(smoothed_err[0]**2 + smoothed_err[1]**2)/dt)
                    elif i == len(smoothed_err[smoothed_times_err > P801_post_quake])-1:
                        vel_err.append(np.sqrt(smoothed_err[i]**2 + smoothed_err[i-1]**2)/dt)
                    else:
                        vel_err.append(np.sqrt(smoothed_err[i-1]**2 + smoothed_err[i+1]**2)/(2*dt))
            else:
                for i in range(len(smoothed_err)):
                    if i == 0:
                        vel_err.append(np.sqrt(smoothed_err[0]**2 + smoothed_err[1]**2)/dt)
                    elif i == len(smoothed_err)-1:
                        vel_err.append(np.sqrt(smoothed_err[i]**2 + smoothed_err[i-1]**2)/dt)
                    else:
                        vel_err.append(np.sqrt(smoothed_err[i-1]**2 + smoothed_err[i+1]**2)/(2*dt))
            
            return smoothed_times, vel, np.asarray(vel_err)


        return smoothed_times, vel

# class ReadGPS
################################################################################################################
    
# Helper function to compile list of ReadGPS objects which can then be passed into plotting functions
# Takes a station_subset which should be a list of strings, being the name of a GPS station.
# Returns a list of ReadGPS objects with each station being specified by the name in station_subset
def get_data_list(station_subset = None):
    directory = "/home/grantblock/Research/Yellowstone/GPS_Data" #IMPORTANT: Change to wherever your GPS_Data folder is
    data_list = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            station = ReadGPS(f) # create ReadGPS station object with the file
            if station_subset == None:
                data_list.append(station)
            elif station.name in station_subset: #check if the station name is in our subset before appending to the list
                data_list.append(station)

    return data_list

# Helper function to calculate the mean velocity of a given station within a time span
# Takes a ReadGPS object (station), optional direction (dir, default is Direction.UP) and optional tuple time span
# [start, stop] (time_span, yrs). If no time span is provided the function will calculated the mean velocity for the entire
# station time series.
def velocity_metric(station, dir=Direction.UP, time_span=None):

    # get velocity
    smoothed_times, vel = station.calculate_velocity(dir=dir)

    # Only calculate velocity within time span if it's specified
    if time_span != None:
        calc_times_idx = (smoothed_times >= time_span[0]) & (smoothed_times <= time_span[1])
    else:
        calc_times_idx = (smoothed_times >= smoothed_times[0]) & (smoothed_times <= smoothed_times[len(smoothed_times)-1])
    
    return np.mean(vel[calc_times_idx])


# Function to calculate the velocity mean of a list of stations.
# Takes station_list, a list of stations. Also takes optional direction, if none is specified the direction will be UP.
# Takes boolean plot argument. Default is false, if True the function will plot the individual station velocities and their mean.
# Takes boolean argument to write csv files of the smoothed times and velocities of all stations in the group
# Returns a tuple of times and mean velocity values in m/yr.
def calc_subset_vel_mean(station_list, dir=Direction.UP, write_individual_file=False, plot=False):

    # get min and max times for all the stations
    min_time = np.inf
    max_time = -1
    for station in station_list:
        if min(station.dec_years) < min_time:
            min_time = min(station.dec_years)
        if max(station.dec_years) > max_time:
            max_time = max(station.dec_years)

    # get list of times and velocities for each station
    times_list = []
    vel_list = []
    for station in station_list:
        smoothed_times, vel = station.calculate_velocity(dir=dir)
        times_list.append(smoothed_times)
        vel_list.append(vel)
        if write_individual_file:
            write_data_file(smoothed_times, vel, station.name+".csv")

    mean = []
    plot_times = []
    dt = times_list[0][1]-times_list[0][0] # time step and largest allowed difference in time between two data points for them to be averaged
    for t in np.arange(min_time, max_time+dt, dt): # loop the times that span all stations
        curr_mean = 0
        curr_time = 0
        num_valid = 0
        for i in range(len(vel_list)): # loop through each station in list
            closest_time_idx = (np.abs(t - times_list[i])).argmin() # get the index of the datum closest in time to t for a given station

            # check if the time selected datum is within the time window and is within dt of t
            # if it is then we add it to the mean 
            if times_list[i][closest_time_idx] >= min_time and times_list[i][closest_time_idx] <= max_time and np.abs(t - times_list[i][closest_time_idx]) <= dt:
                curr_mean += vel_list[i][closest_time_idx]
                curr_time += times_list[i][closest_time_idx]
                num_valid += 1
        if num_valid > 0:
            mean.append(curr_mean/num_valid)
            plot_times.append(curr_time/num_valid)

    # plot velocities and their mean if requested
    if plot:
        plt.xlabel("Time [yrs]", fontsize=20)
        plt.ylabel("Velocity [m/yr]", fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.grid()
        plt.plot(np.asarray(plot_times), np.asarray(mean), lw=4, c='blue')
        for i in range(len(vel_list)):
            if station_list[i].name == 'P801':
                plt.plot(times_list[i][times_list[i] <= P801_pre_quake], vel_list[i][times_list[i] <= P801_pre_quake], c='black', linestyle='dashed')
                plt.plot(times_list[i][times_list[i] >= P801_post_quake], vel_list[i][times_list[i] >= P801_post_quake], c='black', linestyle='dashed')
                plt.text(times_list[i][0], vel_list[i][0], station_list[i].name, fontweight='bold')
            else:
                plt.plot(times_list[i], vel_list[i], c='black', linestyle='dashed')
                plt.text(times_list[i][0], vel_list[i][0], station_list[i].name, fontweight='bold')
        
        # plt.plot(np.linspace(min_time_all, max_time_all, len(mean)), mean, lw=4, c='blue')

        
        plt.xlim([min_time, max_time])

        plt.show()
    
    return np.asarray(plot_times), np.asarray(mean)



# Function to plot GMT location map with caldera, faults, earthquakes, source and CR. It takes a list of ReadGPS objects (station list),
# a tuple for the source dimension [minor_axis, major_axis] (source_dim, km), a tuple for source center [source_long, source_lat], (source center, deg),
# an angle for the source (source_angle, deg), all three of those arguments for the CR (CR_dim), (CR_center), (CR_angle), optionally those arguments for the inner CR
# as well, optional file paths for the caldera (caldera_file), older calderas (add_caldera_file), faults (fault_file), and earthquake epicenters (eq_file). 
# Also takes optional list of bounds [long_min, long_max, lat_min, lat_max] (bounds, deg) and boolean to plot state lines (statelines).
def make_location_map(station_list, source_dim=None, source_center=None, source_angle=None, CR_dim=None, CR_center=None, CR_angle=None,
                      inner_CR_dim=None, inner_CR_center=None, inner_CR_angle=None, caldera_file=None, add_caldera_file=None, fault_file=None, 
                      eq_file=None, bounds=None, statelines=False, plot_stations=False, in_stations=[], out_stations=[]):

   # Find the max and min longitude and latitude in our stations
    data_min_lat = np.inf
    data_min_long = np.inf
    data_max_lat = -np.inf
    data_max_long = -np.inf
    for station in station_list:
        if station.lat > data_max_lat:
            data_max_lat = station.lat
        if station.lat < data_min_lat:
            data_min_lat = station.lat
        if station.long > data_max_long:
            data_max_long = station.long
        if station.long < data_min_long:
            data_min_long = station.long

    # setup for PyGMT
    topo_data = '@earth_relief_15s' #15 arc second global relief (SRTM15+V2.1)
    pygmt.makecpt(cmap='topo',series='-8000/8000/1000',continuous=True)

    # set plot bounds, if none are provided then set them to be +/- 0.5 degrees from the max/min longitude and latitude
    if bounds == None:
        region = [
            data_min_long - 0.5,
            data_max_long + 0.5,
            data_min_lat - 0.5,
            data_max_lat + 0.5,
        ]
    else:
        region = bounds

    fig = pygmt.Figure()

    proj = 'M4i'
    with pygmt.config(FONT="20p"):
        fig.grdimage(grid=topo_data, shading=True, region=region,projection=proj, frame=True)
        if statelines: # plot state lines if requested
            fig.coast(region=region,projection=proj, frame=True, water="lightblue", rivers="lightblue", borders=["2/0.5p,black"])
        else:
            fig.coast(region=region,projection=proj, frame=True, water="lightblue", rivers="lightblue")

    with pygmt.config(FONT="16p"): # plot scale bar
        fig.basemap(map_scale="n0.18/0.09+w50k+f+u")
        

     # plot EQ's if a file is provided
    if eq_file != None:
        
        eq_df = pd.read_csv(eq_file, index_col=False)
        eq_mag = eq_df['mag']
        eq_lat = eq_df['latitude']
        eq_long = eq_df['longitude']

        fig.plot(x=eq_long, y=eq_lat,color="slateblue2", style="c0.1c", pen='black')

    # Plot faults if file is provided
    if fault_file != None:
        fig.plot(data=fault_file, pen="1p,maroon3")

    # Plot older caldera outlines if file is provided
    if add_caldera_file != None:
        fig.plot(data=add_caldera_file, pen="2p,darkbrown")
        # fig.plot(data=add_caldera_file, pen="2p,magenta")



   # Plot source
    if source_dim != None:
        fig.plot(x=source_center[0], y=source_center[1], style="e"+str(source_angle)+"/"+str(source_dim[0])+"/"+str(source_dim[1]), pen="4p,maroon")


    # Plot Caldera Outline
    if caldera_file != None:
        fig.plot(data=caldera_file, pen="2p,black")

    # Plot CR
    if CR_dim != None:
        # fig.plot(x=CR_center[0], y=CR_center[1], style="e"+str(CR_angle)+"/"+str(CR_dim[0])+"/"+str(CR_dim[1]), pen="2p,black,4_2:2p")
        fig.plot(x=CR_center[0], y=CR_center[1], style="e"+str(CR_angle)+"/"+str(CR_dim[0])+"/"+str(CR_dim[1]), pen="4p,maroon,4_2:2p")

    # Plot inner CR if requested
    if inner_CR_dim != None:
        fig.plot(x=inner_CR_center[0], y=inner_CR_center[1], style="e"+str(inner_CR_angle)+"/"+str(inner_CR_dim[0])+"/"+str(inner_CR_dim[1]), pen="3p,maroon,-.")
    
    # Plot stations
    if plot_stations:
        lat = []
        long = []
        name = []
        lat_in = []
        long_in = []
        name_in = []
        lat_out = []
        long_out = []
        name_out = []
        for station in station_list:
            if station.name not in in_stations and station.name not in out_stations:
                lat.append(station.lat)
                long.append(station.long)
                name.append(station.name)
            elif station.name in in_stations:
                lat_in.append(station.lat)
                long_in.append(station.long)
                name_in.append(station.name)
            else:
                lat_out.append(station.lat)
                long_out.append(station.long)
                name_out.append(station.name)

        # plot normal stations
        fig.plot(x=long, y=lat,color="gray", style="c0.35c", pen='black')
        # fig.text(text=name, x=long, y=np.asarray(lat)+0.06, font="8p,Helvetica-Bold", fill="white", transparency=30) # optional label for stations we're not using

        # plot in stations
        fig.plot(x=long_in, y=lat_in,color="purple1", style="s0.35c", pen='black')
        # fig.text(text=name_in, x=long_in, y=np.asarray(lat_in)+0.06, font="8p,Helvetica-Bold", fill="white", transparency=30)

        # plot out stations
        fig.plot(x=long_out, y=lat_out,color="darkslategray2", style="s0.35c", pen='black')
        # fig.text(text=name_out, x=long_out, y=np.asarray(lat_out)+0.06, font="8p,Helvetica-Bold", fill="white", transparency=30)


    # save fig and show
    fig.savefig("location_map.pdf")

    fig.show()

# function to plot vs tomography map at a given depth with caldera and source
# Takes a file path for the tomography data (seismic_file), depth slice to make the map (depth, km), a tuple for the source dimension [minor_axis, major_axis] 
# (source_dim, km), a tuple for source center [source_long, source_lat], (source center, deg), an angle for the source (source_angle, deg), 
# and all three of those arguments for the CR (CR_dim), (CR_center), (CR_angle). Also takes optional file path to plot the caldera (caldera_file) and the older calderas
# (add_caldera_file). Takes optional list of bounds [min_long, max_long, min_lat, max_lat] (bounds, det) as well as an optional reference value to normalize the tomography,
# (ref, km/s). If ref is not passed the tomography will be in absolute units of km/s. If ref is passed as -1 the tomography will be normalized by the average speed at the 
# depth.
def make_vs_map(seismic_file, depth=10, source_dim=None, source_center=None, source_angle=None, CR_dim=None, CR_center=None, CR_angle=None,
                      inner_CR_dim=None, inner_CR_center=None, inner_CR_angle=None, caldera_file=None, add_caldera_file=None, bounds=None, 
                      ref=None, plot_stations=False, station_list=None, in_stations=[], out_stations=[], other_stations=[], plot_profiles=False):
    
    
    # Read the data from the tomography file
    vs_df = pd.read_csv(seismic_file, sep=' ', header=None, index_col=False)
    vs_depth_cut = vs_df[vs_df[2]==depth]
    vs_long = vs_depth_cut[0]
    vs_lat = vs_depth_cut[1]
    vs_val = vs_depth_cut[3]

    # use reference vs if provided
    if ref != None:
        if ref == -1: # if ref is -1, then average all vs at the depth and use that as the reference
            ref = np.mean(vs_val)
        vs_val_ref = 100*(vs_val-ref)/ref # in percent


    # If no bounds are provided then set the bounds to the max and min of the vs data.
    if bounds == None:
        region = [min(vs_long), max(vs_long), min(vs_lat), max(vs_lat)]
    else:
        region = bounds

    # Set up figure
    fig = pygmt.Figure()
    proj = 'M4i'
    if ref == None:
        vs_grid = pygmt.xyz2grd(x=vs_long, y=vs_lat, z=vs_val, region=region, spacing="0.1/0.1+e", projection=proj) # plot the tomography
        pygmt.makecpt(cmap='jet',series='2/4.75/0.0001',continuous=True, reverse=True)
    else:
        vs_grid = pygmt.xyz2grd(x=vs_long, y=vs_lat, z=vs_val_ref, region=region, spacing="0.1/0.1+e", projection=proj)
        pygmt.makecpt(cmap='jet',series='-33/15/0.0001',continuous=True, reverse=True)

    with pygmt.config(FONT="20p"):
        fig.grdimage(grid=vs_grid, region=region,projection=proj, frame=True)

    with pygmt.config(FONT="16p"):
        fig.basemap(map_scale="n0.75/0.95+w50k+f+u") # scale bar
    
    with pygmt.config(FONT="20p"):
        if bounds == None:
            fig.colorbar(frame=["a0.5f0.5", "x+lVs [km/s]"], position="JMR+o1c/0c+w7c/0.5c+n+mc") # plot the color bar
        else:
            fig.colorbar(frame=["a5f1"])

    # Plot caldera outline if a file was provided
    if caldera_file != None:
        fig.plot(data=caldera_file, pen="2p,black")

    # Plot older calderas if a file was provided
    if add_caldera_file != None:
        fig.plot(data=add_caldera_file, pen="2p,darkbrown")

    # Plot source if requested
    if source_dim != None:
        fig.plot(x=source_center[0], y=source_center[1], style="e"+str(source_angle)+"/"+str(source_dim[0])+"/"+str(source_dim[1]), pen="4p,white")


    # Plot CR if requested
    if CR_dim != None:
        fig.plot(x=CR_center[0], y=CR_center[1], style="e"+str(CR_angle)+"/"+str(CR_dim[0])+"/"+str(CR_dim[1]), pen="2p,black,4_2:2p")

    # Plot inner CR if requested
    if inner_CR_dim != None:
        fig.plot(x=inner_CR_center[0], y=inner_CR_center[1], style="e"+str(inner_CR_angle)+"/"+str(inner_CR_dim[0])+"/"+str(inner_CR_dim[1]), pen="2p,black,-.")

    # plot profiles if requested
    if plot_profiles:

        # make YY' line at source angle
        r = 1
        x_xprime = r*np.cos(source_angle)+source_center[0]
        y_xprime = r*np.sin(source_angle)+source_center[1]

        x_x = source_center[0]-r*np.cos(source_angle)
        y_x = source_center[1]-r*np.sin(source_angle)

        fig.plot(x=[x_x, x_xprime], y=(y_x, y_xprime), pen="2p,black")
        fig.text(x=x_x-0.04, y=y_x-0.01, text='Y', font="12p,Helvetica-Bold")
        fig.text(x=x_xprime+0.04, y=y_xprime+0.01, text="Y'", font="12p,Helvetica-Bold")

        # make XX'
        r=0.4
        x_yprime = r*np.cos(source_angle-(np.pi/2.))+source_center[0]+0.15
        y_yprime = r*np.sin(source_angle-(np.pi/2.))+source_center[1]

        x_y = source_center[0]-r*np.cos(source_angle-(np.pi/2.))-0.15
        y_y = source_center[1]-r*np.sin(source_angle-(np.pi/2.))

        fig.plot(x=[x_y, x_yprime], y=(y_y, y_yprime), pen="2p,black")
        fig.text(x=x_y-0.02, y=y_y+0.04, text='X', font="12p,Helvetica-Bold")
        fig.text(x=x_yprime+0.02, y=y_yprime-0.04, text="X'", font="12p,Helvetica-Bold")

     # Plot stations
    if plot_stations:
        lat = []
        long = []
        name = []
        lat_in = []
        long_in = []
        name_in = []
        lat_out = []
        long_out = []
        name_out = []
        lat_other = []
        long_other = []
        for station in station_list:
            if station.name not in in_stations and station.name not in out_stations and station.name not in other_stations:
                lat.append(station.lat)
                long.append(station.long)
                name.append(station.name)
            elif station.name in in_stations:
                lat_in.append(station.lat)
                long_in.append(station.long)
                name_in.append(station.name)
            elif station.name in out_stations:
                lat_out.append(station.lat)
                long_out.append(station.long)
                name_out.append(station.name)
            else: 
                lat_other.append(station.lat)
                long_other.append(station.long)

        # plot in stations
        fig.plot(x=long_in, y=lat_in,color="purple1", style="s0.35c", pen='1.5,black')
        # fig.text(text=name_in, x=long_in, y=np.asarray(lat_in)+0.06, font="8p,Helvetica-Bold", fill="white", transparency=30)

        # plot out stations
        fig.plot(x=long_out, y=lat_out,color="darkslategray2", style="s0.35c", pen='1.5,black')
        # fig.text(text=name_out, x=long_out, y=np.asarray(lat_out)+0.06, font="8p,Helvetica-Bold", fill="white", transparency=30)

        # plot other stations
        fig.plot(x=long_other, y=lat_other,color="gray", style="c0.35c", pen='1.5,black')


    # save fig and show
    if ref == None:
        fig.savefig("vs_map_depth="+str(depth)+".pdf")
    else:
        fig.savefig("vs_map_ref_depth="+str(depth)+".pdf")


    fig.show()


# Function to plot map view of GPS stations, colored by in- or out- of caldera group with arrows corresponding to vertical velocity.
# Takes a full list of ReadGPS objects (station_list), time span tuple organized by [start_time, end_time] (time_span), a list of station names inside the caldera
# (in_stations), a list of station names outside the caldera (out_stations), an optional tuple for CR dimensions [minor_axis, major_axis] (CR_dim, km), 
# optional tuple for CR center [longitude, latitude] (CR_center, degrees), optional CR angle (CR_angle), optional file paths to plot the caldera and faults (caldera_file),
# (fault_file), and an optional list of bounds [min_longitude, max_longitude, min_latitude, max_latitude] (bounds, degrees). If the bounds are not provided the bounds are
# set to the max and min long and lat of all of the stations in the list +/- half a degree.
def plot_GPS_vert(station_list, time_span, in_stations=[], out_stations=[], plot_names=True,
                   CR_dim=None, CR_center=None, CR_angle=None, caldera_file=None, fault_file=None, bounds=None):

    # Find the max and min longitude and latitude in our stations
    data_min_lat = np.inf
    data_min_long = np.inf
    data_max_lat = -np.inf
    data_max_long = -np.inf
    for station in station_list:
        if station.lat > data_max_lat:
            data_max_lat = station.lat
        if station.lat < data_min_lat:
            data_min_lat = station.lat
        if station.long > data_max_long:
            data_max_long = station.long
        if station.long < data_min_long:
            data_min_long = station.long
    
    # set up for PyGMT
    topo_data = '@earth_relief_15s' #15 arc second global relief (SRTM15+V2.1)
    pygmt.makecpt(cmap='topo',series='-8000/8000/1000',continuous=True)

    # set plot bounds, if none are provided then set them to be +/- 0.5 degrees from the max/min longitude and latitude
    if bounds == None:
        region = [
            data_min_long - 0.5,
            data_max_long + 0.5,
            data_min_lat - 0.5,
            data_max_lat + 0.5,
        ]
    else:
        region=bounds


    fig = pygmt.Figure()
    proj = 'M4i'
    with pygmt.config(FONT="20p"): #set fontsize for the longitude and latitude markers
        fig.grdimage(grid=topo_data, shading=True, region=region,projection=proj, frame=True)
        fig.coast(region=region,projection=proj, frame=True, water="lightblue", rivers="lightblue")

    # plot the CR if it's provided
    if CR_dim != None:
        fig.plot(x=CR_center[0], y=CR_center[1], style="e"+str(CR_angle)+"/"+str(CR_dim[0])+"/"+str(CR_dim[1]), pen="2p,black")

    # Plot caldera outline if a file is provided
    if caldera_file != None:
        fig.plot(data=caldera_file, pen="2p,black")

    # Plot faults if a file is provided
    if fault_file != None:
        fig.plot(data=fault_file, pen="1p,maroon3")

    # collect velocities, positions and names for stations in the caldera, out of the caldera and stations not being used in the mean time series plot.
    # We need lists for all three groups so they can be plotted separately.
    ve = []
    vn = []
    lat = []
    long = []
    name = []
    lat_nodata = []
    long_nodata = []
    name_nodata = []
    ve_in = []
    vn_in = []
    lat_in = []
    long_in = []
    name_in = []
    lat_in_nodata = []
    long_in_nodata = []
    name_in_nodata = []
    ve_out = []
    vn_out = []
    lat_out = []
    long_out = []
    name_out = []
    lat_out_nodata = []
    long_out_nodata = []
    name_out_nodata = []
    scaling = 0.175e3 # scale for velocity arrows
    for station in station_list:
        vel_up = velocity_metric(station, dir=Direction.UP, time_span=time_span) # get mean velocity for station over the time span.
        
        if station.name not in in_stations and station.name not in out_stations: # stations not being used for the mean time series plot
            if np.isnan(vel_up):
                lat_nodata.append(station.lat)
                long_nodata.append(station.long)
                name_nodata.append(station.name)
            else:
                ve.append(0) # no east (left/right) component for the arrows
                vn.append(vel_up*scaling) # north component (up/down) given by the vertical velocity
                lat.append(station.lat)
                long.append(station.long)
                name.append(station.name)
        elif station.name in in_stations: # station in the caldera
            if np.isnan(vel_up):
                lat_in_nodata.append(station.lat)
                long_in_nodata.append(station.long)
                name_in_nodata.append(station.name)
            else:
                ve_in.append(0)
                vn_in.append(vel_up*scaling)
                lat_in.append(station.lat)
                long_in.append(station.long)
                name_in.append(station.name)
        else:
            if np.isnan(vel_up):
                lat_out_nodata.append(station.lat)
                long_out_nodata.append(station.long)
                name_out_nodata.append(station.name)
            else:
                ve_out.append(0)
                vn_out.append(vel_up*scaling)
                lat_out.append(station.lat)
                long_out.append(station.long)
                name_out.append(station.name)


    # make data frames to store the velocity components for each set of stations.
    # This is needed for the PyGMT arrow plotting function.
    vh_df = pd.DataFrame(
        data={
            "x": long,
            "y": lat,
            "east_velocity": ve,
            "north_velocity": vn,
        }
    )

    vh_in_df = pd.DataFrame(
        data={
            "x": long_in,
            "y": lat_in,
            "east_velocity": ve_in,
            "north_velocity": vn_in,
        }
    )

    vh_out_df = pd.DataFrame(
        data={
            "x": long_out,
            "y": lat_out,
            "east_velocity": ve_out,
            "north_velocity": vn_out,
        }
    )

    # Make a special data frame to plot the scale arrow
    scale_x = [-111.54]
    scale_y = [44.01]
    scale = [0.02*scaling] #20 mm/yr in this scaling
    vh_scale_df = pd.DataFrame(
        data={
            "x": scale_x,
            "y": scale_y,
            "east_velocity": [0],
            "north_velocity": scale,
        }
    )

    # plot normal stations
    fig.plot(x=long, y=lat,color="gray", style="c0.4c", pen='black') # plot station location
    fig.velo(data=vh_df, line=True, pen='1.0p,black', spec="e0.5/0.39+f0", vector="0.5c+p5p+e+gblack") # plot arrow

    if len(long_nodata) > 0:
        fig.plot(x=long_nodata, y=lat_nodata,color="black", style="c0.4c", pen='black') 
    
    # plot in stations
    fig.plot(x=long_in, y=lat_in,color="purple", style="s0.4c", pen='black')
    fig.velo(data=vh_in_df, line=True, pen='1.0p,black', spec="e0.5/0.39+f0", vector="0.5c+p5p+e+gblack")

    if len(long_in_nodata) > 0:
        fig.plot(x=long_in_nodata, y=lat_in_nodata,color="black", style="s0.4c", pen='1p,purple')

    # plot out stations
    fig.plot(x=long_out, y=lat_out,color="darkslategray2", style="s0.4c", pen='black')
    fig.velo(data=vh_out_df, line=True, pen='1.0p,black', spec="e0.5/0.39+f0", vector="0.5c+p5p+e+gblack")

    if len(long_out_nodata) > 0:
        fig.plot(x=long_out_nodata, y=lat_out_nodata,color="black", style="s0.4c", pen='1p,darkslategray2')

    # plot scale arrow
    fig.velo(data=vh_scale_df, line=True, pen='1.0p,black', spec="e0.5/0.39+f0", vector="0.5c+p5p+e+gblack")
    fig.text(text="20 mm/yr", x=np.asarray(scale_x), y=np.asarray(scale_y)-0.03, font="12p,Helvetica-Bold", fill="white")

     # plot the text boxes separately so they can be moved around
    if plot_names:
        for i in range(len(name)):
            station_name = name[i]
            station_long = long[i]
            station_lat = lat[i]
            fig.text(text=station_name, x=station_long, y=station_lat+0.07, font="12p,Helvetica-Bold", fill="white", transparency=30)
        
        for i in range(len(name_nodata)):
            station_name = name_nodata[i]
            station_long = long_nodata[i]
            station_lat = lat_nodata[i]
            fig.text(text=station_name, x=station_long, y=station_lat+0.07, font="12p,Helvetica-Bold", fill="white", transparency=30)

        for i in range(len(name_in)):
            station_name = name_in[i]
            station_long = long_in[i]
            station_lat = lat_in[i]

            if station_name == 'LKWY' or station_name == 'P801' or station_name == 'P709':
                fig.text(text=station_name, x=station_long, y=station_lat-0.07, font="12p,Helvetica-Bold", fill="white", transparency=30)
            else:
                fig.text(text=station_name, x=station_long, y=station_lat+0.07, font="12p,Helvetica-Bold", fill="white", transparency=30)

        for i in range(len(name_in_nodata)):
            station_name = name_in_nodata[i]
            station_long = long_in_nodata[i]
            station_lat = lat_in_nodata[i]

            if station_name == 'LKWY' or station_name == 'P801' or station_name == 'P709':
                fig.text(text=station_name, x=station_long, y=station_lat-0.07, font="12p,Helvetica-Bold", fill="white", transparency=30)
            else:
                fig.text(text=station_name, x=station_long, y=station_lat+0.07, font="12p,Helvetica-Bold", fill="white", transparency=30)

        for i in range(len(name_out)):
            station_name = name_out[i]
            station_long = long_out[i]
            station_lat = lat_out[i]
            if station_name == 'MAWY' or station_name == 'P714':
                fig.text(text=station_name, x=station_long+0.2, y=station_lat, font="12p,Helvetica-Bold", fill="white", transparency=30)
            else:
                fig.text(text=station_name, x=station_long, y=station_lat+0.07, font="12p,Helvetica-Bold", fill="white", transparency=30)

        for i in range(len(name_out_nodata)):
            station_name = name_out_nodata[i]
            station_long = long_out_nodata[i]
            station_lat = lat_out_nodata[i]
            if station_name == 'MAWY' or station_name == 'P714':
                fig.text(text=station_name, x=station_long+0.2, y=station_lat, font="12p,Helvetica-Bold", fill="white", transparency=30)
            else:
                fig.text(text=station_name, x=station_long, y=station_lat+0.07, font="12p,Helvetica-Bold", fill="white", transparency=30)


    with pygmt.config(FONT="14p"):
        # https://www.pygmt.org/dev/gallery/embellishments/scalebar.html#sphx-glr-gallery-embellishments-scalebar-py reference for the scale bar
        fig.basemap(map_scale="n0.8/0.08+w50k+f+u")

    fig.savefig("GPS_vert"+str(time_span[0])+"-"+str(time_span[1])+"_pygmt.pdf")

    fig.show()

# Function to plot map view of GPS stations, colored by in- or out- of caldera group with arrows corresponding to horizontal velocity.
# Takes a full list of ReadGPS objects (station_list), time span tuple organized by [start_time, end_time] (time_span), a list of station names inside the caldera
# (in_stations), a list of station names outside the caldera (out_stations), an optional tuple for CR dimensions [minor_axis, major_axis] (CR_dim, km), 
# optional tuple for CR center [longitude, latitude] (CR_center, degrees), optional CR angle (CR_angle), optional file paths to plot the caldera and faults (caldera_file),
# (fault_file), and an optional list of bounds [min_longitude, max_longitude, min_latitude, max_latitude] (bounds, degrees). If the bounds are not provided the bounds are
# set to the max and min long and lat of all of the stations in the list +/- half a degree.
def plot_GPS_horiz(station_list, time_span, in_stations=[], out_stations=[], reference_station=None,
                   CR_dim=None, CR_center=None, CR_angle=None, caldera_file=None, fault_file=None, bounds=None, plot_names=True):

    # Find the max and min longitude and latitude in our stations
    data_min_lat = np.inf
    data_min_long = np.inf
    data_max_lat = -np.inf
    data_max_long = -np.inf
    for station in station_list:
        if station.lat > data_max_lat:
            data_max_lat = station.lat
        if station.lat < data_min_lat:
            data_min_lat = station.lat
        if station.long > data_max_long:
            data_max_long = station.long
        if station.long < data_min_long:
            data_min_long = station.long
    
    # set up for PyGMT
    topo_data = '@earth_relief_15s' #15 arc second global relief (SRTM15+V2.1)
    pygmt.makecpt(cmap='topo',series='-8000/8000/1000',continuous=True)

    # set plot bounds, if none are provided then set them to be +/- 0.5 degrees from the max/min longitude and latitude
    if bounds == None:
        region = [
            data_min_long - 0.5,
            data_max_long + 0.5,
            data_min_lat - 0.5,
            data_max_lat + 0.5,
        ]
    else:
        region=bounds


    fig = pygmt.Figure()
    proj = 'M4i'
    with pygmt.config(FONT="20p"): #set fontsize for the longitude and latitude markers
        fig.grdimage(grid=topo_data, shading=True, region=region,projection=proj, frame=True)
        fig.coast(region=region,projection=proj, frame=True, water="lightblue", rivers="lightblue")

    # plot the CR if it's provided
    if CR_dim != None:
        fig.plot(x=CR_center[0], y=CR_center[1], style="e"+str(CR_angle)+"/"+str(CR_dim[0])+"/"+str(CR_dim[1]), pen="2p,black")

    # Plot caldera outline if a file is provided
    if caldera_file != None:
        fig.plot(data=caldera_file, pen="2p,black")

    # Plot faults if a file is provided
    if fault_file != None:
        fig.plot(data=fault_file, pen="1p,maroon3")


    scaling = 0.35e3 # scale velocity arrows

    # get horizontal component shift from reference station
    if reference_station == None:
        e_shift = 0
        n_shift = 0
    elif reference_station == -1: # use average of in stations as reference
        in_station_subset = []
        long_list = []
        lat_list = []
        for station in station_list:
            if station.name in in_stations:
                in_station_subset.append(station)
                long_list.append(station.long)
                lat_list.append(station.lat)
        
        # get mean velocities in the different directions
        time, mean_vel_e = calc_subset_vel_mean(in_station_subset, dir=Direction.EASTING)
        time, mean_vel_n = calc_subset_vel_mean(in_station_subset, dir=Direction.NORTHING)

        # average the group velocity function over the time span
        calc_times_idx = (time >= time_span[0]) & (time <= time_span[1])

        e_shift = np.mean(mean_vel_e[calc_times_idx])*scaling
        n_shift = np.mean(mean_vel_n[calc_times_idx])*scaling
        ref_long = np.mean(np.asarray(long_list))
        ref_lat = np.mean(np.asarray(lat_list))

    else:
        for station in station_list:
            if station.name == reference_station:
                e_shift = velocity_metric(station, dir=Direction.EASTING, time_span=time_span)*scaling
                n_shift = velocity_metric(station, dir=Direction.NORTHING, time_span=time_span)*scaling
                break

    # collect velocities, positions and names for stations in the caldera, out of the caldera and stations not being used in the mean time series plot.
    # We need lists for all three groups so they can be plotted separately.
    ve = []
    vn = []
    lat = []
    long = []
    name = []
    lat_nodata = []
    long_nodata = []
    name_nodata = []
    ve_in = []
    vn_in = []
    lat_in = []
    long_in = []
    name_in = []
    lat_in_nodata = []
    long_in_nodata = []
    name_in_nodata = []
    ve_out = []
    vn_out = []
    lat_out = []
    long_out = []
    name_out = []
    lat_out_nodata = []
    long_out_nodata = []
    name_out_nodata = []
    for station in station_list:
        vel_east = velocity_metric(station, dir=Direction.EASTING, time_span=time_span) # get mean velocities for station over the time span.
        vel_north = velocity_metric(station, dir=Direction.NORTHING, time_span=time_span)
        
        if station.name not in in_stations and station.name not in out_stations: # stations not being used for the mean time series plot
            if np.isnan(vel_east):
                lat_nodata.append(station.lat)
                long_nodata.append(station.long)
                name_nodata.append(station.name)
            else:
                ve.append(vel_east*scaling-e_shift) 
                vn.append(vel_north*scaling-n_shift) 
                lat.append(station.lat)
                long.append(station.long)
                name.append(station.name)
        elif station.name in in_stations: # station in the caldera
            if np.isnan(vel_east):
                lat_in_nodata.append(station.lat)
                long_in_nodata.append(station.long)
                name_in_nodata.append(station.name)
            else:
                ve_in.append(vel_east*scaling-e_shift)
                vn_in.append(vel_north*scaling-n_shift)
                lat_in.append(station.lat)
                long_in.append(station.long)
                name_in.append(station.name)
        else:
            if np.isnan(vel_east):
                lat_out_nodata.append(station.lat)
                long_out_nodata.append(station.long)
                name_out_nodata.append(station.name)
            else:
                ve_out.append(vel_east*scaling-e_shift)
                vn_out.append(vel_north*scaling-n_shift)
                lat_out.append(station.lat)
                long_out.append(station.long)
                name_out.append(station.name)


    # make data frames to store the velocity components for each set of stations.
    # This is needed for the PyGMT arrow plotting function.
    vh_df = pd.DataFrame(
        data={
            "x": long,
            "y": lat,
            "east_velocity": ve,
            "north_velocity": vn,
        }
    )

    vh_in_df = pd.DataFrame(
        data={
            "x": long_in,
            "y": lat_in,
            "east_velocity": ve_in,
            "north_velocity": vn_in,
        }
    )

    vh_out_df = pd.DataFrame(
        data={
            "x": long_out,
            "y": lat_out,
            "east_velocity": ve_out,
            "north_velocity": vn_out,
        }
    )

    # Make a special data frame to plot the scale arrow
    scale_x = [-111.55]
    scale_y = [44.01]
    scale = [0.01*scaling] #10 mm/yr in this scaling
    vh_scale_df = pd.DataFrame(
        data={
            "x": scale_x,
            "y": scale_y,
            "east_velocity": [0],
            "north_velocity": scale,
        }
    )

    # plot normal stations
    fig.plot(x=long, y=lat,color="gray", style="c0.4c", pen='black') # plot station location
    fig.velo(data=vh_df, line=True, pen='1.0p,black', spec="e0.5/0.39+f0", vector="0.5c+p5p+e+gblack") # plot arrow

    if len(long_nodata) > 0:
        fig.plot(x=long_nodata, y=lat_nodata,color="black", style="c0.4c", pen='black') 
    
    # plot in stations
    fig.plot(x=long_in, y=lat_in,color="purple", style="s0.4c", pen='black')
    fig.velo(data=vh_in_df, line=True, pen='1.0p,black', spec="e0.5/0.39+f0", vector="0.5c+p5p+e+gblack")

    if len(long_in_nodata) > 0:
        fig.plot(x=long_in_nodata, y=lat_in_nodata,color="black", style="s0.4c", pen='1p,purple')

    # plot out stations
    fig.plot(x=long_out, y=lat_out,color="darkslategray2", style="s0.4c", pen='black')
    fig.velo(data=vh_out_df, line=True, pen='1.0p,black', spec="e0.5/0.39+f0", vector="0.5c+p5p+e+gblack")

    if len(long_out_nodata) > 0:
        fig.plot(x=long_out_nodata, y=lat_out_nodata,color="black", style="s0.4c", pen='1p,darkslategray2')

    # plot scale arrow
    fig.velo(data=vh_scale_df, line=True, pen='1.0p,black', spec="e0.5/0.39+f0", vector="0.5c+p5p+e+gblack")
    fig.text(text="10 mm/yr", x=np.asarray(scale_x), y=np.asarray(scale_y)-0.03, font="12p,Helvetica-Bold", fill="white")

     # plot the text boxes separately so they can be moved around
    if plot_names:
        for i in range(len(name)):
            station_name = name[i]
            station_long = long[i]
            station_lat = lat[i]
            fig.text(text=station_name, x=station_long, y=station_lat+0.07, font="12p,Helvetica-Bold", fill="white", transparency=30)
        
        for i in range(len(name_nodata)):
            station_name = name_nodata[i]
            station_long = long_nodata[i]
            station_lat = lat_nodata[i]
            fig.text(text=station_name, x=station_long, y=station_lat+0.07, font="12p,Helvetica-Bold", fill="white", transparency=30)

        for i in range(len(name_in)):
            station_name = name_in[i]
            station_long = long_in[i]
            station_lat = lat_in[i]

            if station_name == 'LKWY' or station_name == 'P801' or station_name == 'P709':
                fig.text(text=station_name, x=station_long, y=station_lat-0.07, font="12p,Helvetica-Bold", fill="white", transparency=30)
            else:
                fig.text(text=station_name, x=station_long, y=station_lat+0.07, font="12p,Helvetica-Bold", fill="white", transparency=30)

        for i in range(len(name_in_nodata)):
            station_name = name_in_nodata[i]
            station_long = long_in_nodata[i]
            station_lat = lat_in_nodata[i]

            if station_name == 'LKWY' or station_name == 'P801' or station_name == 'P709':
                fig.text(text=station_name, x=station_long, y=station_lat-0.07, font="12p,Helvetica-Bold", fill="white", transparency=30)
            else:
                fig.text(text=station_name, x=station_long, y=station_lat+0.07, font="12p,Helvetica-Bold", fill="white", transparency=30)

        for i in range(len(name_out)):
            station_name = name_out[i]
            station_long = long_out[i]
            station_lat = lat_out[i]
            if station_name == 'MAWY' or station_name == 'P714':
                fig.text(text=station_name, x=station_long+0.2, y=station_lat, font="12p,Helvetica-Bold", fill="white", transparency=30)
            else:
                fig.text(text=station_name, x=station_long, y=station_lat+0.07, font="12p,Helvetica-Bold", fill="white", transparency=30)

        for i in range(len(name_out_nodata)):
            station_name = name_out_nodata[i]
            station_long = long_out_nodata[i]
            station_lat = lat_out_nodata[i]
            if station_name == 'MAWY' or station_name == 'P714':
                fig.text(text=station_name, x=station_long+0.2, y=station_lat, font="12p,Helvetica-Bold", fill="white", transparency=30)
            else:
                fig.text(text=station_name, x=station_long, y=station_lat+0.07, font="12p,Helvetica-Bold", fill="white", transparency=30)

    # if the average reference frame is used plot its centroid
    if reference_station == -1:
        fig.plot(x=ref_long, y=ref_lat, color="yellow", style="a0.5c", pen="2p,black")

    # # plot scale arrow
    # fig.velo(data=vh_scale_df, line=True, pen='1.0p,black', spec="e0.5/0.39+f0", vector="0.5c+p5p+e+gblack")
    # fig.text(text="10 mm/yr", x=scale_x, y=np.asarray(scale_y)-0.03, font="8p,Helvetica-Bold", fill="white")

    with pygmt.config(FONT="14p"):
        # https://www.pygmt.org/dev/gallery/embellishments/scalebar.html#sphx-glr-gallery-embellishments-scalebar-py reference for the scale bar
        fig.basemap(map_scale="n0.8/0.08+w50k+f+u")

    fig.savefig("GPS_horiz"+str(time_span[0])+"-"+str(time_span[1])+"_pygmt.pdf")

    fig.show()

# Takes in list of stations, a tuple for the geometry of the subplot, component direction (default is UP), and 
# optional bounds for the displacement and velocities it plots across all subplots.
# Plots raw displacements, smoothed displacements, smoothed velocities and their errors or all stations in list.
def plot_all_stations_vel(station_list, subplot_geometry, dir=Direction.UP, disp_bounds=None, vel_bounds=None):
     # get min and max times and velocities for all the stations
    min_time = np.inf
    max_time = -1
    min_vel = np.inf
    max_vel = -np.inf
    min_disp = np.inf
    max_disp = -np.inf
    for station in station_list:
        smoothed_times, component = station.calculate_velocity(dir=dir)
        smoothed_times_disp, smoothed_disp = station.smooth_component(dir=dir)

        if min(station.dec_years) < min_time:
            min_time = min(station.dec_years)
        if max(station.dec_years) > max_time:
            max_time = max(station.dec_years)

        if min(component) < min_vel:
            min_vel = min(component)
        if max(component) > max_vel:
            max_vel = max(component)

        if min(smoothed_disp) < min_disp:
            min_disp = min(smoothed_disp)
        if max(smoothed_disp) > max_disp:
            max_disp = max(smoothed_disp)



    # get rows and columns for subplots
    rows = subplot_geometry[0]
    cols = subplot_geometry[1]

    # declare figure
    fig, ax = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True)

    # error scale factor
    error_mult_factor = 5
    
    # iterate through rows and columns
    curr_station_iter = 0
    curr_col = 0
    curr_row = 0
    for r in ax:
        for c in r:
            station = station_list[curr_station_iter]
            # determine plot colors
            if station.name in ['LKWY', 'HVWY', 'OFW2', 'WLWY', 'P709', 'P801']:
                plot_color = 'indigo'
            elif station.name in ['P361','P676', 'P456', 'P712', 'P460', 'P457', 'P461', 'P721', 'P720', 'P718', 'P710']:
                plot_color = 'black'
            else:
                plot_color = 'teal'

            # get station data
            times = station.dec_years
            station_name = station.name
            if dir == Direction.EASTING:
                label2 = "Easting velocity [m/yr]"
                label1 = "Easting displacement [m]"
                disp = station.easting
                save="easting"
            elif dir == Direction.NORTHING:
                label2 = "Northing velocity [m/yr]"
                label1 = "Northing displacement [m]"
                disp = station.northing
                save = "northing"
            else:
                label2 = "Vertical velocity [m/yr]"
                label1 = "Vertical displacement [m]"
                disp = station.up
                save = "up"

            # get smoothed displacement
            smoothed_times_disp, smoothed_disp, smoothed_disp_error = station.smooth_component(dir=dir, return_errors=True)
            
            # get smoothed velocity
            smoothed_times_vel, vel, smoothed_vel_error = station.calculate_velocity(dir=dir, return_errors=True)

            # plot
            c2 = c.twinx()

            # deal with discontinuity in P801
            if station.name == 'P801':

                # plot velocity errors
                l5 = c2.fill_between(smoothed_times_vel[smoothed_times_vel <= P801_pre_quake], 
                                vel[smoothed_times_vel <= P801_pre_quake]-smoothed_vel_error[smoothed_times_vel <= P801_pre_quake]*error_mult_factor, 
                                vel[smoothed_times_vel <= P801_pre_quake]+smoothed_vel_error[smoothed_times_vel <= P801_pre_quake]*error_mult_factor, alpha = 0.25, color='red',
                                label="smoothed velocity error x5")
                c2.plot(smoothed_times_vel[smoothed_times_vel <= P801_pre_quake], 
                        vel[smoothed_times_vel <= P801_pre_quake]-smoothed_vel_error[smoothed_times_vel <= P801_pre_quake]*error_mult_factor, linestyle='dotted', color='red')
                c2.plot(smoothed_times_vel[smoothed_times_vel <= P801_pre_quake], 
                        vel[smoothed_times_vel <= P801_pre_quake]+smoothed_vel_error[smoothed_times_vel <= P801_pre_quake]*error_mult_factor, linestyle='dotted', color='red')
                
                c2.fill_between(smoothed_times_vel[smoothed_times_vel >= P801_post_quake], 
                                vel[smoothed_times_vel >= P801_post_quake]-smoothed_vel_error[smoothed_times_vel >= P801_post_quake]*error_mult_factor, 
                                vel[smoothed_times_vel >= P801_post_quake]+smoothed_vel_error[smoothed_times_vel >= P801_post_quake]*error_mult_factor, alpha = 0.25, color='red',
                                label="smoothed velocity error x5")
                c2.plot(smoothed_times_vel[smoothed_times_vel >= P801_post_quake], 
                        vel[smoothed_times_vel >= P801_post_quake]-smoothed_vel_error[smoothed_times_vel >= P801_post_quake]*error_mult_factor, linestyle='dotted', color='red')
                c2.plot(smoothed_times_vel[smoothed_times_vel >= P801_post_quake], 
                        vel[smoothed_times_vel >= P801_post_quake]+smoothed_vel_error[smoothed_times_vel >= P801_post_quake]*error_mult_factor, linestyle='dotted', color='red')

                # plot velocities
                l2, = c2.plot(smoothed_times_vel[smoothed_times_vel <= P801_pre_quake], vel[smoothed_times_vel <= P801_pre_quake], lw=3, c=plot_color, linestyle='dashed', label='smoothed velocity', zorder=2)
                c2.plot(smoothed_times_vel[smoothed_times_vel >= P801_post_quake], vel[smoothed_times_vel >= P801_post_quake], lw=3, c=plot_color, linestyle='dashed', zorder=2)

                # plot displacement error lines
                l4 = c.fill_between(smoothed_times_disp, smoothed_disp-np.mean(smoothed_disp)-smoothed_disp_error*error_mult_factor, smoothed_disp-np.mean(smoothed_disp)+smoothed_disp_error*error_mult_factor,
                               alpha = 0.25, color='brown', label="smoothed displacement error x5", zorder=4)
                c.plot(smoothed_times_disp, smoothed_disp-np.mean(smoothed_disp)-smoothed_disp_error*error_mult_factor, linestyle='dotted', color='brown')
                c.plot(smoothed_times_disp, smoothed_disp-np.mean(smoothed_disp)+smoothed_disp_error*error_mult_factor, linestyle='dotted', color='brown')

                # plot displacements
                l3 = c.scatter(times, disp-np.mean(disp), s=20, c='gray', label='data', alpha=0.7)
                l1, = c.plot(smoothed_times_disp, smoothed_disp-np.mean(smoothed_disp), c=plot_color, label='smoothed displacement', lw=3)
               
            else:
                # plot data
                l3 = c.scatter(times, disp-np.mean(disp), s=20, c='gray', label='data', zorder=2, alpha=0.7)

                # plot displacement error lines
                l4 = c.fill_between(smoothed_times_disp, smoothed_disp-np.mean(smoothed_disp)-smoothed_disp_error*error_mult_factor, smoothed_disp-np.mean(smoothed_disp)+smoothed_disp_error*error_mult_factor,
                               alpha = 0.25, color='brown', label="smoothed displacement error x5", zorder=4)
                c.plot(smoothed_times_disp, smoothed_disp-np.mean(smoothed_disp)-smoothed_disp_error*error_mult_factor, linestyle='dotted', color='brown')
                c.plot(smoothed_times_disp, smoothed_disp-np.mean(smoothed_disp)+smoothed_disp_error*error_mult_factor, linestyle='dotted', color='brown')

                # plot smoothed disp
                l1, = c.plot(smoothed_times_disp, smoothed_disp-np.mean(smoothed_disp), lw=3, c=plot_color, label='smoothed data', zorder=4)

                # plot velocity error lines
                l5 = c2.fill_between(smoothed_times_vel, vel-smoothed_vel_error*error_mult_factor, vel+smoothed_vel_error*error_mult_factor, alpha = 0.25, color='red', label="smoothed velocity error x5")
                c2.plot(smoothed_times_vel, vel-smoothed_vel_error*error_mult_factor, linestyle='dotted', color='red')
                c2.plot(smoothed_times_vel, vel+smoothed_vel_error*error_mult_factor, linestyle='dotted', color='red')
                # plot velocity
                l2, = c2.plot(smoothed_times_vel, vel, lw=3, c=plot_color, linestyle='dashed', label='smoothed velocity', zorder=5)

            # format panels
            c.text(0.1, 0.9, station_name, horizontalalignment='center', verticalalignment='center', transform=c.transAxes, color=plot_color, fontsize=25)
        
            c.tick_params(axis='x', labelsize=20)
            c.tick_params(axis='y', labelsize=20)
            c2.tick_params(axis='x', labelsize=20)
            c2.tick_params(axis='y', labelsize=20)

            c.set_xlim([min_time-0.5, max_time+0.5])
            if disp_bounds != None:
                c.set_ylim(disp_bounds)
            else:
                c.set_ylim([-0.15, 0.15])
            if vel_bounds != None:
                c2.set_ylim(vel_bounds)
            else:
                c2.set_ylim([-0.03, 0.065])
            
            c.grid()
            c2.set_axisbelow(True)


            if curr_row < subplot_geometry[0]-1 and subplot_geometry[0] == 5:
                c.xaxis.set_visible(False)

            if curr_col < subplot_geometry[1]-1:
                c2.get_yaxis().set_visible(False)


            # iterate station
            curr_station_iter += 1
            curr_col += 1
        curr_row += 1
        curr_col = 0

    fig.legend((l1, l2, l3, l4, l5), (l1.get_label(), l2.get_label(), l3.get_label(), l4.get_label(), l5.get_label()), bbox_to_anchor=(2.5, 1.1), fontsize=25)

    fig.text(0.5, 0.01, 'Time [yrs]', ha='center', fontsize=25)
    fig.text(0.01, 0.5, label1, va='center', rotation='vertical', fontsize=25)
    fig.text(0.98, 0.5, label2, va='center', rotation='vertical', fontsize=25)

    plt.subplots_adjust(wspace=0, hspace=0)

    fig_path = "/home/grantblock/Research/Yellowstone/Figures/"
    plt.savefig(fig_path+"all_stations_vel"+save+".png", bbox_inches="tight")  
    plt.show()

# function to derive a timedb file for PyLith based on the in-caldera mean velocity.
# Takes a list of ReadGPS objects of all the stations in the caldera (in_station_subset_list), a maximum pressure slope (dpdt_max, kPa/yr),
# a name for the timedb file (run_name), a direction (dir, default is Direction.UP), a time step for the timedb file (dt, default is 0.005, yrs),
# a tuple containing start and end times of the slope in the velocity function used to scale with pressure (time_scale, yrs), 
# a background pressure (P0, kPa, default 100).
# The function will write a .timedb file for PyLith and plot the generated pressure function
def make_center_mean_timedb(in_station_subset_list, dpdt_max, run_name, dir=Direction.UP, dt=0.005, time_scale=[2003, 2006], P0=100):

    # get mean velocity function
    mean_time, mean_vel = calc_subset_vel_mean(in_station_subset_list, dir=dir)
   

    # smooth the velocity
    smoothed_vel = []
    smoothed_times = []

    step = 1 # step in years
    window = 3  # window size in years

    # step = 0.5 # step in years
    # window = 1  # window size in years

    for t in np.arange(mean_time[0]+(window/2), mean_time[len(mean_time)-1]-(window/2)+step, step):

        # calculate average in window
        comp_avg = 0
        time_avg = 0
        t_avg_idx = (mean_time >= t-(window/2)) & (mean_time < t+(window/2))
        for j in mean_vel[t_avg_idx]:
            comp_avg += j
        for k in mean_time[t_avg_idx]:
            time_avg += k
        
        smoothed_times.append(time_avg/(len(mean_time[t_avg_idx])))
        smoothed_vel.append(comp_avg/(len(mean_vel[t_avg_idx])))

    # Smoothing of <V>^inner is turned off when these lines are un-commented
    smoothed_vel = mean_vel
    smoothed_times = mean_time
    #############################################

    # scale smoothed velocity using dp/dt
    m_idx1 = np.argmin(np.abs(np.asarray(smoothed_times)-time_scale[1]))
    m_idx0 = np.argmin(np.abs(np.asarray(smoothed_times)-time_scale[0]))
    current_slope=(smoothed_vel[m_idx1]-smoothed_vel[m_idx0])/(time_scale[1]-time_scale[0])
    
    shifted_pressure = np.asarray(smoothed_vel)*(dpdt_max/current_slope)

    # interpolate pressure function so it will have the dt needed by the timedb file
    interp_times_full = np.arange(smoothed_times[0], smoothed_times[len(smoothed_times)-1], dt)
    shifted_pressure_interp_full = np.interp(interp_times_full, smoothed_times, shifted_pressure)

    interp_times = interp_times_full[interp_times_full >= 2002.6]
    shifted_pressure_interp = shifted_pressure_interp_full[interp_times_full >= 2002.6]

    # calculate initial depressurization step used before the main pressure function
    start_time = 1986
    depress_time = interp_times[0]-start_time # 1986-2002.xx, from Dzurisin et al. Figure 19
    depress_dpdt = -dpdt_max/10 # set the depressurization dpdt to be a tenth the max dpdt
    repress_dpdt = (shifted_pressure_interp[0]-(depress_dpdt*depress_time*0.9))/(depress_time*0.1) # slope to rise back to the initial derived pressure

    depress_times_array = np.arange(start_time, interp_times[0], dt)
    depress_pressures = []
    for t in depress_times_array:
        if t <= start_time + depress_time*0.9: # first 90% of the time depressurize
            depress_pressures.append(depress_dpdt*(t-start_time))
        elif t > start_time + depress_time*0.9 and t< interp_times[0]: # second half of the time rise back to the first derived pressure
            depress_pressures.append((depress_dpdt*depress_time*0.9)+repress_dpdt*(t-start_time-depress_time*0.9))
        else:
            print(t, shifted_pressure[0])
            depress_pressures.append(shifted_pressure_interp[0])


    # write the timedb file
    # note that pressures in the file are normalized by P0
    file_name = run_name + ".timedb"

    spinup_times = np.arange(0, 500, dt) # 500 yr spinup
    shifted_depress_times = depress_times_array - depress_times_array[0] + 500 + dt # depressurization times
    shifted_interp_times = interp_times - interp_times[0] + shifted_depress_times[len(shifted_depress_times)-1] + dt # pressure function times
    final_times = np.arange(shifted_interp_times[len(shifted_interp_times)-1]+dt, 100+shifted_interp_times[len(shifted_interp_times)-1]+2*dt,dt) # 100 yr constant pressure at end

    f = open(file_name, "w")
    f.write("#TIME HISTORY ascii\nTimeHistory {\nnum-points = " + str(len(spinup_times)+len(shifted_interp_times)+len(final_times)) + "\ntime-units = year\n}\n")
    for i in range(len(spinup_times)):
        f.write(str(round(spinup_times[i], 5))+" "+str(round(1, 5))+"\n") #write spinup

    for i in range(len(shifted_depress_times)):
        f.write(str(round(shifted_depress_times[i], 5)) + " " + str(round(1+depress_pressures[i]/P0, 5)) + "\n") # write depressurization phase
    
    for i in range(len(shifted_interp_times)):
        f.write(str(round(shifted_interp_times[i], 5)) + " " + str(round(1+shifted_pressure_interp[i]/P0, 5)) + "\n") # write pressure function
        
    for i in range(len(final_times)):
        f.write(str(round(final_times[i], 5))+" "+str(round(1+shifted_pressure_interp[len(shifted_pressure_interp)-1]/P0, 5)) + "\n") # write ending pressures
    f.close()

    # plot smoothed velocity and derived pressure function
    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Time (yrs)", fontsize=28)
    ax1.set_ylabel("Velocity (mm/yr)", fontsize=28)
    ax1.tick_params(axis='x', labelsize=30)
    ax1.tick_params(axis='y', labelsize=30)
    ax1.grid()
    l1, = ax1.plot(mean_time, np.asarray(mean_vel)*1e3, lw=20, color="indigo", alpha=0.6, label=r"$\langle V(t)\rangle^{inner}$")

   
    ax2 = ax1.twinx()

    ax2.set_ylabel("P (kPa)", fontsize=28)
    ax2.tick_params(axis='y', labelsize=30)
    ax2.plot(depress_times_array, np.asarray(depress_pressures)+P0, lw=10, color="black")
    ax2.plot(np.arange(depress_times_array[0]-10, depress_times_array[0], 1), np.ones(10)*P0, lw=10, color="black", linestyle='dashed')
    l2, = ax2.plot(interp_times, shifted_pressure_interp+P0, lw=10, color="black", label=r"$P(t)$")


    plt.legend((l1, l2), (l1.get_label(), l2.get_label()), fontsize=30)

    plt.show()


# Function to take multiple generated timedb files and plot them together, comparing to the data time series
# Takes in timedb_list: list of filename strings, dpdt_max_list: list of dP/dt max values (floats), in_station_subset_list: list of inner ReadGPS objects, 
# optional color_list: list of strings specifying plot color, optional line_list: list of strings specifying plot linestyles, dir: component, UP by default, 
# and P0: background pressure, 100 kPa by default.
def plot_compare_timedbs(timedb_list, dpdt_max_list, in_station_subset_list, line_list=['solid', 'dashdot'], dir=Direction.UP, P0=100):

    # get data velocity
    mean_time, mean_vel = calc_subset_vel_mean(in_station_subset_list, dir=dir)


    fig, ax1 = plt.subplots()

    # plot velocity data
    ax1.set_xlabel("Time (yrs)", fontsize=28)
    ax1.set_ylabel("Velocity (mm/yr)", fontsize=28)
    ax1.tick_params(axis='x', labelsize=30)
    ax1.tick_params(axis='y', labelsize=30)
    ax1.grid()
    l2, = ax1.plot(mean_time, mean_vel*1e3, lw=20, color="indigo", alpha=0.6, linestyle='dashdot', label=r"$\langle V(t)\rangle^{inner}$")

    legend_list = [l2]
    label_list = [l2.get_label()]

    # set up P(t) axis
    ax2 = ax1.twinx()
    ax2.set_ylabel(r"$P-P_0$ (kPa)", fontsize=28)
    ax2.tick_params(axis='y', labelsize=30)

    # loop through timedb files
    start_time=1986
    itr = 0
    for file in timedb_list:
        dpdt_max = dpdt_max_list[itr]
        df = pd.read_csv(file, sep=' ', names=['times', 'pressures'], skiprows=[0,1,2,3,4])
        pressure_times = df['times'] + start_time - 500 # align with data
        pressure_values = df['pressures']

        ax2.plot(pressure_times[(pressure_times >= start_time-10) & (pressure_times <= start_time)], pressure_values[(pressure_times >= start_time-10) & (pressure_times <= start_time)]*P0 - P0,
                 lw=7, color="black", linestyle='dashed')
        
        l, = ax2.plot(pressure_times[(pressure_times >= start_time) & (pressure_times <= start_time+40)], pressure_values[(pressure_times >= start_time) & (pressure_times <= start_time+40)]*P0 - P0,
                      lw=7, color='black', linestyle=line_list[itr], label=r"$P(t), \frac{\Delta P}{\Delta t}_{max}=$"+str(dpdt_max)+" kPa/yr")

        legend_list.append(l)
        label_list.append(l.get_label())
        
        itr += 1
    
    # Function modified from https://stackoverflow.com/questions/64087309/how-to-set-the-axis-of-two-y-axis-plots-using-the-same-starting-point-instead-of
    def alignZero(ax1, ax2):
        y1Min, y1Max = ax1.get_ylim()
        y2Min, y2Max = ax2.get_ylim()

        ratio1 = -y1Min / (y1Max - y1Min)
        ratio2 = -y2Min / (y2Max - y2Min)
        if ratio2 < ratio1:
            y2newMin = y1Min * y2Max / y1Max
            ax2.set_ylim(y2newMin, y2Max)
        elif ratio2 > ratio1:
            y2newMax = y2Min * y1Max / y1Min
            ax2.set_ylim(y2Min, y2newMax)
        
    alignZero(ax1, ax2)
    
    plt.legend(legend_list, label_list, fontsize=30, bbox_to_anchor=(1.35, 1.0), loc='upper left')
    plt.savefig("/home/grantblock/Research/Yellowstone/Figures/Compare_pressure_funcs.png", bbox_inches="tight")

    plt.show()

# function to calculate the distance (in km) from the center of the source to each station in the list
# Takes station_list: list of ReadGPS objects, source_center: (long, lat) location of the center of the source
# Returns a list of distance tuples (x, y) in km between the stations in the list and the source center
def distance_to_stations(station_list, source_center):

    # long, lat of tip of the caldera/source
    caldera_tip = (-110.28, 44.69) #deg

    # known distance from the center of the model source to the major axis tip
    scale_distance = 27.5 #km

    # angle the coord system is shifted
    shift_angle = (90-58)*(np.pi/180) #rad

    # scale long, lat to km (we assume short enough distances)
    scale_x = scale_distance*np.cos(shift_angle)
    scale_y = scale_distance*np.sin(shift_angle)

    long_to_km = scale_x/(caldera_tip[0]-source_center[0])
    lat_to_km = scale_y/(caldera_tip[1]-source_center[1])
    # print(long_to_km, lat_to_km)

    distance_list = []
    for station in station_list:

        # get x, y for each station
        dif_long = station.long-source_center[0]
        dif_lat = station.lat-source_center[1]

        station_x = dif_long*long_to_km
        station_y = dif_lat*lat_to_km

        # rotate to model coord axis
        rotated_x = station_x*np.cos((np.pi/2)-shift_angle)-station_y*np.sin((np.pi/2)-shift_angle)
        rotated_y = station_x*np.sin((np.pi/2)-shift_angle)+station_y*np.cos((np.pi/2)-shift_angle)

        # print(rotated_x, rotated_y)
        distance_list.append((rotated_x, rotated_y))

    return distance_list

# A function to calculate the cross correlations and distances of stations with a given station.
# Takes a center station (ReadGPS object), station_list (list of ReadGPS objects), optional time window (tuple of start and end times),
# and component direction (default is up.) 
# Prints the cross correlation, p-value and caldera distance for all stations in the list sorted from most negative to most positive correlation.
def calc_cross_correlations(center_station_l, station_list, time_window=None, dir=Direction.UP):

    from scipy.stats import pearsonr

    # get center station times and velocity
    center_station = center_station_l[0]
    center_times, center_vel = center_station.calculate_velocity(dir=dir)

    in_group = [center_station.name]
    out_group = []

    display_list = []

    # iterate through other stations and calculate CC
    for station in station_list:

        times, vel = station.calculate_velocity(dir=dir)

        # find min and max time both stations have data rounded to the nearest half year
        min_time = math.ceil(max(min(times), min(center_times))*5)/5
        max_time = math.floor(min(max(times), max(center_times))*5)/5
        if time_window != None:
            if time_window[0] >= min_time:
                min_time = time_window[0]
            if time_window[1] <= max_time:
                max_time = time_window[1]

        # get data cuts based on max and min times
        center_times_cut = (center_times > min_time+0.1) & (center_times < max_time-0.1)
        times_cut = (times > min_time+0.1) & (times < max_time-0.1)

        center_vel_comp = center_vel[center_times_cut]
        vel_comp = vel[times_cut]

        # deal with P801 discontinuity
        if station.name != 'P801':
            if len(center_vel_comp) >= 2:

                # calculate cross correlation, p-value with scipy pearsonr() function
                result = pearsonr(center_vel_comp, vel_comp)
                # calculate distance of station to caldera
                dist = station_caldera_dist(get_data_list([station.name]), "GMT_data/YScalderaL.txt")

                display_list.append((station.name, result[0], result[1], dist))

                # sort into inner or outer groups based on if CC >= 0.2 or <= 0.2 and is significant (p <= 0.05)
                # To be in the outer group the station must be <= 30 km from the edge of the caldera.
                if station.name != 'LKWY':
                    if result[0] >= 0.2 and result[1] <= 0.05:
                        in_group.append(station.name)
                    if result[0] <= -0.2 and result[1] <= 0.05 and dist <= 30:
                        out_group.append(station.name)
        else:
            # same is done as above but with the P801 discontinuity
            dist = station_caldera_dist(get_data_list([station.name]), "GMT_data/YScalderaL.txt")

            center_times_cut_pre = (center_times > min_time) & (center_times < P801_pre_quake)
            center_times_cut_post = (center_times > P801_post_quake) & (center_times < max_time)

            times_cut_pre = (times > min_time) & (times < P801_pre_quake)
            times_cut_post = (times > P801_post_quake) & (times < max_time)

            center_vel_comp_pre = center_vel[center_times_cut_pre]
            vel_comp_pre = vel[times_cut_pre]

            center_vel_comp_post = center_vel[center_times_cut_post]
            vel_comp_post = vel[times_cut_post]


            if len(vel_comp_pre) >= 2:
                result_pre = pearsonr(center_vel_comp_pre, vel_comp_pre)
                display_list.append((station.name, result_pre[0], result_pre[1], dist))
            
            if len(vel_comp_post) >= 2:
                result_post = pearsonr(center_vel_comp_post, vel_comp_post)
                display_list.append((station.name, result_post[0], result_post[1], dist))

            if len(vel_comp_pre) >= 2 and len(vel_comp_post) >= 2:
                if result_pre[1] <= result_post[1]:
                    P801_result = result_pre
                else:
                    P801_result = result_post
            elif len(vel_comp_pre) >= 2:
                P801_result = result_pre
            elif len(vel_comp_post) >= 2:
                P801_result = result_post
            
            if len(vel_comp_pre) >= 2 or len(vel_comp_post) >= 2:
                if P801_result[1] < 0.05:
                    if P801_result[0] > 0.02:
                        in_group.append('P801')
                    elif P801_result[0] < -0.02:
                        out_group.append('P801')


    # sort CC's for printing
    display_list_sorted = sorted(display_list, key=lambda x: x[1])
    for elm in display_list_sorted:
        print(elm[0], "CC="+str(elm[1]), "p value="+str(elm[2]), "Distance="+str(elm[3]))
    print("In Group:", in_group)
    print("Out Group:", out_group) 


# Helper function find a station's minimum distance from the edge of the caldera
# generate a plot if requested
def station_caldera_dist(station_l, caldera_file, plot=False):

    # Approximate lat/long to km conversions in this region calculated in distance_to_stations()
    long_to_km = 66.63235041229169
    lat_to_km = 97.15186510942182

    # get station
    station = station_l[0]

    # read the caldera file
    caldera_df = pd.read_csv(caldera_file, delimiter=' ', header=None)
    
    caldera_long = caldera_df[0]-360
    caldera_lat = caldera_df[1]

    # iterate through the caldera coordinates to get the minimum distance between the caldera and station point
    min_dist = np.inf
    for i in range(len(caldera_lat)):

        diff_long = caldera_long[i] - station.long
        diff_lat = caldera_lat[i] - station.lat

        diff_x = diff_long*long_to_km
        diff_y = diff_lat*lat_to_km

        dist = np.sqrt(diff_x**2 + diff_y**2)

        if dist <  min_dist:
            min_dist = dist
            min_long = caldera_long[i]
            min_lat = caldera_lat[i]
    
    # make a plot to debug in needed
    if plot:
        topo_data = '@earth_relief_15s' #15 arc second global relief (SRTM15+V2.1)
        pygmt.makecpt(cmap='topo',series='-8000/8000/1000',continuous=True)
        
        region = [-111.8, -109.25, 43.92, 45.5]

        fig = pygmt.Figure()
        proj = 'M4i'
        with pygmt.config(FONT="20p"): #set fontsize for the longitude and latitude markers
            fig.grdimage(grid=topo_data, shading=True, region=region,projection=proj, frame=True)
            fig.coast(region=region,projection=proj, frame=True, water="lightblue", rivers="lightblue")

        # plot caldera and station
        fig.plot(data=caldera_file, pen="2p,black")
        fig.plot(x=station.long, y=station.lat,color="darkslategray2", style="s0.35c", pen='black')

        # plot connecting line
        fig.plot(x=[station.long, min_long], y=[station.lat, min_lat], pen="1p,black")

        fig.show()
    
    return min_dist


# Helper function to write a time, velocity csv file for a station or station group.
def write_data_file(x, y, name):
    f = open(name, "w")
    f.write("time[yrs],velocity[m/yr]\n")
    for i in range(len(x)):
        f.write(str(round(x[i], 3)) + "," + str(round(y[i], 5)) + "\n")
    f.close()


############################################################################################################################
# Plotting function calls

# <V>^inner, <V>^outer, and <V>^regional
in_group = ['LKWY', 'P801', 'WLWY', 'P709', 'HVWY', 'OFW2']
out_group = ['P711', 'MAWY', 'P686', 'P714', 'P360']
regional_group = ['P361','P676', 'P456', 'P712', 'P460', 'P457', 'P461', 'P721', 'P720', 'P718', 'P710']

long_to_km = 66.63235041229169
lat_to_km = 97.15186510942182

# function call to make figure 1a
# aspect_ratio_scaling = 0.1
# make_location_map(get_data_list(), source_dim=(13*0.5*aspect_ratio_scaling, 27.5*aspect_ratio_scaling), source_center=(-110.63, 44.54), source_angle=360-58, 
#                   CR_dim=(32*aspect_ratio_scaling, 55*aspect_ratio_scaling), CR_center=(-110.63-5/long_to_km, 44.54-0.75/lat_to_km), CR_angle=360-58, 
#                   inner_CR_dim=(22*aspect_ratio_scaling, 40*aspect_ratio_scaling), inner_CR_center=(-110.63, 44.54), inner_CR_angle=360-58, 
#                   caldera_file="GMT_data/YScalderaL.txt", add_caldera_file="GMT_data/YScalderas_hrmf.txt", fault_file=None, eq_file="GMT_data/ys_eq.csv", 
#                   bounds=[-112.4, -112.4+3.074377, 43.4, 43.4+2.2585], plot_stations=True, in_stations=in_group, 
#                   out_stations=out_group)

# function call to make the inset for Figure 1 a
# make_location_map(get_data_list(), caldera_file="GMT_data/YScalderaL.txt", add_caldera_file="GMT_data/YScalderas2L.txt", bounds=[-117, -109, 41, 45.5], statelines=True)

# function call to make figure 1b
# aspect_ratio_scaling = 0.132
# make_vs_map("GMT_data/ys_vs3d_m10.dat", depth=5, source_dim=(13*0.5*aspect_ratio_scaling, 27.5*aspect_ratio_scaling), source_center=(-110.63, 44.54), 
#             source_angle=360-58, CR_dim=(32*aspect_ratio_scaling, 55*aspect_ratio_scaling), CR_center=(-110.63-5/long_to_km, 44.54-0.75/lat_to_km), 
#             CR_angle=360-58, inner_CR_dim=(22*aspect_ratio_scaling, 40*aspect_ratio_scaling), inner_CR_center=(-110.63, 44.54), inner_CR_angle=360-58,
#             caldera_file="GMT_data/YScalderaL.txt", add_caldera_file="GMT_data/YScalderas_hrmf.txt", 
#             # bounds=[-112.4, -112.4+3.074377, 43.4, 43.4+2.2585], ref=-1, plot_stations=True, station_list=get_data_list(), 
#             bounds=[-111.8, -109.5, 43.8, 45.3], ref=-1, plot_stations=True, station_list=get_data_list(), 
#             in_stations=in_group, out_stations=out_group,
#             other_stations=['NRWY'],
#             plot_profiles=True)

# function call to make figure 2 a-c
# plot_GPS_vert(get_data_list(), [2014, 2016], in_stations=in_group, out_stations=out_group, fault_file=None, bounds=[-111.8, -109.25, 43.92, 45.5], 
#               CR_dim=None, CR_center=None, CR_angle=None, caldera_file="GMT_data/YScalderaL.txt", plot_names=False) 

# function call to make figure A1
# plot_GPS_horiz(get_data_list(), [2014, 2016], in_stations=in_group, out_stations=out_group, fault_file=None, bounds=[-111.8, -109.25, 43.92, 45.5], 
#                 CR_dim=None, CR_center=None, CR_angle=None, caldera_file="GMT_data/YScalderaL.txt", reference_station=-1, plot_names=True)

# Function calls to make figure A2
# plot_all_stations_vel(get_data_list(station_subset=in_group), (3, 2)) # in stations
# plot_all_stations_vel(get_data_list(station_subset=['P711', 'NRWY', 'MAWY', 'P686', 'P714', 'P360']), (3, 2), disp_bounds=[-0.06, 0.06]) # out stations
# plot_all_stations_vel(get_data_list(station_subset=regional_group), (5, 2), disp_bounds=[-0.06, 0.06]) # RG stations

# function call to make pressure function
# make_center_mean_timedb(get_data_list(station_subset=in_group), 23, "demo_run", dt=0.001)

# function call to make Fig. 3c
# plot_compare_timedbs(['run_demo_23kPa_unsmoothed.timedb', 'run_demo_13kPa_unsmoothed.timedb'], [23, 13], get_data_list(station_subset=in_group))

# function call to get the distance of stations to the source center (used to calculated model inner and outer groups)
# print(distance_to_stations(get_data_list(['LKWY']), (-110.63, 44.54)))

# Function call to determine the inner and outer groups and make part of table B1
calc_cross_correlations(get_data_list(['LKWY']), get_data_list(), time_window=[2004, 2016])
# Function call to calculate the regional group (CC > 0.2 with P361)
# calc_cross_correlations(get_data_list(['P361']), get_data_list(), time_window=[2004, 2016])

