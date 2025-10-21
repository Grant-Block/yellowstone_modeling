import ModelStorage_final
import numpy as np
import h5py
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd


# Class to read in and analyze hdf5 output from pylith
# it will find the derived parameters then write them to the appropriate place
# in the master csv
class Analysis:
    def __init__(self, model_name, r_x=25, r_y=None, x_off=0.0, y_off=0.0, mesh_width=150e3):
        self.model_name = model_name
        self.model = ModelStorage_final.Model(model_name)
        self.ms_to_mmyr = 3.154e+10 #m/s to mm/yr velocity conversion
        self.r_x = r_x #km. Radius of source in x direction. Set to 25 as a default for most models. Needed to calculate sombreros. 
        self.mesh_width = mesh_width
        if r_y == None: #km. Radius of source in y direction. Equal to r_x in case of cylindrically symmetric sources.
            self.r_y = r_x
        else:
            self.r_y = r_y 
        
        self.x_off = x_off*1e3 #x and y offset to interpolate with. Should be passed in in km but needed in m for read_hdf5().
        self.y_off = y_off*1e3

        #default plot formatting parameters
        self.lw = 4
        self.ms = 75
        self.label_fontsize=15
        self.axes_fontsize=15
        self.legend_fontsize=15
        self.title_fontsize=20
        self.tick_fontsize=15

        self.model_data = {}    #initiate as empty dict which will will store
                                #model data at a time step so the hdf5 will only 
                                #need to be read once per time step for a given
                                #model object.
        
        self.source_velocity = {} #a dictionary to store the averaged velocity over the full source at a given time
        
        # get timestep list
        if "Yellowstone" in self.model.model_name:
            path = "../../../../Yellowstone/"+str(self.model.path)+"/"+self.model.path[4:]+"-groundsurf.h5" # TO RUN, CHANGE THIS TO THE DIRECTORY WHERE YOUR
                                                                                                            # MODEL .h5 FILES ARE
       
        with h5py.File(path, "r") as f:
            self.times = f['time'][:,0,0]



    #read hdf5 and store model data
    def read_hdf5(self, time_step, theta=None, x_off=0.0, y_off=0.0):

        #check to see if time step is already in the dictionary
        if (time_step, theta, x_off, y_off) in self.model_data:
            print("Time step has already been read from hdf5.")
            return
    
        #get the file path
        path = "../../../../Yellowstone/"+str(self.model.path)+"/"+self.model.path[4:]+"-groundsurf.h5" # TO RUN, CHANGE THIS TO THE DIRECTORY WHERE YOUR
                                                                                                        # MODEL .h5 FILES ARE
        
        
        #prepare lists for getting data from hdf5 files
        with h5py.File(path, "r") as f:

            group_geometry = f['geometry']
            group_vert_fields = f['vertex_fields']
            
            points = group_geometry['vertices'] #shape: point_num, xyz
            displacements = group_vert_fields['displacement'] #shape: timestep, point_num, xyz
            velocities = group_vert_fields['velocity'] #shape: timestep, point_num, xyz
        
            x = points[:][:,0]
            y = points[:][:,1]
            
            disp_x = displacements[time_step][:][:,0]
            disp_y = displacements[time_step][:][:,1]
            disp_z = displacements[time_step][:][:,2]
            
            vel_x = velocities[time_step][:][:,0]
            vel_y = velocities[time_step][:][:,1]
            vel_z = velocities[time_step][:][:,2]

            #get radial components
            x[x==0] = 0.0001
            angle = np.arctan(y/x)

            disp_r = disp_x*np.cos(angle)+disp_y*np.sin(angle)
            disp_theta = -disp_x*np.sin(angle)+disp_y*np.cos(angle)
        
            vel_r = vel_x*np.cos(angle)+vel_y*np.sin(angle)
            vel_theta = -vel_x*np.sin(angle)+vel_y*np.cos(angle)
            
            
            
            #interpolate data 
            #get points to interpolate
            x_i = np.linspace(-self.mesh_width, self.mesh_width, 11615)

            if theta == None:
                y_i = np.zeros(len(x_i))
            else:
                x_i_temp = x_i
                x_i = np.cos(theta)*x_i_temp 
                y_i = np.sin(theta)*x_i_temp 
                
            x_i += x_off
            y_i += y_off
            
            #interpolate
            disp_x_interp = interpolate.griddata(np.asarray([x, y]).T, np.asarray(disp_x), np.asarray([x_i, y_i]).T, method='cubic')
            disp_y_interp = interpolate.griddata(np.asarray([x, y]).T, np.asarray(disp_y), np.asarray([x_i, y_i]).T, method='cubic')
            disp_z_interp = interpolate.griddata(np.asarray([x, y]).T, np.asarray(disp_z), np.asarray([x_i, y_i]).T, method='cubic')
            
            vel_x_interp = interpolate.griddata(np.asarray([x, y]).T, np.asarray(vel_x), np.asarray([x_i, y_i]).T, method='cubic')
            vel_y_interp = interpolate.griddata(np.asarray([x, y]).T, np.asarray(vel_y), np.asarray([x_i, y_i]).T, method='cubic')
            vel_z_interp = interpolate.griddata(np.asarray([x, y]).T, np.asarray(vel_z), np.asarray([x_i, y_i]).T, method='cubic')

            disp_r_interp = interpolate.griddata(np.asarray([x, y]).T, np.asarray(disp_r), np.asarray([x_i, y_i]).T, method='cubic')
            disp_theta_interp = interpolate.griddata(np.asarray([x, y]).T, np.asarray(disp_theta), np.asarray([x_i, y_i]).T, method='cubic')
        
            vel_r_interp = interpolate.griddata(np.asarray([x, y]).T, np.asarray(vel_r), np.asarray([x_i, y_i]).T, method='cubic')
            vel_theta_interp = interpolate.griddata(np.asarray([x, y]).T, np.asarray(vel_theta), np.asarray([x_i, y_i]).T, method='cubic')

            # add all values to dictionary at the time step
            self.model_data[(time_step, theta, x_off, y_off)] = (x_i, disp_x_interp, disp_y_interp, disp_z_interp, disp_r_interp, disp_theta_interp, 
                                          vel_x_interp, vel_y_interp, vel_z_interp, vel_r_interp, vel_theta_interp)
            
    
    # a helper function to get displacements and velocities either from
    # the dictionary or read the hdf5 file if the data hasn't been read yet
    def get_data(self, model_time, theta=None):
        if (model_time, theta, self.x_off, self.y_off) not in self.model_data:
            self.read_hdf5(model_time, theta=theta, x_off=self.x_off, y_off=self.y_off)
        return self.model_data[(model_time, theta, self.x_off, self.y_off)]

    def get_data_no_shift(self, model_time, theta=None):
        if (model_time, theta, 0, 0) not in self.model_data:
            self.read_hdf5(model_time, theta=theta, x_off=0, y_off=0)
        return self.model_data[(model_time, theta, 0,0)]

    
    # a function to convert times in years to time steps for the h5 file. 
    def get_timesteps(self, time):
        time_idx = (np.abs(time*3.154e+7 - self.times)).argmin() #get time index
        return time_idx
    


    # A function to plot model and GPS <V>^inner and <V>^outer. In addition, it can calculate model CC and R_v metrics as well as 
    # write the calculated model <V>^inner and <V>^outer to files. 
    def plot_point_station_avg(self, times, points_inner, points_outer, GPS_in_mean_file, 
                                GPS_out_mean_file, RG_file=None, shift_time=1986, mult_factor=1, plot_unscaled=False,
                                pressure_func_file=None, calc_metrics=False, write_files=False):
        

         # get base x profile
        (x_prof0, disp_x_prof, disp_y_prof, disp_z_prof, disp_r_prof, disp_theta_prof,
                    vel_x_prof, vel_y_prof, vel_z_prof, 
                    vel_r_prof, vel_theta_prof) = self.get_data_no_shift(0, theta=0)
        
        # set up plot
        fig, ax = plt.subplots(2, 1, figsize=(17, 6),sharex=True)
        fig.text(0.04, 0.5, "Vertical Velocity (mm/yr)", fontsize=30, va='center', rotation='vertical')
        ax[0].grid()
        ax[0].tick_params(axis='y', labelsize=30)
        ax[1].grid()
        ax[1].set_xlabel("Time (yrs)", fontsize=30)
        ax[1].tick_params(axis='y', labelsize=30)
        ax[1].tick_params(axis='x', labelsize=30)


        # loop through list of inner and outer points and collect the model velocity time series at each
        points_inner_list = []
        for point in points_inner:
            # get model point
            x = point[0]*1e3
            y = point[1]*1e3
            r = np.sqrt(x**2 + y**2)

            if x < 0:
                r *= -1

            if x == 0 and y == 0:
                theta = 0
            elif x == 0 and y > 0:
                theta = np.pi/2
            elif x == 0 and y < 0:
                theta = -np.pi/2
            else:
                theta = np.arctan(y/x)
                
            vel_array = []

            # read model data to get velocity at each time step
            for t in times:
                model_time = self.get_timesteps(t)

                (x_prof, disp_x_prof, disp_y_prof, disp_z_prof, disp_r_prof, disp_theta_prof,
                    vel_x_prof, vel_y_prof, vel_z_prof, 
                    vel_r_prof, vel_theta_prof) = self.get_data(model_time, theta=theta)
                    
                point_idx = (np.abs(x_prof0 - r)).argmin() #get point index
                vel_array.append(vel_z_prof[point_idx])
            
            if type(mult_factor) != tuple and type(mult_factor) != list:
                points_inner_list.append(np.asarray(vel_array)*self.ms_to_mmyr*mult_factor)
            else:
                points_inner_list.append(np.asarray(vel_array)*self.ms_to_mmyr*mult_factor[0])

        points_outer_list = []   
        for point in points_outer:
            # get model point velocities
            x = point[0]*1e3
            y = point[1]*1e3
            r = np.sqrt(x**2 + y**2)

            if x < 0:
                r *= -1
            if x == 0 and y == 0:
                theta = 0
            elif x == 0 and y > 0:
                theta = np.pi/2
            elif x == 0 and y < 0:
                theta = -np.pi/2
            else:
                theta = np.arctan(y/x)
            
                
            vel_array = []

            for t in times:
                model_time = self.get_timesteps(t)

                (x_prof, disp_x_prof, disp_y_prof, disp_z_prof, disp_r_prof, disp_theta_prof,
                    vel_x_prof, vel_y_prof, vel_z_prof, 
                    vel_r_prof, vel_theta_prof) = self.get_data(model_time, theta=theta)
                    
                point_idx = (np.abs(x_prof0 - r)).argmin() #get point index
                vel_array.append(vel_z_prof[point_idx])
            if type(mult_factor) != tuple and type(mult_factor) != list:
                points_outer_list.append(np.asarray(vel_array)*self.ms_to_mmyr*mult_factor)
            else:
                points_outer_list.append(np.asarray(vel_array)*self.ms_to_mmyr*mult_factor[1])


        # read inner, outer, and regional GPS mean files
        df_in = pd.read_csv(GPS_in_mean_file)
        GPS_in_times = df_in['time[yrs]'].values
        GPS_in_vel = df_in['velocity[m/yr]'].values

        df_out = pd.read_csv(GPS_out_mean_file)
        GPS_out_times = df_out['time[yrs]'].values
        GPS_out_vel = df_out['velocity[m/yr]'].values

        if RG_file != None:
            df_RG = pd.read_csv(RG_file)
            GPS_RG_times = (df_RG['time[yrs]'].values)[1:]
            GPS_RG_vel = (df_RG['velocity[m/yr]'].values)[1:]


            GPS_out_RG_vel = GPS_out_vel[GPS_out_times >= min(GPS_RG_times)] - GPS_RG_vel

        # get means of model point velocities
        points_inner_mean = np.mean(np.asarray(points_inner_list), axis=0)
        points_outer_mean = np.mean(np.asarray(points_outer_list), axis=0)

        # plot
        inner_color = 'indigo'
        outer_color = 'teal'

        # different plot options depending on what multiplication factor we want to use on the model inner or outer
        # When plotting, and generally working with both model and GPS, it's important to note that the model time starts at 0,
        # and undergoes 500 years of spin-up before being subject to P(t). Therefore, to make the model time and GPS time align, a given model time should 
        # correspond to model_time-500+shift_time, where shift_time is the year that the start of P(t) corresponds to
        if type(mult_factor) != tuple and type(mult_factor) != list:
            if mult_factor == 1: 
                l1, = ax[0].plot(times-500+shift_time, points_inner_mean, color=inner_color, lw=10, alpha=0.5, label=r"Model $\langle V\rangle^{inner}$")
                ax[1].plot(times-500+shift_time, points_outer_mean, color=outer_color, lw=10, alpha=0.5, label=r"Model $\langle V\rangle^{outer}$")
            else:
                l1 = ax[0].plot(times-500+shift_time, points_inner_mean, color=inner_color, lw=10, alpha=0.5, label=r"Model $\langle V\rangle^{inner}$, scaled x"+str(mult_factor))
                ax[1].plot(times-500+shift_time, points_outer_mean, color=outer_color, lw=10, alpha=0.5, label=r"Model $\langle V\rangle^{outer}$, scaled x"+str(mult_factor))
        else:
            if mult_factor[0] == 1:
                l1, = ax[0].plot(times-500+shift_time, points_inner_mean, color=inner_color, lw=10, alpha=0.5, label=r"Model $\langle V\rangle^{inner}$")
            else:
                l1, = ax[0].plot(times-500+shift_time, points_inner_mean, color=inner_color, lw=10, alpha=0.5, linestyle='dotted', label=r"Model $\langle V\rangle^{inner}$, scaled x"+str(mult_factor[0]))
                if plot_unscaled:
                   l1, =  ax[0].plot(times-500+shift_time, points_inner_mean/mult_factor[0], color=inner_color, lw=10, alpha=0.5, label=r"Model $\langle V\rangle^{inner}$, unscaled")


            if mult_factor[1] == 1:
                ax[1].plot(times-500+shift_time, points_outer_mean, color=outer_color, lw=10, alpha=0.5, label=r"Model $\langle V\rangle^{outer}$")
            else:
                ax[1].plot(times-500+shift_time, points_outer_mean, color=outer_color, lw=10, alpha=0.5, linestyle='dotted', label=r"Model $\langle V\rangle^{outer}$, scaled x"+str(mult_factor[1]))
                if plot_unscaled:
                    ax[1].plot(times-500+shift_time, points_outer_mean/mult_factor[1], color=outer_color, lw=10, alpha=0.5, label=r"Model $\langle V\rangle^{outer}$, unscaled")



        # plot the GPS
        l2, = ax[0].plot(GPS_in_times, GPS_in_vel*1e3, color=inner_color, lw=6, linestyle="--", label=r"Smoothed GPS $\langle V\rangle^{inner}$")
        
        ax[1].plot(GPS_out_times, GPS_out_vel*1e3, color=outer_color, lw=6, linestyle="--", label=r"GPS $\langle V\rangle^{outer}$")

        if RG_file != None:
            ax[1].plot(GPS_RG_times, GPS_out_RG_vel*1e3, color='gray', lw=6, linestyle="--", label=r"GPS $\langle V\rangle^{outer} - \langle V\rangle^{RG}$")
        
        # plot pressure function
        if pressure_func_file != None:
            df = pd.read_csv(pressure_func_file, sep=' ', names=['times', 'pressures'], skiprows=[0,1,2,3,4])
            pressure_times = df['times']
            pressure_values = df['pressures']

            ax3 = ax[0].twinx()
            ax3.set_ylabel(r"$P(t)-P_0$ (kPa)", fontsize=25)
            ax3.set_xlim([min(np.asarray(times-500+shift_time)), max(np.asarray(times-500+shift_time))])

            l3, = ax3.plot(pressure_times-500+shift_time-1, (np.asarray(pressure_values)-1)*1e2, linewidth=self.lw, label="Applied Pressure Function", color='black', linestyle='dashed')
            ax3.tick_params(axis='y', labelsize=30)
        
            ax[0].set_zorder(ax3.get_zorder()+1) # put ax in front of ax2
            ax[0].patch.set_visible(False) # hide the 'canvas'
            ax[0].legend((l1, l2, l3), (l1.get_label(), l2.get_label(), l3.get_label()), handlelength=4,handleheight=1.5,fontsize=12, loc='upper left', bbox_to_anchor=(1.3, 1.1))
        else:
            ax[0].legend((l1, l2), (l1.get_label(), l2.get_label()), handlelength=4,handleheight=1.5,fontsize=12, loc='upper left', bbox_to_anchor=(1.15, 1.1))
        
        ax[1].legend(handlelength=4,handleheight=1.5,fontsize=12, loc='upper left', bbox_to_anchor=(1.15, 1.1))

        # calculate CC and R_v for the model and GPS
        if calc_metrics:
            from scipy.stats import pearsonr
            from scipy.interpolate import interp1d

            unscaled_central = points_inner_mean/mult_factor[0]
            unscaled_outer = points_outer_mean/mult_factor[1]
            # cross correlation
            time_window = (times > 2004-shift_time+500) & (times <= 2016-shift_time+500)
            result_model = pearsonr(unscaled_central[time_window], unscaled_outer[time_window])

            # calculate ratio
            time_window = (times > 2005-shift_time+500) & (times <= 2007-shift_time+500)
            max_center_idx = np.argmax(unscaled_central[time_window])
            max_center_uplift = unscaled_central[time_window][max_center_idx]
            point_vel = unscaled_outer[time_window][max_center_idx]
            ratio = np.abs(max_center_uplift/point_vel)
            if point_vel > 0:
                print("Warning: outer velocity > 0, ratio invalid")

            # calculate data metrics
            time_window_GPS_in = (GPS_in_times > 2004) & (GPS_in_times <= 2016)
            time_window_GPS_out = (GPS_out_times >= min(GPS_in_times[time_window_GPS_in])) & (GPS_out_times <= max(GPS_in_times[time_window_GPS_in]))
            
            # interpolate the GPS_in time series
            interp_f = interp1d(GPS_in_times[time_window_GPS_in], GPS_in_vel[time_window_GPS_in])
            GPS_in_vel_interp = interp_f(GPS_out_times[time_window_GPS_out])
            result_data = pearsonr(GPS_in_vel_interp, GPS_out_vel[time_window_GPS_out])

            time_window_GPS_in = (GPS_in_times > 2005) & (GPS_in_times <= 2007)
            time_window_GPS_out = (GPS_out_times > 2005) & (GPS_out_times <= 2007)
            max_center_idx = np.argmax(GPS_in_vel[time_window_GPS_in])
            # print(GPS_in_times[time_window_GPS_in][max_center_idx], GPS_out_vel[np.argmin(np.abs(GPS_out_times -GPS_in_times[time_window_GPS_in][max_center_idx]))])
            max_center_uplift = GPS_in_vel[time_window_GPS_in][max_center_idx]
            point_vel =  GPS_out_vel[np.argmin(np.abs(GPS_out_times -GPS_in_times[time_window_GPS_in][max_center_idx]))]
            ratio_GPS = np.abs(max_center_uplift/point_vel)

            # Calculate data metrics with outer-RG
            time_window_GPS_in = (GPS_in_times > 2004) & (GPS_in_times <= 2016)
            time_window_GPS_RG = (GPS_RG_times >= min(GPS_in_times[time_window_GPS_in])) & (GPS_RG_times <= max(GPS_in_times[time_window_GPS_in]))
            interp_f = interp1d(GPS_in_times[time_window_GPS_in], GPS_in_vel[time_window_GPS_in])
            # print(GPS_in_times[time_window_GPS_in])
            # print(GPS_RG_times[time_window_GPS_RG])
            GPS_in_vel_interp = interp_f(GPS_RG_times[time_window_GPS_RG])
            
            # calculate misfits
            CC_misfit = (result_data[0] - result_model[0])/result_data[0]
            ratio_misfit = (ratio_GPS - ratio)/ratio_GPS
            total_misfit = 0.5*np.sqrt(CC_misfit**2 + ratio_misfit**2)

        
            print("Cross Correlation:", str(result_model[0]), "P-Value", str(result_model[1]), "Ratio:", str(ratio))
            print("GPS Cross Correlation:", str(result_data[0]), "Ratio:", str(ratio_GPS))
            print("CC misfit:", str(CC_misfit), "Ratio misfit:", str(ratio_misfit), "Total misfit:", str(total_misfit))


        # write model output to csv files
        if write_files:
            unscaled_inner = points_inner_mean/mult_factor[0]
            unscaled_outer = points_outer_mean/mult_factor[1]

            file_name_inner = 'model_inner.csv'
            file_name_outer = 'model_outer.csv'

            shifted_time = times-500+shift_time

            # write inner
            f = open(file_name_inner, "w")
            f.write("time[yrs],velocity[mm/yr]\n")
            for i in range(len(shifted_time)):
                f.write(str(round(shifted_time[i], 3)) + "," + str(round(unscaled_inner[i], 5)) + "\n")
            f.close()

            #write outer
            f = open(file_name_outer, "w")
            f.write("time[yrs],velocity[mm/yr]\n")
            for i in range(len(shifted_time)):
                f.write(str(round(shifted_time[i], 3)) + "," + str(round(unscaled_outer[i], 5)) + "\n")
            f.close()

        
        plt.show()
        fig.savefig("../../../Figures/Model_Obs_mean.png", dpi=200, bbox_inches='tight')
        plt.close()

    
    # Function to plot surface velocities at a given time or mean time. Has options to plot velocity profiles across the surface, points signifying the location
    # of GPS stations, and the surface projection of the source and CR.
    def read_plot_groundsurf(self, time_step, mean_time_steps=None, parameter='vz', som_width=False, CR=False, CR_x=100, CR_y=None, CR_inner=None, envelope=None, source=False, 
                             second_source=None, source_disp=None, second_source_disp=None, CR_disp=None, inner_CR_disp=None, profile=False, x_lim=None, 
                             y_lim=None, theta=None, save=False, in_points_list=None, out_points_list=None, plot_caldera=False, log_scale=True):

        #first need to get model time (or times if we're averaging)
        if mean_time_steps == None:
            model_time = self.get_timesteps(time_step)
        else:
            model_time_list = []
            for t in mean_time_steps:
                model_time_list.append(self.get_timesteps(t))
            model_time = np.asarray(model_time_list)

        #get the file path
        path = "../../../../Yellowstone/"+str(self.model.path)+"/"+self.model.path[4:]+"-groundsurf.h5"
        
        #prepare lists for getting data from hdf5 files
        with h5py.File(path, "r") as f:
            group_geometry = f['geometry']
            group_vert_fields = f['vertex_fields']
            
            points = group_geometry['vertices'] #shape: point_num, xyz
            displacements = group_vert_fields['displacement'] #shape: timestep, point_num, xyz
            velocities = group_vert_fields['velocity'] #shape: timestep, point_num, xyz
        
            x = points[:][:,0]
            y = points[:][:,1]
            
            # check if we're averaging multiple time steps
            if mean_time_steps == None:
                disp_x = displacements[model_time][:][:,0]
                disp_y = displacements[model_time][:][:,1]
                disp_z = displacements[model_time][:][:,2]
                
                vel_x = velocities[model_time][:][:,0]
                vel_y = velocities[model_time][:][:,1]
                vel_z = velocities[model_time][:][:,2]
            else:
                # get arrays of 2D data at each time step
                disp_x_list = []
                disp_y_list = []
                disp_z_list = []
                vel_x_list = []
                vel_y_list = []
                vel_z_list = []

                for mt in model_time:
                    disp_x_list.append(displacements[mt][:][:,0])
                    disp_y_list.append(displacements[mt][:][:,1])
                    disp_z_list.append(displacements[mt][:][:,2])
                    
                    vel_x_list.append(velocities[mt][:][:,0])
                    vel_y_list.append(velocities[mt][:][:,1])
                    vel_z_list.append(velocities[mt][:][:,2])


           
            #get profile if needed
            if profile:
                (x_prof, disp_x_prof, disp_y_prof, disp_z_prof, disp_r_prof, disp_theta_prof,
                  vel_x_prof, vel_y_prof, vel_z_prof, 
                  vel_r_prof, vel_theta_prof) = self.get_data(model_time, theta=theta)

            #Select data being plotted
            if parameter == 'dx':
                if mean_time_steps == None:
                    plot_data = disp_x
                else:
                    plot_data = np.asarray(disp_x_list)
                cbar_label = "Displacement X [mm]"
                conversion = 1e3
                if profile:
                    prof_data = disp_x_prof
            elif parameter == 'dy':
                if mean_time_steps == None:
                    plot_data = disp_y
                else: 
                    plot_data = np.asarray(disp_y_list)
                cbar_label = "Displacement Y [mm]"
                conversion = 1e3
                if profile:
                    prof_data = disp_y_prof
            elif parameter == 'dz':
                if mean_time_steps == None:
                    plot_data = disp_z
                else:
                    plot_data = np.asarray(disp_z_list)
                cbar_label = "Displacement Z [mm]"
                conversion = 1e3
                if profile:
                    prof_data = disp_z_prof
            elif parameter == 'vx':
                if mean_time_steps == None:
                    plot_data = vel_x
                else:
                    plot_data = np.asarray(vel_x_list)
                cbar_label = "Velocity X [mm/yr]"
                conversion = self.ms_to_mmyr
                if profile:
                    prof_data = vel_x_prof
            elif parameter == 'vy':
                if mean_time_steps == None:
                    plot_data = vel_y
                else:
                    plot_data = np.asarray(vel_y_list)
                cbar_label = "Velocity Y [mm/yr]"
                conversion = self.ms_to_mmyr
                if profile:
                    prof_data = vel_y_prof
            elif parameter == 'vz':
                if mean_time_steps == None:
                    plot_data = vel_z
                else:
                    plot_data = np.asarray(vel_z_list)
                cbar_label = "Velocity Z [mm/yr]"
                conversion = self.ms_to_mmyr
                if profile:
                    prof_data = vel_z_prof
            else:
                print("Parameter \""+parameter+"\" is not supported. Will not continue.")
                return
            
            if mean_time_steps != None:
                # take the mean of the plot data
                plot_data_mean = np.mean(plot_data, axis=0)

            #interpolate on 2D grid
            grid_x, grid_y = np.mgrid[-self.mesh_width:self.mesh_width:1000j, -self.mesh_width:self.mesh_width:1000j]
            
            if mean_time_steps == None:
                z = interpolate.griddata((x/1e3,y/1e3), plot_data, (grid_x/1e3, grid_y/1e3), method='cubic')
            else:
                z = interpolate.griddata((x/1e3,y/1e3), plot_data_mean, (grid_x/1e3, grid_y/1e3), method='cubic')



            #plot
            plt.figure()
            color_map='seismic'

            # check if we're plotting log scale or not
            if log_scale:
                plt.imshow(z.T*conversion, extent=(-self.mesh_width/1e3,self.mesh_width/1e3,-self.mesh_width/1e3,self.mesh_width/1e3),
                            cmap=color_map, norm=matplotlib.colors.SymLogNorm(linthresh=0.5, linscale=1.0, vmin=-80, vmax=80))
            else:
                from matplotlib import colors
                divnorm=colors.TwoSlopeNorm(vmin=-10., vcenter=0., vmax=30)
                im = plt.imshow(z.T*conversion, extent=(-self.mesh_width/1e3,self.mesh_width/1e3,-self.mesh_width/1e3,self.mesh_width/1e3),
                            cmap=color_map, norm=divnorm)

                ctr = plt.contour(z.T*conversion, [-5, 0, 5, 10, 20], origin='upper', colors=['black', 'black', 'black', 'black', 'black'],
                                extent=(-self.mesh_width/1e3,self.mesh_width/1e3,-self.mesh_width/1e3,self.mesh_width/1e3))

            plt.xlabel("X [km]", fontsize=40)
            plt.ylabel("Y [km]", fontsize=40)
            plt.xticks(fontsize=25)
            plt.yticks(fontsize=40)
            if log_scale:
                cbar = plt.colorbar()
            else:
                cbar = plt.colorbar(im)
                cbar.add_lines(ctr)
            cbar.set_label(label="Velocity (mm/yr)", size=40)
            cbar.ax.tick_params(labelsize=30)
            
            # Plot points showing GPS stations if requested
            inner_color = 'indigo'
            outer_color = 'teal'

            if in_points_list != None:
                for point in in_points_list:
                    plt.scatter(point[0]+self.x_off/1e3, point[1]+self.y_off/1e3, c=inner_color, marker='s', s=200, edgecolor='black', linewidths=2, zorder=2)
            if out_points_list != None:
                for point in out_points_list:
                    plt.scatter(point[0]+self.x_off/1e3, point[1]+self.y_off/1e3, c=outer_color, marker='s', s=200, edgecolor='black', linewidths=2)
            
            # plot caldera outline if requested
            if plot_caldera:
                df_caldera = pd.read_csv("shifted_caldera.csv", sep=',', header=None)
                plt.plot(df_caldera[0]+self.x_off/1e3, df_caldera[1]+self.y_off/1e3, linewidth=3)

            #get points of contour enclosing the sombrero width
            if som_width:
                if parameter=='vz':
                    x_min_list, y_min_list = self.get_som_width_contour(time_step)
                    plt.scatter(np.asarray(x_min_list)/1e3, np.asarray(y_min_list)/1e3, c='black', s=10)
                else:
                    print("Sombrero Width profiles are currently only implemented for vz.")
            
            if CR_disp == None:
                CR_disp = (0, 0)

            # plot CR
            if CR:
                CR_x_points, CR_y_points = get_CR_points(CR_x, CR_y)
                plt.plot(CR_x_points+CR_disp[0], CR_y_points+CR_disp[1], c='black', linestyle='dashed', linewidth=5)
            
            # plot inner CR
            if CR_inner != None:
                CR_x_inner, CR_y_inner = get_CR_points(CR_inner[0], CR_inner[1])
                plt.plot(CR_x_inner+inner_CR_disp[0], CR_y_inner+inner_CR_disp[1], c='black', linestyle='dashdot', linewidth=3)

            # plot the envelope
            if envelope != None:
                envelope_x, envelope_y = get_CR_points(envelope[0], envelope[1])
                plt.plot(envelope_x, envelope_y, linestyle='dashdot', c='black')

            # plot source
            if source:
                source_x_points, source_y_points = get_CR_points(self.r_x, self.r_y)
                if source_disp == None:
                    plt.plot(source_x_points+self.x_off/1e3, source_y_points+self.y_off/1e3, c='black', linestyle='dashdot', linewidth=4.5, zorder=3)
                else:
                    plt.plot(source_x_points+source_disp[0], source_y_points+source_disp[1], c='black', linestyle='dashdot', linewidth=4.5, zorder=2)
                if second_source != None:
                    second_source_x_points, second_source_y_points = get_CR_points(second_source[0], second_source[1])
                    if second_source_disp == None:
                        plt.plot(second_source_x_points, second_source_y_points, c='black', linestyle='dashdot')
                    else:
                        plt.plot(second_source_x_points+second_source_disp[0], second_source_y_points+second_source_disp[1], c='black', linestyle='dashdot')
            
            
            # set plot limits
            if x_lim != None:
                plt.xlim(x_lim)
            if y_lim != None:
                plt.ylim(y_lim)

            #plot profile line if needed
            if profile:
                x_i = np.linspace(-self.mesh_width, self.mesh_width, len(x_prof))
                if theta == None:
                    y_i = np.zeros(len(x_i))
                else:
                    x_i_temp = x_i
                    x_i = np.cos(theta)*x_i_temp 
                    y_i = np.sin(theta)*x_i_temp 
                    
                x_i += self.x_off
                y_i += self.y_off
                plt.plot(x_i/1e3, y_i/1e3, c="black")

            if save:
                file_name = "/home/grantblock/Research/SMBPylith/Figures/plot_groundsurface_gs"
                file_name += self.model_name.replace("/", "_") + parameter + "_"+str(time_step)+".png"
                plt.savefig(file_name, bbox_inches="tight")

            #plot profile
            if profile:
                plt.figure()
                plt.plot(x_i_temp/1e3, prof_data*conversion, linewidth=self.lw)
                plt.xlabel("Distance from center of profile [km]", fontsize=self.axes_fontsize)
                plt.ylabel(cbar_label, fontsize=self.axes_fontsize)
                plt.xticks(fontsize=self.tick_fontsize)
                plt.yticks(fontsize=self.tick_fontsize)
                plt.grid()

                if save:
                    file_name = "/home/grantblock/Research/SMBPylith/Figures/plot_groundsurface_prof"
                    file_name += self.model_name.replace("/", "_") + parameter + ".png"
                    plt.savefig(file_name, bbox_inches="tight")

            plt.show()

    # Function to plot mean velocity profiles over a specified set of times along the model surface, as well as the velocities of selected GPS stations
    def plot_profiles_mean_time(self, mean_times, theta=None, lim=None, shift_time=None, plot_points=None, station_locs=None):

        # get x_0 for plotting
        (x_0, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = self.get_data(0, theta=None)

        # set up plot
        fig, ax = plt.subplots(1, 1, figsize=(17, 6))
        ax.set_xlabel("Distance From Center of Source (km)", fontsize=40)
        ax.set_ylabel("Velocity (mm/yr)", fontsize=40)
        ax.tick_params(axis='x', labelsize=50)
        ax.tick_params(axis='y', labelsize=50)
        ax.grid()

        # get profiles at every specified time
        profile_list = []
        for t in mean_times:
            model_time = self.get_timesteps(t)
            (_, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = self.get_data(model_time, theta=theta)

            profile_list.append(vel_z*self.ms_to_mmyr)
        
        # get the mean profile
        mean_profile = np.mean(np.asarray(profile_list), axis=0)

        if theta == np.pi/2.:
            profile_legend = "YY' at <t>="+str(round(np.mean(np.asarray(mean_times))-500+shift_time, 2))
        else:
            profile_legend = "XX' at <t>="+str(round(np.mean(np.asarray(mean_times))-500+shift_time, 2))

        # plot

        ax.plot(x_0/1e3, mean_profile, linewidth=7, label=profile_legend, c='black')

        # if there are points to plot, iterate and plot them
        if plot_points != None:

            itr = 0
            for point in plot_points:
                # read GPS station
                df = pd.read_csv(point)
                GPS_times = df['time[yrs]'].values
                GPS_vel = df['velocity[m/yr]'].values

                # get data times corresponding to model times
                time_cut = (GPS_times >= mean_times[0]-500+shift_time) & (GPS_times <= mean_times[1]-500+shift_time)
                GPS_mean_vel = np.mean(GPS_vel[time_cut])*1e3

                if len(GPS_vel[time_cut]) > 0:
                
                    # plot point mean and range
                    if point in ['WLWY.csv', 'LKWY.csv', 'P709.csv', 'HVWY.csv', 'P801.csv', 'OFW2.csv']:
                        station_color = 'indigo'
                    else:
                        station_color = 'teal'
                    if theta == np.pi/2.:
                        ax.errorbar(station_locs[itr][1], GPS_mean_vel, yerr=np.asarray([[GPS_mean_vel-min(GPS_vel[time_cut]*1e3)], [max(GPS_vel[time_cut]*1e3)-GPS_mean_vel]]), 
                                    fmt='s', c=station_color,  markersize='20', markeredgecolor='black', elinewidth=5)
                    else:
                        ax.errorbar(station_locs[itr][0], GPS_mean_vel, yerr=np.asarray([[GPS_mean_vel-min(GPS_vel[time_cut]*1e3)], [max(GPS_vel[time_cut]*1e3)-GPS_mean_vel]]), 
                                    fmt='s', c=station_color,  markersize='20', markeredgecolor='black', elinewidth=5)
                else:
                    print("station: ", point, " not included")
                itr+=1  

        if lim != None:
            ax.set_xlim(lim)

        file_name = "/home/grantblock/Research/SMBPylith/Figures/profiles_mean_time_"
        file_name += self.model_name.replace("/", "_") + ".png"
        plt.savefig(file_name, bbox_inches="tight", dpi=200)
        
        plt.show()

    # Function that plots all CC's between a point in the model and the average velocity of the center of the model (points above the source)
    def plot_cc_surface_time(self, times, CR_bounds, CR_shift=[0,0], inner_CR=None, inner_CR_shift=None,
                              in_points_list=None, out_points_list=None, x_lim=None, y_lim=None, save=False):

        # Create points arrays
        dx = 5
        x = np.arange(-CR_bounds[0]+CR_shift[0]-5, CR_bounds[0]+CR_shift[0]+dx, dx)
        y = np.arange(-CR_bounds[1]+CR_shift[1]-5, CR_bounds[1]+CR_shift[1]+dx, dx)

        CC_list = np.empty((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                CC_list[i][j] = self.get_cc_center_point(times, (x[i], y[j]))

        z = CC_list.T


        #plot
        plt.figure()
        #color_map = plt.cm.get_cmap('RdBu').reversed()
        # color_map='seismic'
        color_map='Spectral_r'

        plt.imshow(z, extent=(-CR_bounds[0]+CR_shift[0]-5,CR_bounds[0]+CR_shift[1],-CR_bounds[1]+CR_shift[1]-5,CR_bounds[1]+CR_shift[1]),cmap=color_map, interpolation="nearest", origin="lower")
        # plt.imshow(z, extent=(-CR_bounds[0],CR_bounds[0],-CR_bounds[1],CR_bounds[1]),cmap=color_map, interpolation="nearest", origin="lower")
        plt.xlabel("X [km]", fontsize=self.label_fontsize)
        plt.ylabel("Y [km]", fontsize=self.label_fontsize)
        plt.xticks(fontsize=self.tick_fontsize)
        plt.yticks(fontsize=self.tick_fontsize)
        cbar = plt.colorbar()
        cbar.set_label(label="CC", size=self.label_fontsize)
        cbar.ax.tick_params(labelsize=self.tick_fontsize)
        
        # plot CR
        CR_x_points, CR_y_points = get_CR_points(CR_bounds[0], CR_bounds[1])
        plt.plot(CR_x_points+CR_shift[0], CR_y_points+CR_shift[1], c='black', linestyle='dashed')

        # plot inner CR if requested
        if inner_CR != None:
            inner_CR_x_points, inner_CR_y_points = get_CR_points(inner_CR[0], inner_CR[1])
            plt.plot(inner_CR_x_points+inner_CR_shift[0], inner_CR_y_points+inner_CR_shift[1], c='black', linestyle='dashdot')

        # plot source
        source_x_points, source_y_points = get_CR_points(self.r_x, self.r_y)
        plt.plot(source_x_points+self.x_off/1e3, source_y_points+self.y_off/1e3, c='black', linestyle='dashdot')

        if in_points_list != None:
            for point in in_points_list:
                plt.scatter(point[0]+self.x_off/1e3, point[1]+self.y_off/1e3, c='indigo', marker='s', s=200, edgecolor='black', linewidths=2, zorder=2)
        if out_points_list != None:
            for point in out_points_list:
                plt.scatter(point[0]+self.x_off/1e3, point[1]+self.y_off/1e3, c='teal', marker='s', s=200, edgecolor='black', linewidths=2)

        # set plot limits
        if x_lim != None:
            plt.xlim(x_lim)
        if y_lim != None:
            plt.ylim(y_lim)

        if save:
            file_name = "/home/grantblock/Research/SMBPylith/Figures/CC_surface"
            file_name += self.model_name.replace("/", "_") + "_" + str(times[0])+"-"+str(times[len(times)-1])+".png"
            plt.savefig(file_name, bbox_inches="tight")
        
        plt.show()

    # Function that takes in a point and time span and returns the cross correlation between the point velocity and the central area velocity
    def get_cc_center_point(self, times, point, debug=False):

         # get base x prof
        (x_prof0, disp_x_prof, disp_y_prof, disp_z_prof, disp_r_prof, disp_theta_prof,
                    vel_x_prof, vel_y_prof, vel_z_prof, 
                    vel_r_prof, vel_theta_prof) = self.get_data_no_shift(0, theta=0)

         # convert point to r and theta
        x = point[0]*1e3
        y = point[1]*1e3
        r = np.sqrt(x**2 + y**2)

        if x < 0:
            r *= -1
        if x == 0 and y == 0:
            theta = 0
        elif x == 0 and y > 0:
            theta = np.pi/2
        elif x == 0 and y < 0:
            theta = -np.pi/2
        else:
            theta = np.arctan(y/x)
        

        center_vel = []
        outer_vel = []
        for t in times:
            center_vel.append(self.get_average_vel_area(t, (self.r_x, self.r_y), offset=(self.x_off/1e3, self.y_off/1e3))) # get velocity averaged over the source area

            model_time = self.get_timesteps(t)

            # get point velocity at the time 
            (x_prof, disp_x_prof, disp_y_prof, disp_z_prof, disp_r_prof, disp_theta_prof,
                    vel_x_prof, vel_y_prof, vel_z_prof, 
                    vel_r_prof, vel_theta_prof) = self.get_data_no_shift(model_time, theta=theta)

                
            point_idx = (np.abs(x_prof0 - r)).argmin() #get point index

            outer_vel.append(vel_z_prof[point_idx])
            # outer_vel.append(vel_z)

        # shift the velocities so they're centered at 0 (this is not strictly necessary)
        center_vel_shift = np.asarray(center_vel)-np.mean(center_vel)
        outer_vel_shift = np.asarray(outer_vel)-np.mean(outer_vel)

        
        from scipy.stats import pearsonr
        result = pearsonr(outer_vel_shift, center_vel_shift)

        # debug plot
        if debug:
            fig=plt.figure()
            ax1 = plt.subplot(311)
            ax2 = plt.subplot(312, sharex = ax1)
            ax3 = plt.subplot(313)
            ax1.plot(times, center_vel_shift)
            ax1.set_ylabel("Centered Center Vel [mm/yr]")
                
            ax2.plot(times, outer_vel_shift*self.ms_to_mmyr)
            ax2.set_ylabel("Centered Point Vel [mm/yr]")
            ax2.set_xlabel("Time [yrs]")

            plt.show()
        
        return result[0]
    
    # get the average z velocity of a set of points within a specified elliptical or elliptical shell region. Time step and at least one
    # set of minor/major axis dimensions [km] must be passed in, along with an optional second major/minor axes to define an inner ellipse
    # to cut a shell.
    def get_average_vel_area(self, time_step, outer_axes, offset=[0,0], inner_axes=None):

        if time_step not in self.source_velocity:

            # start by reading h5 file

            #first need to get model time
            model_time = self.get_timesteps(time_step)

            #get the file path
            path = "../../../../Yellowstone/"+str(self.model.path)+"/"+self.model.path[4:]+"-groundsurf.h5"
        
            #prepare lists for getting data from hdf5 files
            with h5py.File(path, "r") as f:
                group_geometry = f['geometry']
                group_vert_fields = f['vertex_fields']
                
                points = group_geometry['vertices'] #shape: point_num, xyz
                velocities = group_vert_fields['velocity'] #shape: timestep, point_num, xyz
            
                x = points[:][:,0]
                y = points[:][:,1]

                vel_z = velocities[model_time][:][:,2]

                # get indices of points in region
                idx_list = []
                for i in range(len(x)):
                    if inner_axes == None:
                        if (x[i]-offset[0]*1e3)**2/(outer_axes[0]*1e3)**2 + (y[i]-offset[1]*1e3)**2/(outer_axes[1]*1e3)**2 <= 1:
                            idx_list.append(i)
                    else:
                        if(x[i]-offset[0]*1e3)**2/(outer_axes[0]*1e3)**2 + (y[i]-offset[1]*1e3)**2/(outer_axes[1]*1e3)**2 <= 1 and (x[i]-offset[0]*1e3)**2/(inner_axes[0]*1e3)**2 + (y[i]-offset[1]*1e3)**2/(inner_axes[1]*1e3)**2 > 1:
                            idx_list.append(i)

                avg_vel = np.mean(vel_z[np.asarray(idx_list)]*self.ms_to_mmyr)

                self.source_velocity[time_step] = avg_vel # add the average velocity to the dictionary

                return avg_vel
        else:
            return self.source_velocity[time_step]
        

############################## Class Analysis #######################################################################   

# helper function to generate x-y locations of the surface projection of the CR
# assumed to be a cyllindrical CR unless CR_y is passed in then it's an ellipsoidal CR
def get_CR_points(CR_x, CR_y=None):

    theta_list = np.arange(0, 2*np.pi+0.1, 0.1)
    if CR_y == None:
        x = CR_x*np.cos(theta_list)
        y = CR_x*np.sin(theta_list)
    else:
        x = CR_x*np.cos(theta_list)
        y = CR_y*np.sin(theta_list)
    return x, y

# Function to plot non-dimensional velocity profiles from different models at a given time and velocity scaling trend from lists of different models
def plot_nondim_profiles(model_list, time, add_models_list=None, burgers_models_list=None, add_models_times=None, add_colors=None, add_symbols=None, add_labels=None,
                          add_linestyles=None, add_line_colors=None, burgers_colors=None, burgers_symbols=None, burgers_labels=None, burgers_linestyles=None, theta=None):

    # get x_0 for plotting
    (x_0, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model_list[0][0].get_data(0, theta=None)

    # set up line styles and colors for plotting
    dpdt_colors = ["blue", "orange", "green", "brown"]
    visc_linestyles = ["solid", "dotted", "dashed"]

    # set up max vel plot
    max_norm_vels = []

    # plot 1E17 visc models
    dpdt_itr = 0
    max_norm_vel = 0
    for model in model_list[0]:
        model_time = model.get_timesteps(time)
        (_, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time, theta=theta)

        #Tr needs to be shifted by a factor 4!
        dynamic_visc = 4*model.model.tr*3.154e+7*10e9 # Pa*s
        kinematic_visc = dynamic_visc/2500 #m^2/s
        #norm_factor = np.sqrt(kinematic_visc*model.model.T*3.154e+7)/kinematic_visc # s/m
        # norm_factor = 5e3/kinematic_visc # s/m
        norm_factor = 3.154e+7*(1/model.model.Delta_P)*model.model.P0/np.sqrt(kinematic_visc*4*model.model.tr*3.154e+7)
        # norm_factor = 1/np.sqrt(model.model.Delta_P*model.model.tr*3.154e+7/2500)
        norm_vel = vel_z*norm_factor
        if max(norm_vel) >= max_norm_vel:
            max_norm_vel = max(norm_vel)
        plt.plot(x_0/5e3, norm_vel, linestyle=visc_linestyles[0], color=dpdt_colors[dpdt_itr], linewidth=6, label=r"dP/dt="+str(model.model.Delta_P/1e3)+" kPa/yr,$\eta_{CR}$="+str(dynamic_visc)+" Pa*s")
        dpdt_itr += 1
    max_norm_vels.append(max_norm_vel)


    # plot 5E16 visc models
    dpdt_itr = 0
    max_norm_vel = 0
    for model in model_list[1]:
        model_time = model.get_timesteps(time)
        (_, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time, theta=theta)

        dynamic_visc = 4*model.model.tr*3.154e+7*10e9 # Pa*s
        kinematic_visc = dynamic_visc/2500 #m^2/s
        # norm_factor = np.sqrt(kinematic_visc*model.model.T*3.154e+7)/kinematic_visc # s/m
        # norm_factor = 5e3/kinematic_visc # s/m
        norm_factor = 3.154e+7*(1/model.model.Delta_P)*model.model.P0/np.sqrt(kinematic_visc*4*model.model.tr*3.154e+7)
        # norm_factor = 1/np.sqrt(model.model.Delta_P*model.model.tr*3.154e+7/2500)
        norm_vel = vel_z*norm_factor
        if max(norm_vel) >= max_norm_vel:
            max_norm_vel = max(norm_vel)
        plt.plot(x_0/5e3, norm_vel, linestyle=visc_linestyles[1], color=dpdt_colors[dpdt_itr], linewidth=6, label=r"dP/dt="+str(model.model.Delta_P/1e3)+" kPa/yr,$\eta_{CR}$="+str(dynamic_visc)+" Pa*s")
        dpdt_itr += 1
    max_norm_vels.append(max_norm_vel)

    # plot 1E16 visc models
    dpdt_itr = 0
    max_norm_vel = 0
    for model in model_list[2]:
        model_time = model.get_timesteps(time)
        (_, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time, theta=theta)

        dynamic_visc = 4*model.model.tr*3.154e+7*10e9 # Pa*s
        kinematic_visc = dynamic_visc/2500 #m^2/s
        # norm_factor = np.sqrt(kinematic_visc*model.model.T*3.154e+7)/kinematic_visc # s/m
        # norm_factor = 5e3/kinematic_visc # s/m
        norm_factor = 3.154e+7*(1/model.model.Delta_P)*model.model.P0/np.sqrt(kinematic_visc*4*model.model.tr*3.154e+7)
        # norm_factor = 1/np.sqrt(model.model.Delta_P*model.model.tr*3.154e+7/2500)
        norm_vel = vel_z*norm_factor
        if max(norm_vel) >= max_norm_vel:
            max_norm_vel = max(norm_vel)
        plt.plot(x_0/5e3, norm_vel, linestyle=visc_linestyles[2], color=dpdt_colors[dpdt_itr], linewidth=6, label=r"dP/dt="+str(model.model.Delta_P/1e3)+" kPa/yr,$\eta_{CR}$="+str(dynamic_visc)+" Pa*s")
        dpdt_itr += 1
    max_norm_vels.append(max_norm_vel)

    plt.grid()
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.xlabel(r"$x/d_{source}$", fontsize=20)
    # plt.ylabel(r"$V_z d_{source}/\nu$", fontsize=20)
    # plt.ylabel(r"$V_z \sqrt{\nu T}/\nu$", fontsize=20)
    plt.ylabel(r"$V_z\left(\frac{dP}{dt}\right)^{-1}P_0/\sqrt{\nu \tau}$", fontsize=20)
    # plt.ylabel(r"$V_z/\sqrt{\frac{dP}{dt}\tau/\rho}$", fontsize=20)
    plt.legend(bbox_to_anchor=(1.15, 1.0), loc='upper left')
    plt.savefig("/home/grantblock/Research/SMBPylith/Figures/nondim_profiles.png", bbox_inches="tight")
    plt.show()

    # plot trend with relax time
    norm_relax_times = 4*np.asarray([(model_list[0][0].model.tr/model_list[0][0].model.T), (model_list[1][0].model.tr/model_list[1][0].model.T), 
                                   (model_list[2][0].model.tr/model_list[2][0].model.T)])

    # calculate dP/dt
    v_norm = max_norm_vels[2]
    tr = 4*model_list[2][0].model.tr*3.154e+7 # s
    v_GPS = 50/model_list[2][0].ms_to_mmyr # m/s
    kinematic_visc = (tr*40e9)/2500 # m^2/s

    dpdt = (model_list[2][0].model.P0*v_GPS)/(v_norm*np.sqrt(kinematic_visc*tr))*(3.154e+7/1e3) # kPa/yr
    print(dpdt)

    # plt.scatter(norm_relax_times, max_norm_vels, label=r"Upside down sawtooth time series, $d_s$=5 km", c='blue', s=75)
    plt.scatter(norm_relax_times, max_norm_vels, label=r"Upside down sawtooth time series, $d_s$=5 km", c='gray', s=300, marker="^", edgecolors='black', zorder=6)
    plt.plot(norm_relax_times, max_norm_vels, c='gray', lw=3)

    # if there are additional models to plot on the trend
    if add_models_list != None:
        list_itr = 0
        for models in add_models_list:
            add_max_norm_vels = []
            add_norm_relax_times = []
            for model in models:
                model_time = model.get_timesteps(add_models_times[list_itr])
                (_, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time, theta=theta)
                dynamic_visc = 4*model.model.tr*3.154e+7*10e9 # Pa*s
                kinematic_visc = dynamic_visc/2500 #m^2/s
                norm_factor = 3.154e+7*(1/model.model.Delta_P)*model.model.P0/np.sqrt(kinematic_visc*4*model.model.tr*3.154e+7)
                norm_vel = vel_z*norm_factor
                add_max_norm_vels.append(max(norm_vel))
                add_norm_relax_times.append(4*model.model.tr/model.model.T)

            plt.scatter(add_norm_relax_times, add_max_norm_vels, c=add_colors[list_itr], marker=add_symbols[list_itr], label=add_labels[list_itr], s=300, edgecolors='black', zorder=7)
            plt.plot(add_norm_relax_times, add_max_norm_vels, c=add_line_colors[list_itr], linestyle=add_linestyles[list_itr], lw=3)
            list_itr += 1

    if burgers_models_list != None:
        MU = 1.5625e10 # models' shear modulus since update of Vs (Pa)
        list_itr = 0
        for models in burgers_models_list:
            norm_vels = []
            norm_relax_times = []
            for model, visc in models:

                model_time = model.get_timesteps(520)
                (_, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time) 

                # get non-dimensional velocity
                dynamic_viscosity = visc # Pa*s
                kinematic_viscosity = dynamic_viscosity/2500 # m^2/s
                tr = dynamic_viscosity/MU # s
                norm_factor = 3.154e+7*(1/model.model.Delta_P)*model.model.P0/np.sqrt(kinematic_viscosity*tr)
                norm_vel = vel_z*norm_factor
                norm_relax_time = tr/(model.model.T*3.154e+7)
    
                norm_vels.append(max(norm_vel))
                norm_relax_times.append(norm_relax_time)

            plt.scatter(norm_relax_times, norm_vels, c=burgers_colors[list_itr], marker=burgers_symbols[list_itr], label=burgers_labels[list_itr], s=300, edgecolors='black', zorder=8)
            plt.plot(norm_relax_times, norm_vels, c='black', linestyle=burgers_linestyles[list_itr], lw=3)
            list_itr += 1

    
    plt.grid()
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.ylabel(r"$V_{max}\left(\frac{\Delta P}{\Delta t}\right)_{max}^{-1}P_0/\sqrt{\nu \tau}$", fontsize=40)
    plt.xlabel(r"$\tau/T$", fontsize=30)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(fontsize=35, bbox_to_anchor=(1.15, 1.0), loc='upper left')
    plt.savefig("/home/grantblock/Research/SMBPylith/Figures/nondim_profiles_trend.png", bbox_inches="tight")
    plt.show()

# Function to make plots comparing models used for mesh resolution tests
def mesh_resolution_analysis(model_list, avg_S, plot_time, ratio_list=None):

    ms_to_mmyr = 3.154e+10

    # get x_0 for plotting profiles
    (x_0, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model_list[0].get_data(0, theta=None)

    # set up first plot
    plt.xlabel("Distance From Center of Source (km)", fontsize=25)
    plt.ylabel("Velocity (mm/yr)", fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.grid()

    # iterate through models and plot profiles
    itr = 0
    max_vel_list = []
    for model in model_list:
        model_time = model.get_timesteps(plot_time)
        (_, disp_x, disp_y, disp_z, disp_r, disp_theta, vel_x, vel_y, vel_z, vel_r, vel_theta) = model.get_data(model_time, theta=None)

        max_vel_list.append(max(vel_z)*ms_to_mmyr) # collect max vels for second plot

        if ratio_list == None:
            plt.plot(x_0/1e3, vel_z*ms_to_mmyr, lw=5, label=r"$\langle S\rangle$="+str(avg_S[itr])+" km, t="+str(plot_time)+" yrs")
        else:
            plt.plot(x_0/1e3, vel_z*ms_to_mmyr, lw=5, label=r"$\langle S\rangle$="+str(avg_S[itr])+r" km, $R_v=$" + str(ratio_list[itr]) + ", t="+str(plot_time)+" yrs")

        itr += 1
    
    plt.xlim([-60, 60])
    plt.legend(fontsize=25, bbox_to_anchor=(1.25, 1.1))
    plt.savefig("/home/grantblock/Research/Yellowstone/Figures/res_test_legend.png", bbox_inches="tight")
    plt.show()

    # set up second plot
    plt.xlabel(r"$\langle S\rangle$ (km)", fontsize=30)
    plt.ylabel(r"$V_z(x=0,y=0,t="+str(plot_time)+r"$) (mm/yr)", fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid()

    plt.plot(avg_S, max_vel_list, lw=6)
    plt.scatter(avg_S, max_vel_list, s=200, edgecolors='black')
    plt.show()
