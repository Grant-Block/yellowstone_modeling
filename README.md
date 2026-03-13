This repository contains the codes, data files, and selected PyLith model input files used to conduct the analysis and generate the figures in 
"Decadal-scale surface motions at Yellowstone within a compliant region produced by magmatic alteration".

The folder Data_analysis_scripts contains all data and code related to analyzing and plotting cGPS data (everything not having to do with the models). The script
Analyze_GPS_final.py contains and specifies each function call used to make a specific figure or panel (e.g. Fig. 1a). Each function is documented in the code.

The folder GPS_data contains the unprocessed cGPS data that is processed and used in Analyze_GPS_final.py.

The folder Data_fig2d contains smoothed individual cGPS velocities, as well as inner, anticorrelated, and regional group velocities which are used for making figures 3 and 4.
The script makeObsFig_MRnew.py uses this data to generate those plots and is internally documented. 

The folder ModelAnalysis_final contains all scripts and data necessary to analyze and plot model results with the caveat of the completed model .h5 files themselves.
These files are too large to upload here, but selected input files to run the models are included (see below). If you would like any model .h5 files or other input
files that are not included, please email the corresponding author at mroy@unm.edu. The script RunAnalysis_final.py contains all function calls used to conduct the  analysis and generate figures having to do with model results (e.g. Figures 5,6). It documents which function call generates which figure. It calls functions from the scripts AnalyzeModel_final.py ad ModelStorage_final.py which are both internally documented.

The folder Quiver_CSV stores data files to generate the quiver plots of Figure S11. The script which generates the panels is make_quiver_from_slice.m. 

The folders Mesh, timedb, and pylith_input contain input files needed to run selected PyLith models (given in Figure S5, Table S3). pylith_input contains the .cfg
files to run the models, while Mesh and timedb contain other input files that the .cfg calls. Each file is consistently named by run number. Again, if you would like results from model runs (.h5 files) or additional model input files not included here, please email the corresponding author at mroy@unm.edu. 

