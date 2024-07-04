- Boutons_interval:
	Bouton-bouton distance (in microns) for each parallel fiber imaged.


- Calbindin_L7-tdTomato_PCs_counting:
	Number and percentage of Purkinje cells labeled by both calbindin and tdTomato (merge) from L7-tdTomato mice.


- GluSnFR_avg_variables_1.5_4mM_Ca_paired_filtered_3sigma:
	Amplitudes, ppr, failures from boutons imaged at 1.5mM and 4mM Ca2+ concentration.


- GluSnFR_avg_variables_allCa_filtered_3sigma:
	Amplitudes, ppr, failures and other measures from boutons imaged at 1.5mM, 2.5mM and 4mM Ca2+ concentration
	during 20Hz or 50Hz extracellular stimulation.


- iGluSnFR_avg_traces_allCa_filtered_3sigma:
	Stores the average fluorescent trace and time from each bouton imaged at 2.5mM and 4mM Ca2+ concentration during 20Hz or 50Hz
	extracellular stimulation.


- iGluSnFR_Tau_averages_allCa:
	Decay-time values from average fluorescent transients at 1.5, 2.5 and 4mM Ca2+ concentration during 20Hz or 50Hz extracellular stimulation.


- PCA_HCPC_6clusters_2.5mM_20Hz_3sigma:
	Amplitudes, PPR values, failures, quantal release for A1, actual and shuffled clusters from the boutons imaged at 2.5mM Ca concentration
	and 20Hz extracellular stimulation.


- Proba_active_boutons_at_least_4_boutons_per_PF:
	Boutons state (active:1, silent:0) for the parallel fibers imaged.


- Proportion_PF_clusters:
	Number of parallel fibers showing 1 to 6 classes of boutons and number of boutons in each class.
	Only parallel fibers with at least 4 boutons are stored.


- PSF_2P_laser:
	Sheet1: Measures from all the microbeads imaged.
	Sheet2 and 3: fluorescence intensity (grey values) as a function of distance (in microns) from the line bisecting the microbead (20220724_bead2).
			Central and axial resolutions. The last tzo columns store the X and Y values of the gaussian fit.


- Random_Forest_scores:
	Test scores from random forest for actual and shuffled data.


- Saturation_data:
	Fluorescent traces from boutons imaged at 2.5mM and 4mM Ca2+ concentration during 200Hz extracellular stimulation.
	Fmax values are stored in sheets '2.5mMCa_amps' amd '4mMCa_amps'.


- Targets distribution for each cluster:
	Number of GC-PC and GC-nonPC synapses in each class of boutons (C1 to C6).