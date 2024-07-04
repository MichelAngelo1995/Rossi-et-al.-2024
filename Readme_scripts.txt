A total of 22 scripts written on Python3.9/Spyder5.4.1 have been used to analyse the datasets from the manuscript Rossi et al., 2024.
The following lines indicate how to use the Graphical User Interfaces and the other scripts used to create the figure panels.


	I. Extractor: Graphical User Interface created with the module PySimpleGUI. Extracts fluorescent traces over time across each linescan from each bouton along the parallel fiber.
		Figures: All.
		Folder required:
			Example: 'GroupLinescan1_20Hz_10pulses_2.5mMCa'
				1. On panel 1, zoom on a bouton.
				2. Left click on the red line (linescan) bisecting the bouton. An arrow appears on panel 2 (first linescan, e.g. 0/6)
					pointing the location of the bouton. Y-axis shows the position on the red line in panel 1. X-axis shows
					the repetition of the red line (i.e. time). The fluorescent trace, expressed as arbitrary units over time, is shown on panel 4.
				3. Right click on a dark area on panel 2 to extract the background noise (Fback, indicated by the letter 'f'). The corresponding trace appears
					on panel 3 where the mean of the first 100 ms is shown by the thick blue line.
				4. Press 'Enter' to validate the first trace and the Fback value. The fluorescent trace appears on panel 5.
				5. Scroll to the second linescan. On panel 2, right click on a dark area to select Fback, then zoom to the location of the bouton and
					scroll click on the fluorescent smir. A second arrow appears.
				6. Scroll to the third linescan. The fluorescent trace appears on panel 5 and overlays to the previous trace.
				7. Repeat steps 5 and 6 until fluorescent traces from all linescans are shown in panel 5.
				8. On the monitor window, click on 'Browse' next to 'Saving path' to select a folder where the data have to be stored. Fill the different inputs.
				9. Click on 'To Excel'. The traces are stored in an excel file (traces.xlsx) containing two sheets: 'Traces' and 'Data'. In 'Traces', each episode is stored as
					a column and time is stored in the last column. In 'Data', for each episode, the position of the bouton is given in the 'Trace_index' row,
					while the position and value of Fback are given in 'Fback_index' and 'Fback_value' rows, respectively.
			

	II. Converter: Graphical User Interface created with the module PySimpleGUI. Converts fluorescent traces from arbitrary units to DF/F0 values.
		Figures: All.
		Files required:
			'[...]traces.xlsx'
				1. Browse both the file created with Extractor and the folder to save the data.
				2. Click on 'Go'. A new window appears. Select the episodes to keep or select 'All' to keep all the episodes.
				3. Set the sigma threshold X and click 'Check_traces' to chech if the F0 values of each episode are in the range -Xsigma < mean > +Xsigma.
				4. Click 'ANALYSE' to create a new excel file (traces_converted.xlsx) containing the following three sheets: 'Traces no Fback', 'Traces DF_F0', 'F0'.


	III. SynaptiPs_v1.5_2 : Graphical User Interface created with the module PySimpleGUI. Calculates the transients amplitudes, decay-times, suppresses photobleaching.
		Figures: All.
		Files required:
			'[...]traces_converted.xlsx'
				The excel files created are named 'traces_converted_Amp.xlsx' and 'traces_converted_Tau.xlsx'.


	IV. Bootstrap_failures: Graphical User Interface created with the module PySimpleGUI. Determines the release failure from the transients.
		Figures: Fig.S5.
		Files required:
			'[...]traces_converted.xlsx'
				1. Browse both the file created with 'Converter' and the folder to save the data.
				2. Tick 'Show noise & peak#' and write the peak number to specifically visualize the peak and noise traces from the selected peak.
				3. Click 'Start File'. The panel shows the DF/F0 traces from each episode and the average.
				4. Set the Correction settings if the traces need to be corrected for photobleaching, leak and residual (i.e. transients summation). If not, skip this step.
				5. Set the frequency of stimulation, the number of peaks to extract the failures from, and the first peak and noise windows.
				6. Tick 'Show histograms' to visualise the bootstraped histograms from peaks and noise from each transient for all episodes.
				7. Click 'GO' to run the analysis. The new excel file ('traces_converted_data_bootstrap.xlsx' and 'traces_converted_histograms_bootstrap' if step 2 activated)
				contains a sheet/peak showing the results of the analysis.


	V. GIMLI (Graphical user Interface for Machine LearnIng): Graphical User Interface created with the module PySimpleGUI. Used for PCA, Clustering and Prediction analyses.
		Figures: Fig.4a-b, Fig.S7.
		Files required:
			'GluSnFR_avg_variables_allCa_filtered_3sigma.xlsx'
				1. Browse the excel file containing the tidy data ('GluSnFR_avg_variables_all_filtered_3sigma.xlsx').
				2. Click 'GO'. On the 'VARIABLES' window, select the variables from the excel sheet (e.g. '20Hz_2,5mM') that are going to be used for the analysis. If the dataset
					needs to be preprocessed, tick 'Pre-processing' and mention the Imputer strategy for NaN values (e.g. 'median' replaces the NaN values from a given variable
					by the median of the values in that variable).
				3. Click 'ANALYSE'. On the 'CLUSTERING ANALYSIS' window, tick 'PCA_on' and set the 'Theshold" value (from 0 to 1) in the 'DIMENSIONAL REDUCTION' tab to create the number
					of principal components explaining the cumulative variance determined by the threshold (e.g. threshold 0.75 creates X principal components explaining 75% of the total variance).
				4. Click 'RUN PCA'. The figures show the cumulative variance explained and the variance explained by each principal component, the PCA subspace, the correlation circle and
					the contribution of each varible on the principal components.
				5. Set the clustering parameters. In the 'CLASSIFICATION' tab and 'Clustering' subtab, tick the clustering method to use and indicate the parameters (e.g.,for hierarchical clustering,
					give the number of clusters to highlight, the metric and the linkage method).
				6. Click 'PERFORMANCE EVALUATION' to show the elbow plot and the silhouette plot.
				7. Click 'RUN CLUSTERING'. The figures show the PCA subspace with the clusters and the dendrogram if HCPC is ticked.
				8. Save the initial dataset containing the cluster labels in 'SAVING'.
				9. Set the prediction parameters. In the 'CLASSIFICATION' tab and 'Prediction' subtab, click 'Select labels after clustering' to confirm the number of clusters.
				10. Tick the 'Splitting method' and indicate the splitting parameters.
				11. Choose the classifier in 'Model'.
				12. Browse the folder to save the figures in 'Data saving'. The figures show the confusion matrices, the learning and curves, the validation curves and the AUC-ROC curves from each spilt.
					The mean confusion matrix and learning and validation curves are shown in three different figures. The comparison between the cross validation scores and the test scores is shown
					on the figure boxplot. The algorithm creates an excel file containing the test scores for each split.


	VI. RiseTime_SNR_TailResidual:
		Figures: Fig.1f, Fig.S2b
		Files required:
			'GluSnFR_avg_variables_allCa_filtered_3sigma.xlsx'
			'iGluSnFR_avg_traces_allCa_filtered_3sigma.xlsx'


	VII. DecayTime_histogram:
		Figures: Fig.1f.
		Files required:
			'iGluSnFR_Tau_averages_allCa.xlsx'


	VIII. Correlation_metrics_F0:
		Figures: Fig.1g, Fig.S3.
		Files required:
			'GluSnFR_avg_variables_allCa_filtered_3sigma.xlsx'


	IX. Ca_comparison:
		Figures: Fig.1h, Fig.3a, Fig.5b-i.
		Files required:
			'GluSnFR_avg_variables_1.5_4mM_Ca_paired_filtered_3sigma.xlsx'
			'GluSnFR_avg_variables_allCa_filtered_3sigma.xlsx'


	X. Silent Synapses:
		Figures: Fig.2a, Fig.2c.
		Files required:
			'Boutons_interval.xlsx'
			'Proba_active_boutons_at_least_4_boutons_per_PF.xlsx'


	XI. Single_PF_boutons_visualisation:
		Figures: Fig.1d, Fig.1e, Fig.1h, Fig.2b, Fig.5a, Fig.6a-b, Fig.7c-d, Fig.S2a, Fig.S9a.
		Folder required:
			'Boutons_analysis'


	XII. Pooled_boutons:
		Figures: Fig.3b-f, Fig.S6.
		Files required:
			'GluSnFR_avg_variables_allCa_filtered_3sigma.xlsx'


	XIII. RandomForest_ACTUALvsSHUFFLED:
		Figures: Fig.4c.
		Files required:
			'Random_Forest_scores.xlsx'


	XIV. PostClustering_GroupsOfBoutons:
		Figures: Fig.4a, Fig.4d-j, Fig.S9.
		Files required:
			'PCA_HCPC_6clusters_2.5mM_20Hz_3sigma.xlsx'
			'iGluSnFR_avg_traces_allCa_filtered_3sigma.xlsx'


	XV. Vesicular_release:
		Figures: Fig.4k, Fig.7f.
		Files required:
			'GluSnFR_avg_variables_allCa_filtered_3sigma.xlsx"


	XVI. PCA_Clustering_Ca_Target_Gender:
		Figures: Fig.5j, Fig.S8.
		Files required:
			'iGluSnFR_avg_traces_allCa_filtered_3sigma.xlsx'
			'GluSnFR_avg_variables_allCa_filtered_3sigma.xlsx'
			'GluSnFR_avg_variables_1.5_4mM_Ca_paired_filtered_3sigma.xlsx'


	XVII. Proportion_clusters_Target_ParallelFiber:
		Figures: Fig.6c, Fig.7e.
		Files required:
			'Targets distribution for each cluster.xlsx'
			'Proportion_PF_clusters.xlsx'


	XVIII. Target_channels_intensities:
		Figures: Fig.7b.
		Files required:
			Example: '20210308_linescan2_bouton2_ZX.png' and '20210308_15_22_37_linescan2_zstack_XYTZ.ini'


	XIX. Point_Spread_Function:
		Figures: Fig.S1.
		Files required:
			'PSF_2P_laser.xlsx'


	XX. Saturation_figures:
		Figures: Fig.S2e, Fig.S2h, Fig.S2i.
		Files required:
			'Saturation_data'

	XXI. iGluSnFR_stability_time:
		Figures: Fig.S4.
		Folder required:
			'Control_iglusnfr_stability_time'

	XXII. Calbindin_staining:
		Figures: Fig.S10.
		File required:
			'Calbindin_L7-tdTomato_PCs_counting.xlsx'