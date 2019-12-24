# coding=utf-8
"""
date:190915
created by takahashi & ishida
いろんな作業を他の.pyファイルに投げる司令官的ぽじしょんのかわいい子だよ
Extended by @Ittan_moment
update:20191021
"""


import makeroicsv as mric
import analysisfiring as af
import detectfiring as df


class AnalysisRunClass:
    def __init__(self, csv_save_path, graph_save_path, rec_time, data_name, histogram_bins=50):
        self.csv_save_path = csv_save_path
        self.graph_save_path = graph_save_path
        self.rec_time = rec_time
        self.histogram_bins = histogram_bins
        self.data_name = data_name

    def analysis_from_unzipped_cxd_data(self, cxd_path, x, y, z, data_pixel_height, data_pixel_width):
        c = mric.MakeRoiIntensityCSV(cxd_path=cxd_path,
                                data_name=self.data_name,
                                x=x,
                                y=y,
                                data_pixel_width=data_pixel_width,
                                data_pixel_height=data_pixel_height)
        c.make_roi_csv(csv_save_path=self.csv_save_path)
        c.check_roi_areas_visually(heatmap_save_path=self.graph_save_path)

        """
        d = df.DetectFiring(roi_intensity_csv_file=c.name_of_roi_intensity_csv_file,
                         rec_time=self.rec_time,
                         graph_save_path=self.graph_save_path,
                         csv_save_path=self.csv_save_path)
        d.plot_mean_intensity_from_raw_data(z=z,data_name = self.data_name)
        d.detect_firing_only_raw_data(result_csv_save_path=self.csv_save_path, standard_max_rise_time_steps=40,
                              peak_detect_sensitivity=30)

        e = af.AnalysisFiring(roi_intensity_csv_file=c.name_of_roi_intensity_csv_file,
                   firing_result_csv_file=d.name_of_firing_result_csv_file,
                   sigma_mean_csv_file=d.name_of_sigma_mean_csv_file,
                   graph_save_path=self.graph_save_path,
                   rec_time=self.rec_time)
        e.network_analysis(CellPercent=0.75, thres_burst_time=6)
        e.burst_analysis()
        e.raster_plot(z=z)
        """

    def analysis_from_roi_intensity_csv(self, roi_intensity_csv_file,z):
        d = df.DetectFiring(roi_intensity_csv_file=roi_intensity_csv_file,
                         rec_time=self.rec_time,
                         graph_save_path=self.graph_save_path, csv_save_path=self.csv_save_path)
        d.plot_mean_intensity_from_raw_data(z=z, data_name=self.data_name)
        d.detect_firing_only_raw_data(result_csv_save_path=self.csv_save_path, standard_max_rise_time_steps=40,
                                      peak_detect_sensitivity=30)
        e = af.AnalysisFiring(roi_intensity_csv_file=roi_intensity_csv_file,
                           firing_result_csv_file=d.name_of_firing_result_csv_file,
                           sigma_mean_csv_file=d.name_of_sigma_mean_csv_file,
                           graph_save_path=self.graph_save_path,
                           rec_time=self.rec_time)
        # e.plot_mod_mean_and_threshold()
        e.network_analysis(CellPercent=0.75, thres_burst_time=6)
        e.burst_analysis()
        e.raster_plot(z=z)
