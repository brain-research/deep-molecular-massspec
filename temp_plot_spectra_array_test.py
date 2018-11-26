import numpy as np
import mass_spec_constants as ms_constants
import plot_spectra_utils

def main():
    max_mass_spec_peak_loc = ms_constants.MAX_PEAK_LOC
    true_spectra = np.zeros(max_mass_spec_peak_loc)
    predicted_spectra = np.zeros(max_mass_spec_peak_loc)
    true_spectra[3:6] = 60
    predicted_spectra[300] = 999
    true_spectra[200] = 780

    test_figure_path_name = '/tmp/test_spectra.png' 
    generated_plot = plot_spectra_utils.plot_true_and_predicted_spectra(
        true_spectra, predicted_spectra, output_filename=test_figure_path_name,
        large_tick_size=True)


if __name__ == '__main__':
  main()
  
