# mspaef_metric_evaluation
Evaluation files for MSPAEF metric and CMIP6 implementation

This project contains six Python files used for evaluating the MSPAEF metric and analyzing CMIP6 model performance.

## Files Description:

* **`Metric_evaluation.py`**:
    * Evaluates the performance of four metrics.
    * **Requires:** The `out_path` variable must be set to a directory where figures will be exported.

* **`Metric_counterexample.py`**:
    * Generates two realistic synthetic examples for comparing the four metrics.
    * **Requires:** The `out_path` variable must be set to a directory where figures will be exported.

* **`Good_example.py`**:
    * Creates example behaviors for curve visualization.
    * **Requires:** The `plt.savefig()` function call must be modified to include a desired file path and filename for figure export.
    * **Usage:** Modify the `plt.savefig()` line within the script.

* **`CMIP6_models_metric_calc.py`**:
    * Reads netCDF files from Climate Explorer and calculates metric values for CMIP6 models.
    * **Requires:**
        * Set `var_name` to either `'tas'` (temperature) or `'pr'` (precipitation) to run the script for the respective variable.
        * Set `out_path` to the desired output directory for the CSV file.
        * Set `path` to the directory containing the input netCDF files.
        * Set `path_comp_pr_direct` to the path of the comparison precipitation netCDF file.
        * Set `path_comp_tas_direct` to the path of the comparison temperature netCDF file.
        * Set `path_temp` to a directory for temporary files used during conservative interpolation.
    * **Output:** A CSV file containing metric values for all CMIP6 models.

* **`CMIP6_metrics_plot_table.py`**:
    * Plots precipitation vs. temperature values for the MSPAEF metric.
    * Generates CSV files with model rankings for the four metrics.
    * **Requires:**
        * Set `path_in` to the directory containing the input CSV files.
        * Set `path' to the output directory for figures and ranked CSV files.
        * Set `file_name_pr` to the filename of the precipitation metric data CSV file.
        * Set `file_name_tas` to the filename of the temperature metric data CSV file.
    * **Output:** Figure and ranked CSV files.

* **`Ranking_differences_CMIP6.py`**:
    * Plots the differences in CMIP6 model rankings between the four metrics.
    * **Requires:**
        * Set `path_in` to the directory containing the input CSV files.
        * Set `path` to the output directory for the figure.
        * Set `file_name_pr` to the filename of the precipitation metric data CSV file (with ranks).
        * Set `file_name_tas` to the filename of the temperature metric data CSV file (with ranks).
    * **Output:** Figure showing ranking differences.

## Dependencies:
* numpy
* scipy
* matplotlib
* netCDF4
* pandas
* os
* random
* cmocean
* warnings
* scipy
* tqdm
* sklearn

## Usage Instructions:
1.  Run `Metric_evaluation.py` and `Metric_counterexample.py` to generate the metric evaluation and counterexample figures.
2.  Run `Good_example.py` to generate the good example figure.
3.  Download CMIP6 data from Climate Explorer and ERA5 from Copernicus CDS.
4.  Place the downloaded netCDF files in the directory specified by the `path` variable in `CMIP6_models_metric_calc.py`.
5.  Run `CMIP6_models_metric_calc.py` to generate the metric CSV files.
6.  Run `CMIP6_metrics_plot_table.py` to generate plots and ranked CSV files.
7.  Run `Ranking_differences_CMIP6.py` to visualize ranking differences.

