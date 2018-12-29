# GSP_energy_disaggregator

## Disclaimer
 This python code has been implemented by translating the Matlab code developed by the authors of the paper titled "On a training-less solution for non-intrusive appliance load monitoring using graph signal processing". Note I don't have Matlab code with me now so please don't email me for the Matlab files. 

## Usage
The directory contains two py files: (i) gsp_disaggregator.py: It is the main file.  (ii) gsp_support.py: It contains the supporting functions called by gsp_disaggregator. Also, it contains a demo data file used by gsp_disaggregator.py. Please read comments provided in the gsp_disaggregator.py to understand what different functions do.

## Citation
If you use this copy of code in your work then cite "On a training-less solution for non-intrusive appliance load monitoring using graph signal processing" paper.

## Releases
2018-12-29 Added the ability to label disaggregated appliances by comparison with a signature database using DTW or FastDTW and some small features. @aleonnet
