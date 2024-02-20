# MIRI-HD141569


These are data processing code dedicated to the analysis of the MIRI datasets of HD 141569, which require a specific treatment with respect to PSF subtraction.
It contains two files:
- HD141569_MIRI_Data_analysis.py : this is the dedicated code to process the MIRI data for HD 141569. It contains basic steps to process the data starting from JWST pipeline Stage 2 outputs.
- mirisim_disk_model.py : this is a code to create MIRI disk models, to be use for (forward) modeling analyses.

SETTING UP:
It is recommended to set up a dedicated conda environment for this work.
- Install the dev veersions  of webbpsf_ext, webbpsf, poppy:

```
pip install -e git+https://github.com/spacetelescope/poppy.git#egg=poppy
pip install -e git+https://github.com/spacetelescope/webbpsf.git#egg=webbpsf
pip install -e git+https://github.com/JarronL/webbpsf_ext#egg=webbpsf_ext
```
