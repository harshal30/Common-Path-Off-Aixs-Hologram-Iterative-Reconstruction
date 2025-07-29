# Iterative Off-Axis Hologram Reconstruction using Python

This repository provides a Matlab and Python code implementation of an **iterative hologram reconstruction algorithm** recorded in common path off-axis digital holographic microscopy (DHM). An iterative algorithm enhance the quality of phase estimation in the focal plane by suppressing the contributions of those from out-of-focus planes and also provides denoised phase estimates.


**Prerequisite**

The focal plane of the object present in both beams

**Data**

Add the recorded hologram in the data folder

**Usage**

Run Iterative_reconstruction.m

The helper function are also provided.

**Select the +1 diffraction order region:**

An interactive window will open for selecting the region in the Fourier spectrum. Drag the box around the +1 order and press Enter.

**Output:**

Final amplitude and phase reconstructions at two propagation distances.

Phase profile plots comparing conventional and iterative results.