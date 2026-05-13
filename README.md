# Sparse-to-Dense-Wall-Mapping-with-Beamsteered-mmWave-Sensing-and-GAN-Upsampling
This project presents a indoor environment reconstruction framework using beamsteered mmWave radar sensing, TDM-MIMO spatial processing, and GAN-based sparse-to-dense upsampling for wall and structural mapping.

**Key Features**
* Beamsteered mmWave sensing using multiple steering angles
   (-30°, -15°, 0°, +15°, +30°)
* TDM-MIMO processing for:
Azimuth estimation
Elevation estimation
3D spatial localization
* Sparse radar point cloud generation from:
Range FFT
Doppler FFT
Angle FFT / Beamforming
* Multi-angle probabilistic fusion of radar observations
* Sparse-to-dense wall reconstruction using:
GAN-based upsampling
Spatial interpolation
Occupancy enhancement
* Indoor wall and structural mapping in:
LOS environments
NLOS environments
* Support for real-time or offline radar data processing

**System Pipeline**
* Raw ADC capture from mmWave radar
* Beamsteering configuration sweep
* TDM-MIMO virtual array processing
* Range–Doppler–Angle estimation
* Sparse point cloud extraction
* Multi-view spatial fusion
* Occupancy grid generation
* GAN-based dense reconstruction
* Dense wall map visualization

**Hardware**
* TI IWR1843BOOST and DCA1000EVM
* UART + Ethernet data capture
* Beamsteering-enabled antenna configuration

**Software Stack**
Python, NumPy, SciPy, OpenRadar, PyTorch / TensorFlow, mmWave DSP processing

**Research Focus
This work explores how directional beamsteering combined with probabilistic multi-angle fusion can improve sparse radar perception and how deep generative models can reconstruct dense indoor wall structures from incomplete radar observations.
The project is particularly focused on:
*Sparse radar sensing
*mmWave spatial reconstruction
*Radar-based scene understanding
*Deep learning assisted occupancy enhancement
*3D indoor mapping using RF sensing
