# Scanner-Independent MyoMapNet for Accelerated Cardiac T1 Mapping Across Vendors and Field Strength

Purpose: To develop and evaluate an accelerated cardiac T1 mapping cardiac MR approach based on MyoMapNet, a convolution neural network T1 estimator, that can be used across different vendors and field strengths by incorporating the relevant scanner information as additional inputs to the model.

Materials and methods: The proposed scanner-independent (SI)-MyoMapNet is a deep fully convolutional U-Net model that generate T1 map from ten inputs: four T1-weighted images and their corresponding inversion times, vendor, and field strength. In a retrospective study, we collected Modified Look-Locker (MOLLI) images from patients undergoing clinical CMR using Siemens or Philips, and 1.5T and 3T from three medical centers (1249 patients using Siemens 3T (MAGNETOM Vida) in Binstitution1, 99 patients using Siemens 1.5T (MAGNETOM Sola fit), in institution2, and 75 patients using Philips 1.5T (Achieva) from institution3. Patients were divided into training/validation (N=853/285) and testing (N=285), having similar representation of vendors and field strengths. The myocardial and blood T1 were measured manually and compared between SI-MyoMapNet and MOLLI using regression and Bland-Altman analyses. 

Results: The proposed SI-MyoMapNet successfully created T1 maps. Native and post-contrast T1 values, measured from SI-MyoMapNet, were strongly correlated with MOLLI, despite using only 4 T1w images at both field-strengths and vendors (all r >0.86). In Bland-Altman analysis, SI-MyoMapNet and MOLLI were in good agreement for myocardial and blood T1 values. 

Conclusion: Inclusion of field-strength and vendor as additional inputs to the deep learning architecture improves generalizability of MyoMapNet across different vendors or field strength.


![Fig1_architecture](https://user-images.githubusercontent.com/9512423/196737339-444a4867-beca-4dc6-a472-bd39fef1e201.png)
