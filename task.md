Dataset Description
Wheat Disease Multimodal Classification Dataset
Overview
This dataset contains multimodal UAV imagery for the classification of wheat diseases. The data was collected to identify the spread of downy mildew and rust at critical growth stages. The dataset has been pre-processed into three modalities: RGB, Multispectral (MS), and Hyperspectral (HS), allowing for the development of multimodal deep learning models.

Data Acquisition
Dates: May 3, 2019 (Pre-grouting stage) and May 8, 2019 (Middle grouting stage).
Equipment: DJI M600 Pro UAV with an S185 snapshot hyperspectral sensor.
Flight Altitude: 60 meters (Spatial resolution ~4cm/pixel).
Spectral Range: 450-950nm (Visible to Near-Infrared).
Spectral Resolution: 4nm.
Data Modalities
For each sample, three aligned data types are provided:

RGB Images (/RGB)

Format: .png
Description: True-color images generated from the hyperspectral bands (Red: ~650nm, Green: ~550nm, Blue: ~480nm).
Multispectral Data (/MS)

Format: .tif (GeoTIFF)
Bands: 5 bands critical for vegetation health analysis:
Blue (~480nm)
Green (~550nm)
Red (~650nm)
Red Edge (740nm)
NIR (833nm)
Hyperspectral Data (/HS)

Format: .tif (GeoTIFF)
Bands: 125 bands (450-950nm).
Note: While the raw data contains 125 bands, the spectral ends (first ~10 and last ~14 bands) may contain sensor noise.
Dataset Structure
The dataset is organized into Training and Validation sets.

Training Set (/train)
Organized by class folders:

Health: Healthy wheat samples.
Rust: Samples infected with Rust.
Other: Other conditions or background.
Validation Set (/val)
Contains samples with randomized filenames to mask the class labels.

Files: Located in val/RGB, val/MS, and val/HS.
Labels: The ground truth for the validation set is provided in result.csv.
Task
Multimodal Classification: The goal is to classify each image patch into one of three categories using one or more of the provided modalities (RGB, MS, HS). 

Classes:

Health
Rust
Other
Submission Format
A CSV file with the following columns:

Id: The filename (e.g., val_a1b2c3d4.tif)
Category: The predicted class (Health, Rust, or Other)