# Digital Image Processing Pipeline for QR Code Extraction

![Tech | Python & OpenCV](https://img.shields.io/badge/Tech-Python_%7C_OpenCV-blue)
![Domain | Computer Vision](https://img.shields.io/badge/Domain-Computer_Vision-purple)
![Status | Course Project](https://img.shields.io/badge/Status-Course_Project-gold)

## Overview
This project implements a Digital Image Processing pipeline using OpenCV for the geometric extraction and decoding of distorted QR markers. The software processes raw, warped, or degraded images containing QR codes and systematically reconstructs them into readable binary matrices. The final output is validated using the PyZbar decoding library.

## Key Highlights & Features
* **Digital Image Processing:** The pipeline implements noise reduction techniques, including Gaussian Blur, to clean the input images. It applies spatial filtering via custom sharpening kernels to enhance edge definitions. The system also utilizes dynamic contrast stretching to normalize pixel intensities prior to analysis. Furthermore, adaptive and standard thresholding methods are used to binarize the image and isolate the QR markers.
* **Topological & Structural Analysis:** The project relies on contour hierarchy analysis to identify the nested shapes characteristic of QR codes. Image moments are computed to accurately locate the geometric centers of candidate patterns. Polygon approximation algorithms are deployed to ensure robust finder-pattern detection by isolating precise square shapes.
* **Planar Homography Transformation:** The software computes perspective transformation matrices based on the identified finder patterns. These matrices are applied to rectify severe perspective warping, flattening the QR code into a square.
* **Spatial Sub-sampling:** The code features a dynamic, localized grid-sampling voting algorithm. This algorithm accurately reconstructs binary matrices from degraded inputs by evaluating an optimal sample ratio within each module block. The script tests multiple standard QR grid sizes (e.g., 21x21, 29x29) to determine the best structural fit based on pixel purity, alignment, and timing patterns.

## Processing Pipeline Steps
The extraction process operates in a step-by-step manner, generating debug outputs for each stage:
1. **Preprocessing:** Grayscale conversion, blurring, contrast stretching, and thresholding.
2. **Finder Pattern Detection:** Locating the three main alignment squares in the corners of the QR code.
3. **Rectification:** Calculating homography points to warp the distorted code into a flat 2D perspective.
4. **Grid Estimation:** Dynamically testing grid sizes to determine the optimal module matrix (e.g., matching the spatial frequency of the QR version).
5. **Bit Extraction:** Using a spatial sub-sampling algorithm to assign a 0 or 1 value to each grid cell.
6. **Decoding:** Reading the reconstructed grid data using PyZbar.

## Requirements
The environment requires the following dependencies to function correctly:
* Python version 3.14 or greater.
* The `numpy` library, version 2.3.5 or higher.
* The `opencv-python` library, version 4.11.0.86 or greater.
* The `pyzbar` package, version 0.1.9 or higher, which is used for final data validation.

## Usage
Ensure your target images are placed in the input directory (e.g., `0_example_qr/`).
Run the main script to process the images. The pipeline will automatically generate separate folders detailing every mathematical step of the transformation, ending with a reconstructed binary grid that is evaluated for readable data.

## Dataset & Acknowledgements
The testing and validation of this computer vision pipeline were conducted using the following open-source dataset:
* **hamidl.** *YOLO-QR-labeled. QR code images labeled for YOLO format.* (2020). Available at: [Kaggle Dataset](https://www.kaggle.com/datasets/hamidl/yoloqrlabeled)

<img width="1721" height="696" alt="image" src="https://github.com/user-attachments/assets/d5257000-f3da-4e23-ab56-2e2aee2a4a9e" />
<img width="1819" height="779" alt="image" src="https://github.com/user-attachments/assets/911fb9c6-0043-4f0f-97f1-9cb8db4803ad" />
