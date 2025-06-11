# 🛰️ Bachelor Thesis

## 📌 About the Project  
This repository contains code, data, and documentation for our **bachelor thesis** at the **Technical University of Denmark (DTU)** in collaboration with the **Danish Meteorological Institute (DMI)**. The project explores the use of **simulated CIMR (Copernicus Imaging Microwave Radiometer)** and **MWI (Microwave Imager)** data for **sea ice applications**.

## 🎯 Objectives  
- **Process & analyze** simulated CIMR and MWI data  
- **Investigate** their potential for **retrieving sea ice concentration (SIC) and snow depth**  
- **Compare** simulated observations with outputs from the **CICE sea ice model**  
- **Evaluate** retrieval algorithm performance under different ice conditions  
- **Quantify** the impact of footprint resolution on sea ice estimates  
- **Support** development of high-resolution satellite products for climate research  

## 📂 Data  
All data used and generated in this project can be found here:  
📎 [Google Drive Data Folder](https://drive.google.com/drive/folders/1Eu3oZKTxjCPQAR15tLhq0fMi_W31Y2x-?usp=sharing)

Contents include:
- SMRT brightness temperature simulations  
- Sea ice and snow property inputs from DMI-CICE  
- CIMR and MWI frequency-specific sensor configurations  
- Retrieval outputs from SIC and snow depth algorithms  

## ⚙️ Methods  
- **SMRT Radiative Transfer Modeling** for passive microwave simulation  
- **Snow/Ice Layering Schemes** based on literature and model outputs  
- **Sensor Emulation** for CIMR (high-resolution, wide-swath) and MWI (lower resolution)  
- **Retrieval Algorithms** tested:  
  - Bristol SIC algorithm  
  - Rostosky et al. (2018) snow depth algorithm  
- **Footprint Mismatch Assessment** for evaluating coarse vs. fine spatial resolutions  

## 📊 Key Results  
- Coarser resolution (MWI) caused significant overestimation of sea ice extent  
- CIMR's 5 km resolution preserved key features in marginal ice zones  
- Bristol algorithm was sensitive to snow depth on FYI, leading to positive biases  
- Rostosky snow depth retrieval failed on MYI and bare ice, indicating a need for recalibration  
- Combining data from both sensors may reduce regional uncertainty in retrieval products  

## 📚 References  
Select references:  
- Shokr & Sinha (2015) – *Sea Ice: Physics and Remote Sensing*  
- Ulaby et al. (1986) – *Microwave Remote Sensing: Active and Passive*  
- Rostosky et al. (2018) – Snow depth retrieval algorithm  
- Galeazzi et al. (2023) – CIMR mission overview  
- Meier & Stewart (2019) – Uncertainty from footprint mismatch  
- Wernecke et al. (2024) – Passive microwave intercomparison studies

Full reference list is available in the accompanying thesis PDF.

## 👩‍🔬 Authors  
- Ida Grum-Schwensen Andersen  
- Josephine Juul

Supervised by:  
- **Dr. Rasmus Tonboe** (DTU Space)  
- **Dr. Till Rasmussen** (DMI)

## 🌍 Relevance  
This project supports the development of next-generation Earth observation missions for Arctic sea ice monitoring. It highlights the importance of high-resolution radiometry and physically consistent retrieval methods for accurate climate assessment.

