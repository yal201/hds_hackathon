# Sytnthetic data hds_hackathon
This hackaton was part of the Making a Difference With Health Data module for the  2020/21 MSc Health Data Science course at the University of Exeter.
This was done alongside with a team of 4 with other coursemates from the course; Sagar Jhamb, Lee Colson, Abi Bunkum and myself (Yudhis Lumadyo)

## Overview
The task of the hackathon was to create synthethic data with a dataset of choice. 

## Data set
we used the UCI Health data set available from https://www.kaggle.com/ronitf/heart-disease-uci

Targets
  - 0 = no presence of heart disease
  - 1 = presence of heart disease
 
Target value counts
  - 1 = 165
  - 0 = 138

Good mix of binary, continous and categorical features:


## Methods

For this Github repository I will share the Principal Component Analysis (PCA) and SMOTE methods used to generate synthetic data from the dataset. PCA meausures correlation between fearures and projects this on a 2D plane, this dimension reduction will be used to sample the synthetic data. 
SMOTE uses measures distance between points on a 2D plane, and samples a synthetic point from between the 2 points.

Categotirical data were re-labelled with one hot encoding.
