# DPGLO_SRNet
Dual-Path Global-Local Feature Fusion with Shadow-Aware Guidance for Image Shadow Removal
- This code is directly related to the manuscript submitted to The (Journal): `Dual-Path Global-Local Feature Fusion with Shadow-Aware Guidance for Image Shadow Removal.' 
If you use this code or models in your research, please cite the corresponding manuscript.
# Requirement
- Python 3.11
- Pytorch 2.0
- CUDA 11.7
- MATLAB R2023b
# Datasets
- AISTD (ISTD+) [link](https://github.com/cvlab-stonybrook/SID)
- SRD [Training](https://drive.google.com/file/d/1W8vBRJYDG9imMgr9I2XaA13tlFIEHOjS/view) [Testing](https://drive.google.com/file/d/1GTi4BmQ0SJ7diDMmf-b7x2VismmXtfTo/view) [Mask](https://uofmacau-my.sharepoint.com/:u:/g/personal/yb87432_um_edu_mo/EZ8CiIhNADlAkA4Fhim_QzgBfDeI7qdUrt6wv2EVxZSc2w?e=wSjVQT) (detected by [DHAN](https://github.com/vinthony/ghost-free-shadow-removal))
# Pre-trained models
The corresponding pre-trained models:
- AISTD (ISTD+) [checkpoint](https://drive.google.com/file/d/1dmZfU-iX8cy4V26NVPhZKtRIOe3n5Sv8/view?usp=drive_link)
- SRD [checkpoint](https://drive.google.com/file/d/10oyVLdH3-WnZ-rAcPSHMHPFj2FttDxUN/view?usp=drive_link)
# Test the model
You can directly test the performance of the pre-trained model as follows:
Modify the paths to dataset and pre-trained model. You need to modify the following path in the `test.py` and run
- python test.py --load [checkpoint numbers, e.g 860]
# Train
1. Download datasets and set the following structure

    ```
    -- AISTD_Dataset
       |-- train
       |   |-- train_A  # shadow image
       |   |-- train_B  # shadow mask 
       |   |-- train_C  # shadow-free GT
       |
       |-- test
           |-- test_A  # shadow image
           |-- test_B  # shadow mask 
           |-- test_C  # shadow-free GT

    -- SRD_Dataset
       |-- train
       |   |-- train_A  # shadow image
       |   |-- train_B  # shadow mask 
       |   |-- train_C  # shadow-free GT
       |
       |-- test
           |-- test_A  # shadow image
           |-- test_B  # shadow mask 
           |-- test_C  # shadow-free GT
# Evaluation
The results reported in the paper are calculated by the `matlab` script used in [previouse method](https://github.com/hhqweasd/G2R-ShadowNet/blob/main/evaluate.m)
# Visual results
The Visual results on dataset  AISTD (ISTD+), SRD are:
- AISTD (ISTD+) [Results](https://drive.google.com/file/d/1eGR9_z8JOu_uC8GaSG_NMKQWTrfekNzG/view?usp=drive_link)
- SRD [Results](https://drive.google.com/file/d/1ApP25Vd2obCPiyxUovAHUaTytktMKbdz/view?usp=drive_link)

# Contact
If you have any questions, please contact idreeskhan045@gmail.com/ huangying@cqupt.edu.cn
