## Dataset

Here should be your provided dataset DatasetChallenge/

It contains the 250 CT images + segmentations from MSKCC and TCGA open repositories.
Data has been filtered and some segmentations have been corrected.
CT scans and corresponding segmentations share the same file name. 

The anonymized data is in NIFTI format (compressed nii.gz extension), 
it can be loaded with the classic nibabel package in Python. 

The segmentation labels are: 
- 0: background
- 1: primary tumor (ovarian carcinoma)
- 2: metastasis (peritoneal carcinomatosis and others)

DO NOT PUSH IT TO THE REPO ! 
(you should not be able to do so but just in case)
