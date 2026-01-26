## PINKCC Inference

Initial Setup 

```bash
export OV_DATA_BASE="path/to/data"
```
path/to/data : Path to the directory in which we find raw_data/

```bash
pip install .
```

File architecture example
```
/path/to/data/
    └── raw_data/
        └── DATASET/
            ├── image_001.nii.gz
            ├── image_002.nii.gz
            ├── ...
            ovseg_predictions/
    └── clara_models/
```

Run this command for inference
```bash
ovseg_inference /path/to/DATASET/ --models pinkcc
```