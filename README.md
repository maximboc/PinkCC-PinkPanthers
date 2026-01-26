## PINKCC Inference

Initial Setup 


```bash
pip install -e .
```

File architecture example
```
/path/to/DATASET/
            ├── image_001.nii.gz
            ├── image_002.nii.gz
            ├── ...
            ovseg_predictions/

```

Run this command for inference
```bash
ovseg_inference /path/to/DATASET/ --models pinkcc
```

