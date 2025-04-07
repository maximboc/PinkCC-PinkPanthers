## PINKCC Challenge - PinkPanthers Team Repository

# Workflow
- A `main` branch
- A `dev` branch
- A branch for each model implemented

# Project Structure 
- `models/` : contains each model implemented (each subdirectory is a studied model)
- `utils/` : functions that everyone can use
- `data/` : directory containing the dataset 

# Rules
- Make branches on top of the `dev` branch
- Merge on dev when the model is "finished"
- Never EVER merge on main except if it was approved by tests made on GPU
