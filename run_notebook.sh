#!/bin/bash


# Deactivate conda stuff (if present)
conda deactivate 

# Activate venv
source env/bin/activate

# Run notebook
jupyter-notebook explorer_local.ipynb
