# The synthetic galaxy image dataset

This repository contains scripts for creating and visualizing synthetic galaxy light profiles. The simple light profiles depend on generating factors that are sampled from a latin-Hypercube design. 

## Structure

- `generate_factors.py`: Generates a Latin Hypercube design of parameters and saves them to a file in the `data` directory.
- `generate_galaxies.py`: Takes the parameter file from the `data` directory and creates galaxy light profiles, saving the images in an HDF5 and FITS files in the `data` directory.
- `catalog_read_plot_utils.py`: Contains HDF5 reading and plotting routines.
- `read_plot_galaxies.ipynb`: Jupyter notebook demonstrating the usage of the scripts.
- `data/`: Directory to store parameter files and galaxy images.

## Requirements

To create the Conda environment for generating this dataset, use the following command:
```bash
conda env create -f environment.yml
```

## Usage

1. Run `generate_factors.py` to generate the parameter file.
   ```bash
   python generate_factors.py
   ```
2. Run `generate_galaxies.py` to create and save the galaxy light profiles.
   ```bash
   python generate_galaxies.py
   ```
3. Open and run the `read_plot_galaxies.ipynb` notebook for a demonstration.

