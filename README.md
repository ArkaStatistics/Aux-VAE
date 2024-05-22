# Aux-VAE: Auxiliary Variational Autoencoder

Aux-VAE is an enhanced implementation of the variational autoencoder (VAE) that incorporates auxiliary information into the latent space to improve disentanglement and reconstruction accuracy. This project specifically targets complex datasets such as galaxy simulations and is also tested on standard datasets like Cars3D and DSprites to demonstrate its versatility and robustness.

## Features

- **Disentanglement of Latent Space**: Leverages auxiliary information for better separation of latent factors.
- **Enhanced Reconstruction**: Maintains high fidelity in data reconstruction, even with complex datasets.
- **Flexibility**: Tested on both scientific (galaxy simulation) and standard machine learning datasets (Cars3D, DSprites).

## Installation

To install Aux-VAE, clone this repository and install the required dependencies.

The current setting demonstrates the implementation of Aux-VAE in the galaxy simulation data. To generate the dataset, use the codes provided in the "synthetic_galaxy_dataset" folder.

The `models_all.py` file in the `Aux-VAE` folder contains various configurations of beta-VAE and Aux-VAE (including both MLP and CNN versions). Choose the appropriate model based on the specific settings required for your experiments.
