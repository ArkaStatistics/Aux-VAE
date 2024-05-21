
"""
Simple function that, given input parameters (flux, radius, psf fwhm, shear profile).
Returns an image of a galaxy without noise using GalSim.
"""

import os
import shutil
import numpy as np
import galsim
import h5py
import matplotlib.pyplot as plt
from astropy.io import fits

def GenGalIm(params):
    """
    Generate a galaxy image given input parameters.

    Parameters:
    params : array-like
        - flux : Flux of the galaxy in counts.
        - radius : Radius of the galaxy in arcsec.
        - g1 : Reduced shear component.
        - g2 : Reduced shear component.
        - psf_fwhm : FWHM of the Gaussian PSF used for convolution.

    Returns:
    galsim.Image
        33x33 pixels image of a galaxy given input parameters.
    """
    galflux, galradius, g1, g2, psffwhm = params
    nx, ny = 33, 33  # Image dimensions
    pixel_scale = 0.2  # arcsec/pixel
    sky_level = 0  # counts / arcsec^2

    # Define the galaxy profile
    gal = galsim.Exponential(flux=galflux, scale_radius=galradius)
    gal = gal.shear(g1=g1, g2=g2)

    # Define the PSF profile
    psf = galsim.Gaussian(fwhm=psffwhm)

    # Convolution
    big_fft_params = galsim.GSParams(maximum_fft_size=12300)
    final = galsim.Convolve([gal, psf], gsparams=big_fft_params)

    # Draw the image
    image = final.drawImage(nx=nx, ny=ny, scale=pixel_scale)

    # Add Poisson noise
    rng = galsim.BaseDeviate(150000)
    sky_level_pixel = sky_level * pixel_scale ** 2
    noise = galsim.PoissonNoise(rng, sky_level=sky_level_pixel)
    image.addNoise(noise)

    return image

def GenSetGal(file_name):
    """
    Generate a set of galaxy images from a file of parameters.

    Parameters:
    file_name : str
        File containing the input parameters.

    Returns:
    np.ndarray
        Array of galaxy images.
    """
    print('Loading parameter file from: ' + file_name)
    params = np.loadtxt(file_name)
    nx, ny = 33, 33  # Image dimensions
    setgal = np.zeros((params.shape[0], nx, ny))

    for i in range(params.shape[0]):
        setgal[i] = GenGalIm(params[i]).array

    return setgal

def SaveGal(images, fname, dname, dir_out):
    """
    Save galaxy images to an HDF5 and FITS file.

    Parameters:
    images : np.ndarray
        Galaxy images to save.
    fname : str
        File name.
    dname : str
        Dataset name.
    dir_out : str
        Output directory.
    """
    if not os.path.isdir(dir_out):
        os.mkdir(dir_out)

    file_name = os.path.join(dir_out, fname + '.hdf5')
    with h5py.File(file_name, 'w') as f:
        f.create_dataset(dname, data=images)

    hdu = fits.PrimaryHDU(images)
    hdu.writeto(os.path.join(dir_out, fname + '.fits'), overwrite=True)
    print('Data saved at: ' + dir_out)

def PlotGal():
    """
    Generate and plot a galaxy image for testing purposes.
    """
    galflux = 1e4
    galradius = 0.5
    g1, g2 = 0.1, 0.4
    psffwhm = 0.1

    image = GenGalIm((galflux, galradius, g1, g2, psffwhm))

    plt.imshow(image.array)
    plt.show()

dir_in = './data/'
file_name = 'lhc_16384_5_training.txt'
dir_out = './data/'

#shutil.copyfile(os.path.join(dir_in, file_name), os.path.join(dir_out, file_name))

images = GenSetGal(os.path.join(dir_in, file_name))
SaveGal(images, 'train_16384_5_training', 'galaxies', dir_out)

# Uncomment the following line to plot a test galaxy
# PlotGal()
