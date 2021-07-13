"""
Using the metacalibration function, this script loops over a variety of
different parameters to measure the response of the shear response matrix.

Parameters to vary include:
Galaxy size
Galaxy shape
PSF size
PSF shape
Calibration shear magnitude

Shear estimation method (?)
"""
import galsim
import numpy as np
import metacal

def vary_parameters():
    """
Problem is that in creating objects to loop through, you can't replace
more than one parameters within the object at a time

Could have multiple loops. One for the observed galaxy creation, one for deconvolution psf creation,
one for reconvolution psf creation
    """
    results = []

    # Creating lists of different parameters to loop through
    deconv_psf_size_variation = [galsim.Gaussian(flux=1., sigma=sig) for sig in np.arange(0.5, 2.5, 0.1)]

    # Creating initial observed galaxy to test
    gal_flux = 1.e5
    gal_sigma = 2.
    dg1 = 0.01
    dg2 = 0.01
    gal = galsim.Gaussian(flux=gal_flux, sigma=gal_sigma)

    psf1 = galsim.Gaussian(flux=1., sigma=gal_sigma)

    observed_galaxy = galsim.Convolve(gal, psf1)

    for deconv_psf in deconv_psf_size_variation:
        shear_response = metacal.metacalibration(observed_galaxy, deconv_psf, psf1, dg1, dg2)
        results.append((observed_galaxy, psf1, deconv_psf, psf1, dg1, dg2, shear_response))




    # print(results)

def main():
    vary_parameters()


if __name__ == '__main__':
    main()