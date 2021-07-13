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
import pickle
import sys

def vary_parameters(storage_file):
    """
    Problem is that in creating objects to loop through, you can't replace
    more than one parameters within the object at a time

    Could have multiple loops. One for the observed galaxy creation, one for deconvolution psf creation,
    one for reconvolution psf creation

    """
    results = []

    # Creating lists of different parameters to loop through

    psf_beta = 5.

    # Deconvolution PSF type and size variations
    deconv_Gaussian_size_variation = [galsim.Gaussian(flux=1., sigma=sig) for sig in np.arange(0.5, 2.0, 0.1)]
    deconv_Moffat_size_variation = [galsim.Moffat(beta=psf_beta, flux=1., half_light_radius=r0) for r0 in np.arange(0.8, 2.0, 0.2)]

    # Reconvolution PSF type and size variations
    reconv_Gaussian_size_variation = [galsim.Gaussian(flux=1., sigma=sig) for sig in np.arange(1.0, 3.0, 0.1)]
    reconv_Moffat_size_variation = [galsim.Moffat(beta=psf_beta, flux=1., half_light_radius=r0) for r0 in np.arange(0.8, 2.0, 0.2)]

    # different sized calibration shears
    dg = np.arange(0.01, 0.11, 0.01)

    # Creating initial observed galaxy to test
    gal_flux = 1.e5
    gal_sigma = 2.
    dg1 = 0.01
    dg2 = 0.01
    gal = galsim.Gaussian(flux=gal_flux, sigma=gal_sigma)

    psf1 = galsim.Gaussian(flux=1., sigma=gal_sigma)

    observed_galaxy = galsim.Convolve(gal, psf1)

    for deconv_psf in deconv_Gaussian_size_variation + deconv_Moffat_size_variation:
        for reconv_psf in reconv_Gaussian_size_variation + reconv_Moffat_size_variation:
            print(deconv_psf, reconv_psf)
            shear_response = metacal.metacalibration(observed_galaxy, deconv_psf, reconv_psf, dg1, dg2)
            results.append((observed_galaxy, psf1, deconv_psf, reconv_psf, dg1, dg2, shear_response))

    # for deconv_psf in deconv_Gaussian_size_variation + deconv_Moffat_size_variation:
    #     for reconv_psf_type in reconv_Gaussian_size_variation + reconv_Moffat_size_variation:
    #         for delta_g in dg:
    #             print(deconv_psf, reconv_psf_type)
    #             shear_response = metacal.metacalibration(observed_galaxy, deconv_psf, reconv_psf_type, delta_g, delta_g)
    #             results.append((observed_galaxy, psf1, deconv_psf, reconv_psf_type, delta_g, delta_g, shear_response))


    # print(results)

    with open(storage_file, 'wb') as f:
        pickle.dump(results, f)

    print("\n" * 4)
    print(f"Result stored to {storage_file}")
    print("\n" * 4)


def main():

    args = sys.argv[1:]

    if args[0] == '-generate':
        vary_parameters('Results.pickle')

    if args[0] == '-display':
        with open('Results.pickle', 'rb') as f:
            stored_results = pickle.load(f)
        print(stored_results)
        print("\n" * 2)
        print(f"Displayed metacalibration results for {len(stored_results)} different cases")
        print("\n" * 2)

    return 0



if __name__ == '__main__':
    main()