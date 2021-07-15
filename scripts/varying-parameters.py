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
from multiprocessing import Pool
import pandas as pd


def squared_distance_metric(r):
    """
    Takes in a matrix r and returns its scalar "distance"
    from 2 * identity
    """
    return np.sum(np.square(r - 2*np.eye(2)))


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

    map_list = []

    # for deconv_psf in deconv_Gaussian_size_variation + deconv_Moffat_size_variation:
    #     for reconv_psf in reconv_Gaussian_size_variation + reconv_Moffat_size_variation:
    #
    #         # print(deconv_psf, reconv_psf)
    #         # shear_response = metacal.metacalibration(observed_galaxy, deconv_psf, reconv_psf, dg1, dg2)
    #         # results.append((observed_galaxy, psf1, deconv_psf, reconv_psf, dg1, dg2, shear_response))
    #
    #         map_list.append((observed_galaxy, deconv_psf, reconv_psf, dg1, dg2))

    for deconv_psf in deconv_Gaussian_size_variation + deconv_Moffat_size_variation:
        for reconv_psf in reconv_Gaussian_size_variation + reconv_Moffat_size_variation:
            for delta_g in dg:
                map_list.append((observed_galaxy, deconv_psf, reconv_psf, delta_g, delta_g))
    #             print(deconv_psf, reconv_psf_type)
    #             shear_response = metacal.metacalibration(observed_galaxy, deconv_psf, reconv_psf_type, delta_g, delta_g)
    #             results.append((observed_galaxy, psf1, deconv_psf, reconv_psf_type, delta_g, delta_g, shear_response))

    with Pool(4) as p:
        results = p.starmap(metacal.metacalibration, map_list)

    with open(storage_file, 'wb') as f:
        pickle.dump(results, f)

    print("\n" * 4)
    print(f"Result stored to {storage_file}")
    print("\n" * 4)

def identify_profile(obj):
    if isinstance(obj, galsim.gaussian.Gaussian):
        return 'Gaussian'
    if isinstance(obj, galsim.moffat.Moffat):
        return 'Moffat'


def filter_results(results):
    R_moffat = []
    R_gaussian = []
    for result in results:
        if isinstance(result[1], galsim.moffat.Moffat):
            R_moffat.append(result[-1])
        if isinstance(result[1], galsim.gaussian.Gaussian):
            R_gaussian.append(result[-1])

    # print(np.mean(R_moffat, axis=0))
    # print(np.mean(R_gaussian, axis=0))

# Trying to load the results table into a Pandas DataFrame
    results_df = pd.DataFrame(results, columns=['observed_galaxy', 'deconv_psf', 'reconv_psf', 'dg1', 'dg2', 'R'])

    # entry = results_df['deconv_psf'][0]
    # print(isinstance(entry, galsim.gaussian.Gaussian))

    results_df['profile_type'] = [identify_profile(obj) for obj in results_df['deconv_psf']]
    print(results_df)

    # clean this up later

    gaussian_rows = results_df[[isinstance(res, galsim.gaussian.Gaussian) for res in results_df['deconv_psf']]]
    moffat_rows = results_df[[isinstance(res, galsim.moffat.Moffat) for res in results_df['deconv_psf']]]

    gaussian_Rs = gaussian_rows['R']
    gaussian_R_mean = np.sum(gaussian_Rs) / len(gaussian_Rs)
    gaussian_R_distance = squared_distance_metric(gaussian_R_mean)
    print(gaussian_R_distance)

    moffat_Rs = moffat_rows['R']
    moffat_R_mean = np.sum(moffat_Rs) / len(moffat_Rs)
    moffat_R_distance = squared_distance_metric(moffat_R_mean)
    print(moffat_R_distance)


def main():

    args = sys.argv[1:]

    if args[0] == '-generate':
        vary_parameters('Results.pickle')

    with open('Results.pickle', 'rb') as f:
        stored_results = pickle.load(f)

    if args[0] == '-display':

        print(stored_results)
        print("\n" * 2)
        print(f"Displayed metacalibration results for {len(stored_results)} different cases")
        print("\n" * 2)

    if args[0] == '-filter':
        filter_results(stored_results)

    return 0


if __name__ == '__main__':
    main()