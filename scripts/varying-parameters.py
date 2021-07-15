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

# TODO Think of other metrics that could be used to calculate shear response matrix quality
def squared_distance_metric(r):
    """
    Takes in a matrix r and returns its scalar "distance"
    from 2 * identity
    """
    return np.sum(np.square(r - 2*np.eye(2)))

def generate_combinations():
    """
    Generates a list of different combinations of
    (observed galaxy, deconv_psf, reconv_psf, delta_g, delta_g)

    to run metacalibration over. Feeds into vary_parameters()
    """

    # TODO will eventually need to change this function to vary the observed galaxy used as well

    # Loop over one observed galaxy only (to test)
    gal_flux = 1.e5
    gal_sigma = 2.
    gal = galsim.Gaussian(flux=gal_flux, sigma=gal_sigma)

    # initial shear TODO change this to use an "intrinsic ellipticity" parameter on generation instead of manually shearing?
    dg1 = 0.01
    dg2 = 0.01

    psf1 = galsim.Gaussian(flux=1., sigma=gal_sigma)

    observed_galaxy = metacal.generate_observed_galaxy(gal, psf1, dg1, dg2)

    # Creating lists of different parameters to loop through
    psf_beta = 5.

    # Deconvolution PSF type and size variations
    deconv_Gaussian_size_variation = [galsim.Gaussian(flux=1., sigma=sig) for sig in np.arange(0.5, 2.0, 0.1)]
    deconv_Moffat_size_variation = [galsim.Moffat(beta=psf_beta, flux=1., half_light_radius=r0) for r0 in
                                    np.arange(0.8, 2.0, 0.2)]

    # Reconvolution PSF type and size variations
    reconv_Gaussian_size_variation = [galsim.Gaussian(flux=1., sigma=sig) for sig in np.arange(1.0, 3.0, 0.1)]
    reconv_Moffat_size_variation = [galsim.Moffat(beta=psf_beta, flux=1., half_light_radius=r0) for r0 in
                                    np.arange(0.8, 2.0, 0.2)]

    # different sized calibration shears
    dg = np.arange(0.01, 0.11, 0.01)

    # Creating long master list of all combinations to loop through
    combination_list = []
    for deconv_psf in deconv_Gaussian_size_variation + deconv_Moffat_size_variation:
        for reconv_psf in reconv_Gaussian_size_variation + reconv_Moffat_size_variation:
            for delta_g in dg:
                combination_list.append((observed_galaxy, deconv_psf, reconv_psf, delta_g, delta_g))

    return combination_list


def vary_parameters(combo_list, storage_file):
    """
    """

    # Using multiprocessing to generate shear response matrices for all combinations
    with Pool(4) as p:
        results = p.starmap(metacal.metacalibration, combo_list)

    # Storing metacalibration results to disk
    with open(storage_file, 'wb') as f:
        pickle.dump(results, f)

    print("\n" * 4)
    print(f"Result stored to {storage_file}")
    print("\n" * 4)


def identify_psf_profile(obj):
    """
    Takes in a galsim PSF object and returns a tuple
    of its type and relevant parameters
    """

    # TODO incorporate more types of PSF profiles

    if isinstance(obj, galsim.gaussian.Gaussian):
        return ('Gaussian', obj.flux, obj.sigma)
    if isinstance(obj, galsim.moffat.Moffat):
        return ('Moffat', obj.flux, obj.beta, obj.half_light_radius)


# TODO write a function that takes in a set of rows (as a dataframe) and computes the metric on all the matrices in those rows

def filter_results(results):

    # Loading the results table into a Pandas DataFrame
    results_df = pd.DataFrame(results, columns=['observed_galaxy', 'deconv_psf', 'reconv_psf', 'dg1', 'dg2', 'R'])

    # creating a new column 'deconv_profile_type' to mark the
    results_df['deconv_profile_type'] = [identify_psf_profile(obj)[0] for obj in results_df['deconv_psf']]

    # creating a new column 'reconv_profile'
    results_df['reconv_profile_type'] = [identify_psf_profile(obj)[0] for obj in results_df['reconv_psf']]

    # TODO think about how to display different observed galaxies in the dataframe

    print(results_df)

    # # clean this up later
    # gaussian_rows = results_df[[isinstance(res, galsim.gaussian.Gaussian) for res in results_df['deconv_psf']]]
    # moffat_rows = results_df[[isinstance(res, galsim.moffat.Moffat) for res in results_df['deconv_psf']]]
    #
    # gaussian_Rs = gaussian_rows['R']
    # gaussian_R_mean = np.sum(gaussian_Rs) / len(gaussian_Rs)
    # gaussian_R_distance = squared_distance_metric(gaussian_R_mean)
    # print(gaussian_R_distance)
    #
    # moffat_Rs = moffat_rows['R']
    # moffat_R_mean = np.sum(moffat_Rs) / len(moffat_Rs)
    # moffat_R_distance = squared_distance_metric(moffat_R_mean)
    # print(moffat_R_distance)

    # splitting Gaussian vs Moffat deconvolution psfs
    gaussian_deconv = results_df[results_df['deconv_profile_type'] == 'Gaussian']
    moffat_deconv = results_df[results_df['deconv_profile_type'] == 'Moffat']

    # splitting Gaussian vs Moffat reconvolution psfs
    gaussian_reconv = results_df[results_df['reconv_profile_type'] == 'Gaussian']
    moffat_reconv = results_df[results_df['reconv_profile_type'] == 'Moffat']


    print(gaussian_deconv)
    print(moffat_deconv)

    print(gaussian_reconv)
    print(moffat_reconv)


def main():

    args = sys.argv[1:]

    if args[0] == '-generate':
        combinations = generate_combinations()
        vary_parameters(combinations, 'Results.pickle')

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