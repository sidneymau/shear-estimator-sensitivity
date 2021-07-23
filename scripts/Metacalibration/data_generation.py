"""
Using the metacalibration function, this script loops over a variety of
different parameters to measure the response of the shear response matrix.

TODO Update docstrings and parameters to PEP 8 standards
"""
import galsim
import numpy as np
import metacal
import pickle
import sys
from multiprocessing import Pool
import pandas as pd
import matplotlib.pyplot as plt
import os.path

# TODO Think of other metrics that could be used to calculate shear response matrix quality
def frobenius_norm(r):
    """
    Takes in a matrix r and returns its frobenius distance
    from 2 * identity
    """
    return np.sqrt(np.sum(np.square(r - 2*np.eye(2))))


def sum_abs_differences(r):
    """
    Takes in a matrix r and returns the sum of the element-wise distances
    from 2 * identity
    """
    return np.sum(np.absolute(r - 2*np.eye(2)))


 def sanity_check_1():
    # TODO change units to be in fwhm

    gal_flux = 1.e5
    gal_sigma = 2.
    gal = galsim.Gaussian(flux=gal_flux, sigma=gal_sigma)

    # initial shear
    dg1 = 0.00
    dg2 = 0.00

    # Original PSF size / galaxy size variations

    true_psf_vary_sigma = [galsim.Gaussian(flux=1., sigma=sig) for sig in 1 / 2.355 * np.arange(0.5, 1.3, 0.1)]

    observed_galaxy_variation = [metacal.generate_observed_galaxy(gal, psf, dg1, dg2) for psf in true_psf_vary_sigma]

    # Deconvolution PSF type and size variations
    deconv_Gaussian_size_variation = [galsim.Gaussian(flux=1., sigma=sig) for sig in
                                      1 / 2.355 * np.arange(0.5, 1.3, 0.1)]

    # Reconvolution PSF type and size variations  TODO Look up by how much the reconvolution PSF is dilated
    dilation_factor = 1.2
    reconv_Gaussian_size_variation = [galsim.Gaussian(flux=1., sigma=sig) for sig in
                                      1 / 2.355 * dilation_factor * np.arange(0.5, 1.3, 0.1)]

    dg = [0.01]  # same as Sheldon and Huff value

    # Creating long master list of all combinations to loop through
    combination_list = []
    for i in range(len(observed_galaxy_variation)):
        for delta_g in dg:
            combination_list.append((observed_galaxy_variation[i], true_psf_vary_sigma[i], deconv_Gaussian_size_variation[i], reconv_Gaussian_size_variation[i], delta_g, delta_g))

    return combination_list


def sanity_check_2():

    gal_flux = 1.e5
    dg = [0.01]
    dilation_factor = 1.2

    gal_psf_ratios = np.arange(0.5, 2.1, 0.1)
    true_psf_sigmas = 1/2.355 * np.arange(0.5, 1.3, 0.1)  # 0.5 - 1.3 arcseconds fwhm converted to sigma
    reconv_psf_sigmas = dilation_factor * true_psf_sigmas

    combinations = []
    for ratio in gal_psf_ratios:        # gal_sigma / psf_sigma should be equal to ratio
        for i in range(len(true_psf_sigmas)):
            for delta_g in dg:
                gal = galsim.Gaussian(flux=gal_flux, sigma=true_psf_sigmas[i] * ratio)
                true_psf = galsim.Gaussian(flux=gal_flux, sigma=true_psf_sigmas[i])
                deconv_psf = true_psf
                reconv_psf = galsim.Gaussian(flux=gal_flux, sigma=reconv_psf_sigmas[i])
                observed_galaxy = galsim.Convolve(gal, true_psf)

                combinations.append((observed_galaxy, true_psf, deconv_psf, reconv_psf, delta_g, delta_g))

    return combinations


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
    gal = galsim.Gaussian(flux=gal_flux, sigma=gal_sigma,)

    # initial shear TODO change this to use an "intrinsic ellipticity" parameter on generation instead of manually shearing?
    dg1 = 0.00
    dg2 = 0.00

    psf1 = galsim.Gaussian(flux=1., sigma=gal_sigma)

    observed_galaxy = metacal.generate_observed_galaxy(gal, psf1, dg1, dg2)

    # Creating lists of different parameters to loop through
    psf_beta = 5.

    # Deconvolution PSF type and size variations
    deconv_Gaussian_size_variation = [galsim.Gaussian(flux=1., sigma=sig) for sig in 1/2.355 * np.arange(0.5, 1.3, 0.1)]
    deconv_Moffat_size_variation = [galsim.Moffat(beta=psf_beta, flux=1., half_light_radius=r0) for r0 in
                                    np.arange(0.8, 2.0, 0.2)]

    # Reconvolution PSF type and size variations
    reconv_Gaussian_size_variation = [galsim.Gaussian(flux=1., sigma=sig) for sig in 1/2.355 * np.arange(0.5, 1.3, 0.1)]
    reconv_Moffat_size_variation = [galsim.Moffat(beta=psf_beta, flux=1., half_light_radius=r0) for r0 in
                                    np.arange(0.8, 2.0, 0.2)]

    # different sized calibration shears
    # dg = np.arange(0.01, 0.11, 0.01)
    dg = [0.01]  # same as Sheldon and Huff Value

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


def create_psf_parameter_columns(dataframe, object_column_name):
    """
    """
    dataframe[object_column_name + '_type'] = [identify_psf_profile(obj)[0] for obj in dataframe[object_column_name]]

    gauss_flux = []
    gauss_sigma = []

    moffat_flux = []
    moffat_beta = []
    moffat_hlr = []


    for obj in dataframe[object_column_name]:
        profile_tuple = identify_psf_profile(obj)
        profile_type = profile_tuple[0]

        if profile_type == 'Gaussian':
            gauss_flux.append(profile_tuple[1])
            gauss_sigma.append(profile_tuple[2])

            for lst in [moffat_flux, moffat_beta, moffat_hlr]:
                lst.append(np.nan)

        if profile_type == 'Moffat':
            moffat_flux.append(profile_tuple[1])
            moffat_beta.append(profile_tuple[2])
            moffat_hlr.append(profile_tuple[3])

            for lst in [gauss_flux, gauss_sigma]:
                lst.append(np.nan)


    dataframe[object_column_name + '_gaussian_flux'] = gauss_flux
    dataframe[object_column_name + '_sigma'] = gauss_sigma
    dataframe[object_column_name + '_moffat_flux'] = moffat_flux
    dataframe[object_column_name + '_beta'] = moffat_beta
    dataframe[object_column_name + '_half_light_radius'] = moffat_hlr


def apply_metric(dataframe, metric):
    """
    Takes in the function metric (that acts on a 2x2 np array)
    and adds a column to the dataframe passed in with that metric applied to
    each row
    """
    dataframe[metric.__name__] = list(map(metric, dataframe['R']))


def element_columns(dataframe):
    """
    Adds as columns the 4 individual elements of the shear response matrix
    """
    for i in range(0, 2):
        for j in range(0, 2):
            dataframe['R_' + str(i + 1) + str(j + 1)] = list(map(lambda r: r[i][j], dataframe['R']))


def true_psf_columns(dataframe):

    dataframe['true_psf_sigma'] = list(map(lambda obj: obj.sigma, dataframe['true_psf']))
    return dataframe


# TODO add this column to the dataframe, trace the parameter through the code
def gal_psf_ratio(dataframe):
    pass

def generate_df(results):
    """
    Takes in the results array and returns a pandas dataframe with columns
    for each parameter
    """
    # Loading the results table into a Pandas DataFrame
    results_df = pd.DataFrame(results, columns=['observed_galaxy', 'true_psf', 'deconv_psf', 'reconv_psf', 'dg1', 'dg2', 'R'])

    # creating columns for psf parameters
    create_psf_parameter_columns(results_df, 'deconv_psf')
    create_psf_parameter_columns(results_df, 'reconv_psf')

    # creating columns of the metrics for shear response matrix "closeness"
    apply_metric(results_df, frobenius_norm)
    apply_metric(results_df, sum_abs_differences)

    # creating columns for the individual shear response matrix elements
    element_columns(results_df)

    # creating a column for the sigma of the true psf
    true_psf_columns(results_df)

    return results_df

    # TODO think about how to display different observed galaxies in the dataframe


def main():

    args = sys.argv[1:]

    if args[0] == '-generate':

        # combinations = generate_combinations()
        # combinations = sanity_check1()
        combinations = sanity_check_2()

        if not os.path.exists('Results.pickle'):
            vary_parameters(combinations, 'Results.pickle')

        else:
            version = 1
            while os.path.exists('Results' + '(' + str(version) + ').pickle'):
                version += 1

        vary_parameters(combinations, 'Results' + '(' + str(version) + ').pickle')

        return 0




    with open('Results2.pickle', 'rb') as f:
        stored_results = pickle.load(f)


    if args[0] == '-filter':
        pd_table = generate_df(stored_results)
        pd_table.to_csv('table.csv')
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print(pd_table)

    return 0


if __name__ == '__main__':
    main()