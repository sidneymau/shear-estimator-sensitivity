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
from IPython.display import display

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
    dg1 = 0.00
    dg2 = 0.00

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
        dataframe[object_column_name + '_type'] = profile_type

        if profile_type == 'Gaussian':
            gauss_flux.append(profile_tuple[1])
            gauss_sigma.append(profile_tuple[2])

            for list in [moffat_flux, moffat_beta, moffat_hlr]:
                list.append(np.nan)

        if profile_type == 'Moffat':
            moffat_flux.append(profile_tuple[1])
            moffat_beta.append(profile_tuple[2])
            moffat_hlr.append(profile_tuple[3])

            for list in [gauss_flux, gauss_sigma]:
                list.append(np.nan)

    dataframe[object_column_name + '_flux'] = gauss_flux
    dataframe[object_column_name + '_sigma'] = gauss_sigma
    dataframe[object_column_name + '_flux'] = moffat_flux
    dataframe[object_column_name + '_beta'] = moffat_beta
    dataframe[object_column_name + '_half_light_radius'] = moffat_hlr


def apply_metric(dataframe, metric):
    """
    Takes in the function metric (that acts on a 2x2 np array)
    and adds a column to the dataframe passed in with that metric applied to
    each row
    """
    dataframe[metric.__name__] = list(map(metric, dataframe['R']))

# TODO -----
def select_data():
    """
    TODO Implement a function that allows for easier selection of certain slices of data

    should return a data frame (?)
    """
    pass


def generate_df(results):
    """
    Takes in the results array and returns a pandas dataframe with columns
    for each parameter
    """
    # Loading the results table into a Pandas DataFrame
    results_df = pd.DataFrame(results, columns=['observed_galaxy', 'deconv_psf', 'reconv_psf', 'dg1', 'dg2', 'R'])

    # creating columns for psf parameters
    create_psf_parameter_columns(results_df, 'deconv_psf')
    create_psf_parameter_columns(results_df, 'reconv_psf')

    # creating columns of the metrics for shear response matrix "closeness"
    apply_metric(results_df, frobenius_norm)
    apply_metric(results_df, sum_abs_differences)

    return results_df

    # TODO think about how to display different observed galaxies in the dataframe


    # # clean this up later

    # TODO update the below code to work with the new table

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

    # # splitting Gaussian vs Moffat deconvolution psfs
    # gaussian_deconv = results_df[results_df['deconv_profile_type'] == 'Gaussian']
    # moffat_deconv = results_df[results_df['deconv_profile_type'] == 'Moffat']
    #
    # # splitting Gaussian vs Moffat reconvolution psfs
    # gaussian_reconv = results_df[results_df['reconv_profile_type'] == 'Gaussian']
    # moffat_reconv = results_df[results_df['reconv_profile_type'] == 'Moffat']
    #
    #
    # print(gaussian_deconv)
    # print(moffat_deconv)
    #
    # print(gaussian_reconv)
    # print(moffat_reconv)


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
        pd_table = generate_df(stored_results)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(pd_table)

    return 0


if __name__ == '__main__':
    main()