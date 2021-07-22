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
    deconv_Gaussian_size_variation = [galsim.Gaussian(flux=1., sigma=sig) for sig in 1/2.355 * np.arange(0.5, 1.3, 0.1)]
    deconv_Moffat_size_variation = [galsim.Moffat(beta=psf_beta, flux=1., half_light_radius=r0) for r0 in
                                    np.arange(0.8, 2.0, 0.2)]

    # Reconvolution PSF type and size variations
    reconv_Gaussian_size_variation = [galsim.Gaussian(flux=1., sigma=sig) for sig in 1/2.355 * np.arange(0.5, 1.3, 0.1)]
    reconv_Moffat_size_variation = [galsim.Moffat(beta=psf_beta, flux=1., half_light_radius=r0) for r0 in
                                    np.arange(0.8, 2.0, 0.2)]

    # different sized calibration shears
    # dg = np.arange(0.01, 0.11, 0.01)
    dg = [0.01] # same as Sheldon and Huff Value

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


def plot_data(dataframe):
    """
    TODO Implement a function that allows for easier selection of certain slices of data

    should return a data frame (?)
    """

    # # test plotting R closeness to 2I vs. calibration shear magnitude
    #
    # grouped_by_dg = dataframe.groupby('dg1').mean()
    # grouped_by_dg = grouped_by_dg[['frobenius_norm', 'sum_abs_differences']] # don't need to include dg1 because that's the table
    # print(grouped_by_dg.index.name) # accessing name of variable indexed by
    # print(grouped_by_dg.index.values) # to access the variable grouped by, need to do df.index.values
    #
    # fig, axs = plt.subplots(1, 1)
    #
    # frob = axs.scatter(grouped_by_dg.index.values, grouped_by_dg['frobenius_norm'])
    # sumdif = axs.scatter(grouped_by_dg.index.values, grouped_by_dg['sum_abs_differences'])
    # axs.set_ylim([0, 1.25])
    # axs.set_xticks(grouped_by_dg.index.values)
    # axs.legend([frob, sumdif], ['frobenius distance', 'sum of absolute differences', ])
    # axs.set_title('R closeness to 2I vs. calibration shear magnitude')
    # axs.set_xlabel('calibration shear magnitude')
    # plt.savefig('plots/closeness_dg.png')
    # plt.show()

    # # "PSF reconvolution profile does not matter"
    # grouped_by_reconv_type = dataframe.groupby('reconv_psf_type').mean()
    # print(grouped_by_reconv_type) # gives some weird values that shouldn't be there. Gaussian has non Nan value in grouped_by table for reconv_psf parameters #TODO why??
    # # grouped_by_reconv_type.to_csv('table2.csv')
    #
    #
    # fig, axs = plt.subplots(1, 1)
    # x = np.asarray([0.2, 1.0])
    # frob = axs.bar(x, grouped_by_reconv_type['frobenius_norm'], width=0.2)
    # sumdif = axs.bar(x + 0.2, grouped_by_reconv_type['sum_abs_differences'], width=0.2)
    # axs.set_xticks([0.3, 1.1])
    # axs.set_xticklabels(['Gaussian', 'Moffat'])
    # axs.legend([frob, sumdif], ['frobenius distance', 'sum of absolute differences'])
    # axs.set_title('R closeness to 2I vs. reconvolution psf profile')
    # axs.set_xlabel('reconvolution PSF profile')
    # plt.savefig('plots/closeness_reconv_psf_type.png')
    # plt.show()
    #

    # # # Seeing the effect of deconvolution PSF size (Gaussian only) on R
    # grouped_by_deconv_size_gaussian_mean = dataframe.groupby('deconv_psf_sigma').mean()
    # grouped_by_deconv_size_gaussian_stdev = dataframe.groupby('deconv_psf_sigma').std()
    #
    # fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    #
    true_psf_sigma = 1.0 / 2.355
    # sigmas = grouped_by_deconv_size_gaussian_mean.index.values
    # dist_from_true = sigmas - true_psf_sigma * np.ones(len(sigmas))
    #
    # # frob = axs[0].plot(sigmas, grouped_by_deconv_size_gaussian_mean['frobenius_norm'], label='frobenius distance')
    # # sumdif = axs[1].plot(sigmas, grouped_by_deconv_size_gaussian_mean['sum_abs_differences'], label='sum of absolute differences')
    # frob_stdevs = axs[0].errorbar(sigmas, grouped_by_deconv_size_gaussian_mean['frobenius_norm'], yerr=grouped_by_deconv_size_gaussian_stdev['frobenius_norm'], capsize=5.0, label='frobenius distance')
    # sumdif_stdevs = axs[1].errorbar(sigmas, grouped_by_deconv_size_gaussian_mean['sum_abs_differences'], yerr=grouped_by_deconv_size_gaussian_stdev['sum_abs_differences'], capsize=5.0, label='sum of absolute differences')
    #
    # for ax in axs:
    #     actual = ax.axvline(true_psf_sigma, 0, 1, color='r', label='true PSF sigma')
    #     ax.legend(loc=1)
    #
    # fig.suptitle('Closeness of R matrix to 2*I for Gaussian deconvolution PSFs of varying sizes')
    #
    # plt.savefig('plots/deconv_gaussian_sigma.png')
    # plt.show()

    ## Violin plots for the same data
    # print(dataframe)
    gaussian_subframe = dataframe[dataframe['deconv_psf_type'] == 'Gaussian']
    gaussian_subframe = gaussian_subframe[dataframe['reconv_psf_type'] == 'Gaussian']
    # print(gaussian_subframe)
    sigma_distribution = gaussian_subframe[['deconv_psf_sigma', 'frobenius_norm', 'sum_abs_differences']]
    # print(sigma_distribution)

    grouped = sigma_distribution.groupby(by='deconv_psf_sigma')
    values = []
    frob_dataset = []
    sumdif_dataset = []
    for name, group in grouped:
        print(name)
        values.append(name)
        frob_dataset.append(group['frobenius_norm'].to_numpy())
        sumdif_dataset.append(group['sum_abs_differences'].to_numpy())



    fig, axs = plt.subplots(1, 2, figsize = (16, 8))
    width = 0.1
    axs[0].violinplot(frob_dataset, positions=values, showmeans=True, widths=np.ones(len(values))*width)
    axs[1].violinplot(sumdif_dataset, positions = values, showmeans=True, widths=np.ones(len(values))*width)
    axs[0].set_title('Frobenius Distance')
    axs[1].set_title('Sum of element-wise absolute differences')

    for ax in axs:
        actual = ax.axvline(true_psf_sigma, 0, 1, color='orange', label='true PSF sigma')
        ax.legend()
        ax.set_xlabel('Deconvolution PSF sigmas')

    fig.suptitle('Closeness of R to 2I for different deconvolution PSF sizes')

    version = 1

    if not os.path.exists('plots/violinplot.png'):
        plt.savefig('plots/violinplot.png')

    else:
        while os.path.exists('plots/violinplot' + '(' + str(version) + ').png'):
            version += 1

        plt.savefig('plots/violinplot' + '(' + str(version) + ').png')


    plt.show()


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

    # creating columns for the individual shear response matrix elements
    element_columns(results_df)

    print(results_df.columns)
    return results_df

    # TODO think about how to display different observed galaxies in the dataframe


def main():

    args = sys.argv[1:]

    if args[0] == '-generate':
        combinations = generate_combinations()
        vary_parameters(combinations, 'Results2.pickle')

    with open('Results2.pickle', 'rb') as f:
        stored_results = pickle.load(f)


    if args[0] == '-filter':
        pd_table = generate_df(stored_results)
        pd_table.to_csv('table.csv')
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print(pd_table)
        plot_data(pd_table)

    return 0


if __name__ == '__main__':
    main()