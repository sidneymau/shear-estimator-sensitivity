import data_generation
import galsim
import numpy as np
import metacal
import pickle
import sys
from multiprocessing import Pool
import pandas as pd
import matplotlib.pyplot as plt
import os.path
import seaborn as sns


# TABLE-MODIFYING FUNCTIONS 
# Should be used within plotting functions as needed 
# TODO consolidate these functions, make them more modular
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
    
    return dataframe


def true_psf_column_gaussian(dataframe):

    dataframe['true_psf_sigma'] = list(map(lambda obj: obj.sigma, dataframe['true_psf']))
    return dataframe


def true_psf_column_moffat(dataframe):
    dataframe['true_psf_fwhm'] = list(map(lambda obj: obj.fwhm, dataframe['true_psf']))


def gal_psf_ratio_gaussian(dataframe):

    dataframe['gal_sigma'] = list(map(lambda gal: gal.sigma, dataframe['original_gal']))
    dataframe['gal_psf_ratio'] = dataframe['gal_sigma'] / dataframe['true_psf_sigma']

    return dataframe


def gal_psf_ratio_moffat(dataframe):
    dataframe['gal_fwhm'] = list(map(lambda gal: gal.fwhm, dataframe['original_gal']))
    dataframe['gal_psf_ratio'] = dataframe['gal_fwhm'] / dataframe['true_psf_fwhm']    


def generate_df(results):
    """
    Takes in the results array and returns a pandas dataframe with columns
    for each parameter
    """
    # Loading the results table into a Pandas DataFrame
    results_df = pd.DataFrame(results, columns=['original_gal', 'oshear_g1', 'oshear_g2', 'true_psf', 'deconv_psf', 'reconv_psf', 'shear_estimation_psf', 'cshear_dg1', 'cshear_dg2', 'shear_estimator', 'pixel_scale', 'R', 'reconvolved_noshear', 'reconvolved_noshear_e1', 'reconvolved_noshear_e2'])
    return element_columns(results_df)
    # creating columns for psf parameters
    # create_psf_parameter_columns(results_df, 'deconv_psf')
    # create_psf_parameter_columns(results_df, 'reconv_psf')

    # # creating columns of the metrics for shear response matrix "closeness"
    # apply_metric(results_df, frobenius_norm)
    # apply_metric(results_df, sum_abs_differences)

    # # creating columns for the individual shear response matrix elements
    # element_columns(results_df)

    # # creating a column for the sigma of the true psf
    # # true_psf_column_gaussian(results_df)
    # true_psf_column_moffat(results_df)

    # # creating columns for original_gal sigma and gal/psf size ratio
    # # gal_psf_ratio_gaussian(results_df)
    # gal_psf_ratio_moffat(results_df)

    return results_df


# INDIVIDUAL PLOTTING FUNCTIONS
def save_fig_to_plots(figname):

    # finding file version
    version = 1
    if not os.path.exists('plots/' + figname + '.png'):
        plt.savefig('plots/' + figname + '.png')

    else:
        while os.path.exists('plots/' + figname + '(' + str(version) + ').png'):
            version += 1

        plt.savefig('plots/' + figname + '(' + str(version) + ').png')


def plot_R_elements(dataframe, xaxis_column, color_column, filename, x_units='', color_units='arcseconds'):

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    # fixing plotting scales
    diagmax = np.max([np.max(dataframe['R_11']), np.max(dataframe['R_22'])])
    # print(diagmax)
    diagmin = np.min([np.min(dataframe['R_11']), np.min(dataframe['R_22'])])
    offdiagmax = np.max([np.max(dataframe['R_21']), np.max(dataframe['R_12'])])
    # print(offdiagmax)
    offdiagmin = np.min([np.min(dataframe['R_21']), np.min(dataframe['R_12'])])

    scaling_factor = 1.01
    im = None
    for i in range(2):
        for j in range(2):
            element_string = 'R_' + str(i + 1) + str(j + 1)
            axs[i][j].set_title(element_string)
            im = axs[i][j].scatter(dataframe[xaxis_column], dataframe[element_string], c=dataframe[color_column], cmap='viridis', vmin=np.min(dataframe[color_column]), vmax=np.max(dataframe[color_column]))
            axs[i][j].tick_params(labelright=True)
            axs[i][j].set_xlabel(f"{xaxis_column} [{x_units}]")

            if i == j:
                axs[i][j].set_ylim(top=2 + scaling_factor * (diagmax - 2), bottom=diagmin)
            else:
                axs[i][j].set_ylim(top=scaling_factor * offdiagmax, bottom=offdiagmin)

    cbaxes = fig.add_axes([0.2, 0.1, 0.6, 0.01])
    cb = fig.colorbar(im, ax=axs[:], orientation='horizontal', shrink=0.5, cax=cbaxes)
    plt.subplots_adjust(hspace=0.3, wspace=0.4, bottom=0.2)
    cb.set_label(f"{color_column} [{color_units}]")
    fig.suptitle(f"Shear response matrix element values vs {xaxis_column}")

    save_fig_to_plots(filename)

    plt.show()


def all_gaussian(dataframe):

    dataframe['gal_sigma'] = [gal.sigma for gal in dataframe['original_gal']]
    dataframe['psf_sigma'] = [psf.sigma for psf in dataframe['true_psf']]
    dataframe['gal_psf_ratio'] = dataframe['gal_sigma'] / dataframe['psf_sigma']

    plot_R_elements(dataframe, 'gal_psf_ratio', 'gal_sigma', 'all_gaussian_gal_psf_ratio')
    

def all_moffat(dataframe):

    dataframe['gal_fwhm'] = [gal.fwhm for gal in dataframe['original_gal']]
    dataframe['moffat_psf_fwhm'] = [psf.fwhm for psf in dataframe['true_psf']]
    dataframe['gal_psf_ratio'] = dataframe['gal_fwhm'] / dataframe['moffat_psf_fwhm']
    
    # print(dataframe['reconvolved_noshear_e1'])
    # print(dataframe['reconvolved_noshear_e2'])

    plot_R_elements(dataframe, 'gal_psf_ratio', 'gal_fwhm', 'moffat_psfs_gal_psf_ratio')
    

def all_gaussian_different_ellipticies(dataframe):
    
    dataframe['gal_sigma'] = [gal.sigma for gal in dataframe['original_gal']]
    dataframe['psf_sigma'] = [psf.sigma for psf in dataframe['true_psf']]
    dataframe['gal_psf_ratio'] = dataframe['gal_sigma'] / dataframe['psf_sigma']
    # print(dataframe.columns)


    R_inv_list = [np.linalg.inv(R) for R in dataframe['R']]
    R_inv_array = np.asarray(R_inv_list)
    
    estimated_ellip_vec_list = []
    for i in range(len(dataframe['R'])):
        e1 = dataframe['reconvolved_noshear_e1'][i]
        e2 = dataframe['reconvolved_noshear_e2'][i]
        estimated_ellip_vec_list.append(np.array([[e1],[e2]]))
    
    estimated_ellip_vec_array = np.asarray(estimated_ellip_vec_list)
   
    estimated_shear_array = R_inv_array @ estimated_ellip_vec_array

    estimated_e1 = estimated_shear_array[:,0, 0]
    estimated_e2 = estimated_shear_array[:,1, 0]

    estimated_g1 = estimated_e1 / 2
    estimated_g2 = estimated_e2 / 2

    true_g1 = dataframe['oshear_g1'].to_numpy()[:]
    true_g2 = dataframe['oshear_g2'].to_numpy()[:]

 
    # print(estimated_g1)
    # print(estimated_g2)
    # print(true_g1)
    # print(true_g2)

    fig, axs = plt.subplots(1, 2, figsize=(15, 9))

    # axs[0].scatter(true_g1, estimated_g1 - true_g1)
    axs[0].set_xlabel('true_g1')
    axs[0].set_ylabel('(estimated_g1 - true_g1) / true_g1')
    axs[0].set_title('g1')
    # axs[1].scatter(true_g2, estimated_g2 - true_g2)
    axs[1].set_xlabel('true_g2')
    axs[1].set_ylabel('(estimated_g2 - true_g2) / true_g2')
    axs[1].set_title('g2')

    im = axs[0].scatter(true_g1, (estimated_g1 - true_g1)/true_g1, c=dataframe['gal_psf_ratio'][:], cmap='viridis')
    im = axs[1].scatter(true_g2, (estimated_g2 - true_g2)/true_g2, c=dataframe['gal_psf_ratio'][:], cmap='viridis')

    # axs[0].plot([0, 0.1], [0, 0.1], label='estimated g1 = true g1')
    # axs[1].plot([0, 0.1], [0, 0.1], label='estimated g2 = true g2')

    # axs[0].legend()
    # axs[1].legend()

    # cbaxes = fig.add_axes([0.2, 0.1, 0.6, 0.01])
    cb = fig.colorbar(im, ax=axs[:], orientation='horizontal', shrink=0.45) #, cax=cbaxes)
    cb.set_label('galaxy size to psf size ratio')

    fig.suptitle('Estimated g fractional error vs true g by element')

    # save_fig_to_plots('fractional error vs true g1 and g2')
    
    plt.show()
   

def all_gaussian_varying_cshear_pixelscale(dataframe):
    # print(dataframe['cshear_dg1'])
    pass

def generate_images(dataframe):
    """
    Goal: make images of one of the cases where R11 and R22 were the highest
    """
    # find the row with the parameters that generated the highest R11

    max_R_11 = np.max(dataframe['R_11'])

    max_R_11_combo = dataframe[dataframe['R_11'] == max_R_11]

    image_dict = {}

    true_galaxy = max_R_11_combo['original_gal'].values[0]

    image_dict['true_galaxy'] = true_galaxy

    true_psf = max_R_11_combo['true_psf'].values[0]
    image_dict['true_psf'] = true_psf

    convolved_galaxy = galsim.Convolve(true_galaxy, true_psf)
    image_dict['convolved_galaxy'] = convolved_galaxy

    deconvolved_galaxy = galsim.Convolve(convolved_galaxy, galsim.Convolve(max_R_11_combo['deconv_psf'].values[0])) # TODO could be a possible problem line
    image_dict['deconvolved_galaxy'] = deconvolved_galaxy

    reconvolved_galaxy = galsim.Convolve(deconvolved_galaxy, max_R_11_combo['reconv_psf'].values[0])
    image_dict['reconvolved_galaxy'] = reconvolved_galaxy

    # important parameters
    print('\n' * 4)
    print('original galaxy sigma: ', true_galaxy.sigma)
    print('true psf sigma: ', true_psf.sigma)
    print('\n' * 4)


    pixel_scale = 0.2

    maximum_list = []
    minimum_list = []

    for name, obj in image_dict.items():
        image_array = obj.drawImage(scale=pixel_scale).array
        image_dict[name] = image_array
        maximum_list.append(np.max(image_array))
        minimum_list.append(np.min(image_array))

    vmax = np.max(maximum_list)
    vmin = np.min(minimum_list)

    fig, axs = plt.subplots(1, len(image_dict))

    counter = 0
    im = None
    for name, image_array in image_dict.items():
        im = axs[counter].imshow(image_array)
        axs[counter].set_title(name)

        counter += 1

    plt.show()
    print(image_dict)


# MASTER FUNCTIONS
def master_plotting(dataframe):

    ## Calling different plotting functions ##
    # r_vs_calshearmag(dataframe)
    # r_vs_reconv_profile(dataframe)
    # r_vs_gaussian_deconv_psf_size(dataframe)
    # r_vs_gaussian_deconv_psf_size_violin(dataframe)
    # sanity_check_1(dataframe)
    # all_gaussian(dataframe)
    # all_moffat(dataframe)
    # generate_images(dataframe)
    all_gaussian_different_ellipticies(dataframe)
    # all_gaussian_varying_cshear_pixelscale(dataframe)


def pickle_to_modified_dataframe(filename):

    with open(filename, 'rb') as f:
        stored_results = pickle.load(f)

    dataframe = generate_df(stored_results)

    return dataframe


def main():

    args = sys.argv[1:]

    if (len(args) != 1):
        print('Argument missing')
        print('Use: python metacal_plotting.py [filename]')
        return 1

    filename = args[0]
    modified_dataframe = pickle_to_modified_dataframe(filename)

    master_plotting(modified_dataframe)



if __name__ == '__main__':
    main()
