import data_generation
import galsim
import numpy as np
import scipy
import metacal
import pickle
import sys
from multiprocessing import Pool
import pandas as pd
import matplotlib.pyplot as plt

# matplotlib.rc('xtick', labelsize=20)
# matplotlib.rc('ytick', labelsize=20)
# plt.rcParams.update({'font.size': 18})

import os.path
import seaborn as sns


# TABLE-MODIFYING FUNCTIONS 

def element_columns(dataframe):
    """
    Adds as columns the 4 individual elements of the shear response matrix
    """
    for i in range(0, 2):
        for j in range(0, 2):
            dataframe['R_' + str(i + 1) + str(j + 1)] = list(map(lambda r: r[i][j], dataframe['R']))
    
    return dataframe


def generate_df(results):
    """
    Takes in the results array and returns a pandas dataframe with columns
    for each parameter
    """
    # Loading the results table into a Pandas DataFrame
    results_df = pd.DataFrame(results, columns=['original_gal', 'oshear_g1', 'oshear_g2', 'true_psf', 'deconv_psf', 'reconv_psf', 'shear_estimation_psf', 'cshear_dg1', 'cshear_dg2', 'shear_estimator', 'pixel_scale', 'R', 'reconvolved_noshear', 'reconvolved_noshear_e1', 'reconvolved_noshear_e2'])
    return element_columns(results_df)


# INDIVIDUAL PLOTTING FUNCTIONS
def save_fig_to_plots(figname):
    """
    Function for my own sanity, used for saving files to a specific directory
    without overwriting any old ones.
    """
    # finding file version
    version = 1
    if not os.path.exists('plots/' + figname + '.png'):
        plt.savefig('plots/' + figname + '.png')

    else:
        while os.path.exists('plots/' + figname + '(' + str(version) + ').png'):
            version += 1
        plt.savefig('plots/' + figname + '(' + str(version) + ').png')

def plot_R_elements(dataframe, xaxis_column, color_column, filename, x_units='', color_units='arcseconds'):
    """
    Generates a plot of each element of the shear response matrix as a function of the parameter
    "xaxis_column"
    """
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
    """
    Plots the elements of the shear response matrix R for a master dataframe of
    gaussian original galaxies and Gaussian PSFs for different ratios of galaxy size
    to PSF size
    """
    dataframe['gal_sigma'] = [gal.sigma for gal in dataframe['original_gal']]
    dataframe['psf_sigma'] = [psf.sigma for psf in dataframe['true_psf']]
    dataframe['gal_psf_ratio'] = dataframe['gal_sigma'] / dataframe['psf_sigma']

    plot_R_elements(dataframe, 'gal_psf_ratio', 'gal_sigma', 'all_gaussian_gal_psf_ratio')
    

def all_moffat(dataframe):
    """
    Plots the elements of the shear response matrix R for a master dataframe of
    gaussian original galaxies and Moffat PSFs for different ratios of galaxy size
    to PSF size
    """
    dataframe['gal_fwhm'] = [gal.fwhm for gal in dataframe['original_gal']]
    dataframe['moffat_psf_fwhm'] = [psf.fwhm for psf in dataframe['true_psf']]
    dataframe['gal_psf_ratio'] = dataframe['gal_fwhm'] / dataframe['moffat_psf_fwhm']
    
    # print(dataframe['reconvolved_noshear_e1'])
    # print(dataframe['reconvolved_noshear_e2'])

    plot_R_elements(dataframe, 'gal_psf_ratio', 'gal_fwhm', 'moffat_psfs_gal_psf_ratio')
    

def all_gaussian_different_ellipticies_log_m(dataframe, plotname):
    """
    Takes in the master dataframe, and generates a plot of m = (estimated_gi - true_gi) / true_gi
    for each element
    """

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
    axs[0].set_ylabel('[(estimated_g1 - true_g1)/true_g1]' )
    axs[0].set_ylabel(r'$(\frac{{g_1}_{est} - {g_1}_{est}}{{g_1}_{true}})$')
    axs[0].set_title('g1')
    # axs[1].scatter(true_g2, estimated_g2 - true_g2)
    axs[1].set_xlabel('true_g2')
    axs[1].set_ylabel(r'$(\frac{{g_2}_{est} - {g_2}_{est}}{{g_2}_{true}})$')
    axs[1].set_title('g2')

    y1 = (estimated_g1 - true_g1)/true_g1
    y2 = (estimated_g2 - true_g2)/true_g2

    im = axs[0].scatter(true_g1, y1, c=dataframe['gal_psf_ratio'][:], cmap='cividis')
    im = axs[1].scatter(true_g2, y2, c=dataframe['gal_psf_ratio'][:], cmap='cividis')

    plt.subplots_adjust(hspace=1.5, wspace=0.3)


    for ax in axs:
        ax.set_ylim(bottom=1e-3, top=1e-1)
        ax.set_yscale('log')

    # m1, b1 = np.polyfit(true_g1, estimated_g1 - true_g1, 1)
    # m2, b2 = np.polyfit(true_g1, estimated_g1 - true_g1, 1)

    # axs[0].plot(true_g1, m1 * true_g1 + b1, label=f"m1 = {m1}, b1 = {b1}")
    # axs[1].plot(true_g2, m2 * true_g2 + b2, label=f"m2 = {m2}, b2 = {b2}")
    # axs[0].legend()
    # axs[1].legend()

    # print('m1: ', m1)
    # print('m2: ', m2)

    # axs[0].plot([0, 0.1], [0, 0.1], label='estimated g1 = true g1')
    # axs[1].plot([0, 0.1], [0, 0.1], label='estimated g2 = true g2')

    # axs[0].legend()
    # axs[1].legend()

    # cbaxes = fig.add_axes([0.2, 0.1, 0.6, 0.01])
    cb = fig.colorbar(im, ax=axs[:], orientation='horizontal', shrink=0.45) #, cax=cbaxes)
    cb.set_label('galaxy size to psf size ratio')

    fig.suptitle(r'$(\frac{{g_i,}_{est} - {g_i,}_{est}}{{g_i,}_{true}})$ by element')

    # save_fig_to_plots(plotname)
    
    plt.show()
   

def all_gaussian_different_ellipticities_m(dataframe, color_column, plotname, color=True, save=False):
    """
    Takes in the master dataframe, and generates a plot of m = (estimated_gi - true_gi) / true_gi
    for each element
    """ 


    # criterion1 = dataframe['gal_sigma'].map(lambda sigma: abs(sigma - 0.424628 < 0.01))
    # dataframe = dataframe[criterion1]

    R_inv_list = [np.linalg.inv(R) for R in dataframe['R']]
    R_inv_array = np.asarray(R_inv_list)
   
    
    estimated_ellip_vec_list = []
    for i in range(len(dataframe['R'])):
        e1 = dataframe['reconvolved_noshear_e1'].to_numpy()[i]
        e2 = dataframe['reconvolved_noshear_e2'].to_numpy()[i]
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

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    # axs[0].scatter(true_g1, estimated_g1 - true_g1)
    axs[0].set_xlabel('true_g1')
    axs[0].set_ylabel('[(estimated_g1 - true_g1)/true_g1]' )
    axs[0].set_ylabel(r'$m = (\frac{{g_1}_{est} - {g_1}_{est}}{{g_1}_{true}})$')
    axs[0].set_title('g1')
    # axs[1].scatter(true_g2, estimated_g2 - true_g2)
    axs[1].set_xlabel('true_g2')
    axs[1].set_ylabel(r'$m = (\frac{{g_2}_{est} - {g_2}_{est}}{{g_2}_{true}})$')
    axs[1].set_title('g2')

    y1 = (estimated_g1 - true_g1)/true_g1
    y2 = (estimated_g2 - true_g2)/true_g2

    if color:
        im = axs[0].scatter(true_g1, y1, c=dataframe[color_column][:], cmap='cividis')
        im = axs[1].scatter(true_g2, y2, c=dataframe[color_column][:], cmap='cividis')
        cbaxes = fig.add_axes([0.2, 0.05, 0.6, 0.01])
        cb = fig.colorbar(im, ax=axs[:], orientation='horizontal', shrink=0.45, cax=cbaxes)
        cb.set_label(color_column)
    
    else:
        im = axs[0].scatter(true_g1, y1, c='r', label='m')
        im = axs[1].scatter(true_g2, y2, c='r', label='m')

    plt.subplots_adjust(hspace=2.0, wspace=0.3)

    idx1 = np.nonzero(true_g1) 
    idx2 = np.nonzero(true_g2)

    print(true_g1)
    print(true_g2)

    # import pdb;pdb.set_trace()

    a1, b1, c1 = np.polyfit(true_g1[idx1], y1[idx1], 2)
    a2, b2, c2 = np.polyfit(true_g2[idx2], y2[idx2], 2)

    x = np.linspace(-0.05, 0.05, 20)


    axs[0].plot(x, a1*x*x + b1*x + c1, c='k', label=f"a1 = {a1:.2f} \n b1 = {b1:.2f} \n c1 = {c1:.2f}", zorder=0)
    axs[1].plot(x, a2*x*x + b2*x + c2, c='k', label=f"a2 = {a2:.2f} \n b2 = {b2:.2f} \n c1 = {c2:.2f}", zorder=0)
    axs[0].legend()
    axs[1].legend()



    fig.suptitle(r'$m = (\frac{{g_i,}_{est} - {g_i,}_{est}}{{g_i,}_{true}})$ by element')

    if save:
        save_fig_to_plots(plotname)
    
    plt.show()
   


def all_gaussian_varying_cshear_oshear_pixelscale(dataframe, filename, pixel_scale=0.2, cshear_dg=0.01):
    """
    INCOMPLETE
    """
    print(dataframe.columns)
    print(dataframe.shape)

    dataframe['gal_sigma'] = [gal.sigma for gal in dataframe['original_gal']]
    dataframe['psf_sigma'] = [psf.sigma for psf in dataframe['true_psf']]
    dataframe['gal_psf_ratio'] = dataframe['gal_sigma'] / dataframe['psf_sigma']
  
    filtered = dataframe
    
        # print(dataframe['gal_psf_ratio'])
    ratio_criterion = filtered['gal_psf_ratio'].map(lambda ratio: abs(ratio - 2.0) < 0.001)
    filtered = filtered[ratio_criterion]

    # # filter by absolute galaxy size as well
    # size_criterion = filtered['gal_sigma'].map(lambda sigma: abs(sigma - 1.019108) < 0.00001)
    # filtered = filtered[size_criterion]

    print(np.mean(filtered.gal_sigma))
    
    # pixel scale filter
    filtered_by_pixel = filtered[filtered['pixel_scale'] == pixel_scale]

    # cshear_dg filter
    criterion1 = filtered_by_pixel['cshear_dg1'].map(lambda x: x == cshear_dg or x == 0.0)
    criterion2 = filtered_by_pixel['cshear_dg2'].map(lambda x: x == cshear_dg or x == 0.0)

    filtered_by_cshear = filtered_by_pixel[criterion1]
    filtered_by_cshear = filtered_by_pixel[criterion2]

    
    # print(filtered.shape)

    filtered_final = filtered_by_cshear.reset_index(drop=True)
    # print(filtered['reconvolved_noshear_e1'])

    # all_gaussian_different_ellipticies_log_m(filtered, filename)
    all_gaussian_different_ellipticities_m(filtered_final, 'gal_psf_ratio', filename, color=True, save=False) # change back to filtered if needed
    
def generate_images(dataframe):
    """
    Generates images of one of the cases where R11 and R22 were the highest
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


# MASTER FUNCTIONS
def master_plotting(dataframe, filename):
    """
    Calls whichever plotting function isn't commented out
    """
    ## Calling different plotting functions ##
    # r_vs_calshearmag(dataframe)
    # r_vs_reconv_profile(dataframe)
    # r_vs_gaussian_deconv_psf_size(dataframe)
    # r_vs_gaussian_deconv_psf_size_violin(dataframe)
    # sanity_check_1(dataframe)
    # all_gaussian(dataframe)
    # all_moffat(dataframe)
    # generate_images(dataframe)
    # all_gaussian_different_ellipticies_log_m(dataframe, filename)
    # all_gaussian_varying_cshear_oshear_pixelscale(dataframe, 'test')
    # all_gaussian_different_ellipticies_m(dataframe, filename)
    all_gaussian_varying_cshear_oshear_pixelscale(dataframe, filename, pixel_scale=0.02, cshear_dg=0.01)

def pickle_to_modified_dataframe(filename):
    """
    Opens a pickle file and generates the master dataframe of galsim
    objects and parameters
    """
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

    # pulling out plot name from cmd line argument
    slash_loc = len(filename) - 1
    while slash_loc >= 0 and filename[slash_loc] != '/':
        slash_loc -= 1

    period_loc = len(filename) - 1
    while period_loc >= 0 and filename[period_loc] != '.':
        period_loc -= 1
    
    
    plotname = filename[slash_loc + 1:period_loc]


    master_plotting(modified_dataframe, plotname)



if __name__ == '__main__':
    main()
