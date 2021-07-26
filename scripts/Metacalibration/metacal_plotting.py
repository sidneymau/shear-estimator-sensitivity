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

def save_fig_to_plots(figname):

    # finding file version
    version = 1
    if not os.path.exists('plots/' + figname + '.png'):
        plt.savefig('plots/' + figname + '.png')

    else:
        while os.path.exists('plots/' + figname + '(' + str(version) + ').png'):
            version += 1

        plt.savefig('plots/' + figname + '(' + str(version) + ').png')


def r_vs_calshearmag(dataframe):

    # test plotting R closeness to 2I vs. calibration shear magnitude
    grouped_by_dg = dataframe.groupby('dg1').mean()
    grouped_by_dg = grouped_by_dg[['frobenius_norm', 'sum_abs_differences']] # don't need to include dg1 because that's the table
    print(grouped_by_dg.index.name) # accessing name of variable indexed by
    print(grouped_by_dg.index.values) # to access the variable grouped by, need to do df.index.values

    fig, axs = plt.subplots(1, 1)

    frob = axs.scatter(grouped_by_dg.index.values, grouped_by_dg['frobenius_norm'])
    sumdif = axs.scatter(grouped_by_dg.index.values, grouped_by_dg['sum_abs_differences'])
    axs.set_ylim([0, 1.25])
    axs.set_xticks(grouped_by_dg.index.values)
    axs.legend([frob, sumdif], ['frobenius distance', 'sum of absolute differences', ])
    axs.set_title('R closeness to 2I vs. calibration shear magnitude')
    axs.set_xlabel('calibration shear magnitude')
    plt.savefig('plots/closeness_dg.png')
    plt.show()


def r_vs_reconv_profile(dataframe):

    # "PSF reconvolution profile does not matter"
    grouped_by_reconv_type = dataframe.groupby('reconv_psf_type').mean()
    print(grouped_by_reconv_type) # gives some weird values that shouldn't be there. Gaussian has non Nan value in grouped_by table for reconv_psf parameters #TODO why??
    # grouped_by_reconv_type.to_csv('table2.csv')


    fig, axs = plt.subplots(1, 1)
    x = np.asarray([0.2, 1.0])
    frob = axs.bar(x, grouped_by_reconv_type['frobenius_norm'], width=0.2)
    sumdif = axs.bar(x + 0.2, grouped_by_reconv_type['sum_abs_differences'], width=0.2)
    axs.set_xticks([0.3, 1.1])
    axs.set_xticklabels(['Gaussian', 'Moffat'])
    axs.legend([frob, sumdif], ['frobenius distance', 'sum of absolute differences'])
    axs.set_title('R closeness to 2I vs. reconvolution psf profile')
    axs.set_xlabel('reconvolution PSF profile')
    plt.savefig('plots/closeness_reconv_psf_type.png')
    plt.show()


def r_vs_gaussian_deconv_psf_size(dataframe):

    # Seeing the effect of deconvolution PSF size (Gaussian only) on R
    grouped_by_deconv_size_gaussian_mean = dataframe.groupby('deconv_psf_sigma').mean()
    grouped_by_deconv_size_gaussian_stdev = dataframe.groupby('deconv_psf_sigma').std()

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    true_psf_sigma = 1.0 / 2.355
    sigmas = grouped_by_deconv_size_gaussian_mean.index.values
    dist_from_true = sigmas - true_psf_sigma * np.ones(len(sigmas))

    # frob = axs[0].plot(sigmas, grouped_by_deconv_size_gaussian_mean['frobenius_norm'], label='frobenius distance')
    # sumdif = axs[1].plot(sigmas, grouped_by_deconv_size_gaussian_mean['sum_abs_differences'], label='sum of absolute differences')
    frob_stdevs = axs[0].errorbar(sigmas, grouped_by_deconv_size_gaussian_mean['frobenius_norm'], yerr=grouped_by_deconv_size_gaussian_stdev['frobenius_norm'], capsize=5.0, label='frobenius distance')
    sumdif_stdevs = axs[1].errorbar(sigmas, grouped_by_deconv_size_gaussian_mean['sum_abs_differences'], yerr=grouped_by_deconv_size_gaussian_stdev['sum_abs_differences'], capsize=5.0, label='sum of absolute differences')

    for ax in axs:
        actual = ax.axvline(true_psf_sigma, 0, 1, color='r', label='true PSF sigma')
        ax.legend(loc=1)

    fig.suptitle('Closeness of R matrix to 2*I for Gaussian deconvolution PSFs of varying sizes')

    plt.savefig('plots/deconv_gaussian_sigma.png')
    plt.show()


def r_vs_gaussian_deconv_psf_size_violin(dataframe):

    # Violin plots for the same data
    gaussian_subframe = dataframe[dataframe['deconv_psf_type'] == 'Gaussian']
    gaussian_subframe = gaussian_subframe[dataframe['reconv_psf_type'] == 'Gaussian']
    sigma_distribution = gaussian_subframe[['deconv_psf_sigma', 'frobenius_norm', 'sum_abs_differences']]

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

    true_psf_sigma = 1.0 / 2.355
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


def plot_R_elements(dataframe, xaxis_column, gal_sigma_column):

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    # fixing plotting scales
    diagmax = np.max([np.max(dataframe['R_11']), np.max(dataframe['R_22'])])
    # print(diagmax)
    diagmin = np.min([np.min(dataframe['R_11']), np.min(dataframe['R_22'])])
    offdiagmax = np.max([np.max(dataframe['R_21']), np.max(dataframe['R_12'])])
    # print(offdiagmax)
    offdiagmin = np.min([np.min(dataframe['R_21']), np.min(dataframe['R_12'])])

    # # splitting out by color
    # color_dict = {}
    # for sigma in gal_sigma:
    #     if sigma not in color_dict:
    #         color_dict[sigma] = ''
    #     color_dict[sigma] = c


    scaling_factor = 1.01
    im = None
    for i in range(2):
        for j in range(2):
            element_string = 'R_' + str(i + 1) + str(j + 1)
            axs[i][j].set_title(element_string)
            im = axs[i][j].scatter(dataframe[xaxis_column], dataframe[element_string], c=dataframe[gal_sigma_column], cmap='viridis', vmin=np.min(dataframe[gal_sigma_column]), vmax=np.max(dataframe[gal_sigma_column]))
            axs[i][j].tick_params(labelright=True)
            axs[i][j].set_xlabel(xaxis_column)

            if i == j:
                axs[i][j].set_ylim(top=2 + scaling_factor * (diagmax - 2), bottom=diagmin)
            else:
                axs[i][j].set_ylim(top=scaling_factor * offdiagmax, bottom=offdiagmin)

    cbaxes = fig.add_axes([0.2, 0.1, 0.6, 0.01])
    cb = fig.colorbar(im, ax=axs[:], orientation='horizontal', shrink=0.5, cax=cbaxes)
    plt.subplots_adjust(hspace=0.3, wspace=0.4, bottom=0.2)
    cb.set_label(gal_sigma_column + ' [arcseconds]')
    fig.suptitle(f'Shear response matrix element values vs {xaxis_column}')

    save_fig_to_plots(xaxis_column)

    plt.show()


def sanity_check_1(dataframe):
    print(dataframe.columns)
    # print(dataframe['deconv_psf'])
    # print(dataframe['true_psf'])

    fig, axs = plt.subplots(2, 2, figsize=(12, 9))

    # fixing plotting scales
    diagmax = np.max([np.max(dataframe['R_11']), np.max(dataframe['R_22'])])
    print(diagmax)
    diagmin = np.min([np.min(dataframe['R_11']), np.min(dataframe['R_22'])])
    offdiagmax = np.max([np.max(dataframe['R_21']), np.max(dataframe['R_12'])])
    print(offdiagmax)
    offdiagmin = np.min([np.min(dataframe['R_21']), np.min(dataframe['R_12'])])

    scaling_factor = 1.01
    for i in range(2):
        for j in range(2):
            element_string = 'R_' + str(i + 1) + str(j + 1)
            axs[i][j].set_title(element_string)
            axs[i][j].plot(dataframe['true_psf_sigma'], dataframe[element_string])
            axs[i][j].tick_params(labelright=True)
            axs[i][j].set_xlabel('True PSF sigma)')

            if i == j:
                axs[i][j].set_ylim(top=2 + scaling_factor * (diagmax - 2), bottom=diagmin)
            else:
                axs[i][j].set_ylim(top=scaling_factor * offdiagmax, bottom=offdiagmin)

    fig.suptitle('Shear response matrix element values vs true PSF size')

    save_fig_to_plots('element_plot')

    plt.show()


def sanity_check_2(dataframe):
    # print(dataframe)
    # print(dataframe.columns)
    # print(dataframe['gal_psf_ratio'])
    print(dataframe.columns)
    plot_R_elements(dataframe, 'gal_psf_ratio', 'gal_sigma')
    


def all_moffat(dataframe):
    print(dataframe.columns)
    plot_R_elements(dataframe, 'gal_psf_ratio', 'gal_fwhm')

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
    print('Original galaxy sigma: ', true_galaxy.sigma)
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


def master_plotting(dataframe):

    ## Calling different plotting functions ##
    # r_vs_calshearmag(dataframe)
    # r_vs_reconv_profile(dataframe)
    # r_vs_gaussian_deconv_psf_size(dataframe)
    # r_vs_gaussian_deconv_psf_size_violin(dataframe)
    # sanity_check_1(dataframe)
    # sanity_check_2(dataframe)
    all_moffat(dataframe)
    # generate_images(dataframe)


def pickle_to_modified_dataframe(filename):

    with open(filename, 'rb') as f:
        stored_results = pickle.load(f)

    dataframe = data_generation.generate_df(stored_results)

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
