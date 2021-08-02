# CONTAINS SCRATCH CODE

# Old plotting function that I don't need anymore
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


# generation function for early plot where I forgot to plot ratios
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


