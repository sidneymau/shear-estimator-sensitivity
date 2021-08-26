import mcObjects
import sys
import numpy as np
import pdb
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm

true_psf_profiles = {'gaussian': 0}
shear_ests = ['REGAUSS']
pixel_scales = [0.2]
# calibration shears unchanged
true_psf_fwhms = [0.7]

# Defining psf_gal_ratios for even spacing on plots
psf_gal_ratios = np.arange(0.25, 2.6, 0.25)
gal_psf_ratios = 1 / psf_gal_ratios

# Generating known values of fractional errors
psf_var_frac_errs = np.arange(-0.02, 0.02, 0.001)

# mathetmatical conversion for generating wrong_fwhms from desired psf_var_frac_errs
wrong_psf_fwhms = 0.7 * np.sqrt(np.ones(len(psf_var_frac_errs)) - psf_var_frac_errs)



def generate():
    # DESCRIPTION OF WHAT'S BEING GENERATED

    comboObj = mcObjects.comboObject()

    comboObj.true_psf_profiles = true_psf_profiles
    comboObj.shape_mes_algs = shear_ests
    comboObj.pixel_scales = pixel_scales
    comboObj.true_psf_fwhms = [0.7]
    comboObj.gal_psf_ratios = gal_psf_ratios
    comboObj.psf_var_frac_errs = psf_var_frac_errs
    comboObj.wrong_psf_fwhms = wrong_psf_fwhms


    comboObj.print_parameters()

    pdb.set_trace()

    comboObj.generate_combinations('psf_var_frac_err=pm0.02_psf_gal_ratios0.25_2.5', 4)

    pass

def plot():
    # DESCRIPTION OF WHAT'S BEING PLOTTED

    obj = mcObjects.mcSummaryObject('pickles/psf_var_frac_err=pm0.02_psf_gal_ratios0.25_2.5.pickle')

    def get_c(obj, dataframe):

        a1, b1, c1 = obj.get_fit_parameters(dataframe)

        return c1 

    obj.df['psf_gal_ratio'] = 1 / obj.df['gal_psf_ratio']
    obj.df['psf_var_frac_err'] = obj.df['mcObject'].map(lambda obj: obj.psf_var_frac_err)


    psf_var_frac_err_to_c = {}
    for err in psf_var_frac_errs:
    # for wrong_fwhm in wrong_psf_fwhms:

        sliced_by_drpsf = obj.slice('psf_var_frac_err', value=err, table_in=obj.df)

        c_list = []
        for ratio in psf_gal_ratios:
            subset = obj.slice('psf_gal_ratio', value=ratio, table_in=sliced_by_drpsf)

            print(subset.head())

            c = get_c(obj, subset)
            c_list.append(c) 

        psf_var_frac_err_to_c[err] = c_list
     

    def plot_for_err(ax, err, cmap='cividis', many=False):
        color_map = matplotlib.cm.get_cmap(cmap)
        norm = matplotlib.colors.Normalize(vmin=np.min(psf_var_frac_errs), vmax=np.max(psf_var_frac_errs))

        color = color_map(norm(err))

        c_array = psf_var_frac_err_to_c[err]
        ax.scatter(psf_gal_ratios, c_array, color=color) 

        x = np.linspace(0, np.max(psf_gal_ratios)* 1.1)

        ax.plot(x, -err*x*x, color=color)
        # ax.plot(psf_gal_ratios_array, -psf_gal_ratios_array*err)

        if many:
            ax.set_xlabel('psf_gal_ratio')
            ax.set_ylabel('c')
            # ax.set_xticks(psf_gal_ratios_array)

            ax.legend()

        return norm, cmap

    size = 9
    fig, axs = plt.subplots(1, 1, figsize=(size * 1.2, size))   

    norm = None
    cmap = None
    for err in psf_var_frac_errs[::4]:
        norm, cmap = plot_for_err(axs, err)

    # creating color bar
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs)
    cb.set_label(fr"$\frac{{\Delta (r_{{PSF}}^2)}}{{r_{{PSF}}^2}}$", rotation=0, fontsize=14)

    # Adding axis labels
    fontsize = 16
    axs.set_xlabel(r"$\frac{r_{PSF}}{r_{Galaxy}}$", fontsize=fontsize)
    axs.set_ylabel(r"$m_0$", fontsize=fontsize)
    axs.set_title(r"$m_0$ vs. $\frac{FWHM_{PSF}}{FWHM_{Galaxy}}$ for different values of $\frac{\Delta(r_{PSF}^2)}{r_{PSF}^2}$", fontsize=fontsize)

    plt.savefig('m0_vs_psf_gal_ratio_for_different_psf_variance_fractional_errors.png')

    plt.show()

    pdb.set_trace()



    pass


def main():
    args = sys.argv[1:]

    if len(args) != 1:
        raise Exception('-generate or -plot')
    
    if args[0] == '-generate':
        generate()

    if args[0] == '-plot':
        plot()




if __name__ == '__main__':
    main()


