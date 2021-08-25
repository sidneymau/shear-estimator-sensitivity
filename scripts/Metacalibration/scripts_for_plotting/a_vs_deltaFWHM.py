import mcObjects
import sys
import numpy as np
import matplotlib.pyplot as plt

# no change to galaxy profiles
true_psf_profiles = {'gaussian': 0}
shear_ests = ['REGAUSS']
pixel_scales = [0.02]
# calibration shears unchanged
true_psf_fwhms = [0.7]
# gal_psf_ratios = [2.0]
gal_psf_ratios = [1.0]
# offsets unchanged
wrong_psf_fwhms = np.arange(true_psf_fwhms[0] - 0.005, true_psf_fwhms[0] + 0.006, 0.001)
#wrong_psf_fwhms = np.arange(0.68, 0.721, 0.001)
# wrong_psf_fwhms = [0.7]


def generate():
    # DESCRIPTION OF WHAT'S BEING GENERATED
    pass


def plot_a(obj, plotname=None):

    true_fwhm = 0.7

    obj.slice('deconv_psf_fwhm', lambda f: abs(f - true_fwhm) < 0.0055)
    nomc_ylims, mc_ylims, c1_nouse, c2_nouse, a1_nouse, a2_nouse = obj.with_without_metacal(color_column='deconv_psf_fwhm')

    
    obj.reset()

    if True:

        wrong_psf_fwhm_list = []
        a1s = [] 

        for i in range(len(wrong_psf_fwhms)):
            if abs(wrong_psf_fwhms[i] - true_fwhm) < 0.0055:
                obj.slice('deconv_psf_fwhm', value=wrong_psf_fwhms[i])
                # obj.slice('deconv_psf_fwhm', lambda dummy: True)
                diff = wrong_psf_fwhms[i] - true_fwhm

                nomc_ylims, mc_ylims, c1_mc, c2_mc, a1, a2 = obj.with_without_metacal(show=False, nomc_ylims=nomc_ylims, mc_ylims=mc_ylims) #, plotname=f"{i}_fwhm_diff={diff:5f}_ratio=1.0_scale=0.02")

                wrong_psf_fwhm_list.append(wrong_psf_fwhms[i])
                a1s.append(a1)
                # obj.plot_quadratic_m(color_column='deconv_psf_fwhm', show=False, ylims=ylims, plotname=str(i))
                obj.reset()

        fig, axs = plt.subplots(1, 1, figsize=(5, 5))

        x = np.asarray(wrong_psf_fwhm_list) - true_fwhm * np.ones(len(wrong_psf_fwhm_list))
        y = np.asarray(a1s)
        axs.scatter(x, y)

        slope, intercept = np.polyfit(x, y, 1)
        axs.plot(x, slope*x + intercept, label=f"slope = {slope} \n intercept = {intercept}")

        axs.set_xlabel(f'delta_fwhm')
        axs.set_ylabel(f'a (prefactor of quadratic fit to fractional error')
        axs.set_title('a (prefactor of quadratic error) vs. delta_fwhm')

        axs.axhline(color='k')
        axs.axvline(color='k')

        axs.legend()

        if plotname is not None:
            plt.savefig('plots/' + plotname)

        plt.show()


def plot():
    # DESCRIPTION OF WHAT'S BEING PLOTTED

    for ratio in [0.5, 1.0, 2.0]:
        for scale in [0.2, 0.02]:
            obj = mcObjects.mcSummaryObject(f'pickles/all_gaussian_wrong_psfs_trueFWHM=0.7_ratio={ratio}_scale={scale}_miniscule_errors.pickle')
            plot_a(obj, plotname=f'a_vs_deltaFWHM_ratio={ratio}_scale={scale}.png')

    import pdb; pdb.set_trace()

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


