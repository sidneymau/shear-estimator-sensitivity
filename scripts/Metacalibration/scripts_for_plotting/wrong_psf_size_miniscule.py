import mcObjects
import sys
import galsim
import numpy as np
import string
import matplotlib.pyplot as plt




# no change to galaxy profiles
true_psf_profiles = {'gaussian': 0}
shear_ests = ['REGAUSS']
pixel_scales = [0.2]
# calibration shears unchanged
true_psf_fwhms = [0.7]
# gal_psf_ratios = [2.0]
gal_psf_ratios = [0.5]
# offsets unchanged
wrong_psf_fwhms = np.arange(true_psf_fwhms[0] - 0.005, true_psf_fwhms[0] + 0.006, 0.001)
#wrong_psf_fwhms = np.arange(0.68, 0.721, 0.001)
# wrong_psf_fwhms = [0.7]



def generate():
    # DESCRIPTION OF WHAT'S BEING GENERATED
    

    comboObj = mcObjects.comboObject()
    
    comboObj.true_psf_profiles = true_psf_profiles
    comboObj.shape_mes_algs = shear_ests
    comboObj.pixel_scales = pixel_scales
    comboObj.true_psf_fwhms = true_psf_fwhms
    comboObj.gal_psf_ratios = gal_psf_ratios
    comboObj.wrong_psf_fwhms = wrong_psf_fwhms

    import pdb; pdb.set_trace()
    comboObj.generate_combinations('all_gaussian_wrong_psfs_trueFWHM=0.7_ratio=0.5_miniscule_errors', 4)


def plot():
    # DESCRIPTION OF WHAT'S BEING PLOTTED
    obj = mcObjects.mcSummaryObject('pickles/all_gaussian_wrong_psfs_trueFWHM=0.7_ratio=0.5_miniscule_errors.pickle')
    # obj.slice('deconv_psf_fwhm', lambda fwhm: fwhm == 0.7)


    wrong_psf_fwhms = np.arange(0.68, 0.721, 0.001)
    true_fwhm = 0.7

    obj.slice('deconv_psf_fwhm', lambda f: abs(f - true_fwhm) < 0.0055)
    ylims, all_c1 = obj.plot_quadratic_m(color_column='deconv_psf_fwhm')
    obj.reset()

    if True:

        wrong_psf_fwhm_list = []
        c1s = []

        for i in range(len(wrong_psf_fwhms)):
            if abs(wrong_psf_fwhms[i] - true_fwhm) < 0.0055:
                obj.slice('deconv_psf_fwhm', value=wrong_psf_fwhms[i])
                # obj.slice('deconv_psf_fwhm', lambda dummy: True)
                diff = wrong_psf_fwhms[i] - true_fwhm

                combined_letters =  sorted(string.ascii_uppercase + string.ascii_lowercase)
                print(combined_letters[i])

                ylimits, c1 = obj.plot_quadratic_m(show=False, ylims=ylims) # plotname=f"{i}_fwhm_diff={diff:5f}")

                wrong_psf_fwhm_list.append(wrong_psf_fwhms[i])
                c1s.append(c1)
                # obj.plot_quadratic_m(color_column='deconv_psf_fwhm', show=False, ylims=ylims, plotname=str(i))
                obj.reset()

        fig, axs = plt.subplots(1, 1, figsize=(5, 5))

        print(len(wrong_psf_fwhm_list), len(c1s))
        print(wrong_psf_fwhm_list, c1s)
        x = np.asarray(wrong_psf_fwhm_list) - true_fwhm * np.ones(len(wrong_psf_fwhm_list))
        y = np.asarray(c1s)
        axs.scatter(x, y)

        slope, intercept = np.polyfit(x, y, 1)
        axs.plot(x, slope*x + intercept, label=f"slope = {slope} \n intercept = {intercept}")

        axs.set_xlabel(f'delta_fwhm')
        axs.set_ylabel(f'c (intercept of quadratic error')
        axs.set_title('c (intercept of quadratic error) vs. delta_fwhm')

        axs.axhline(color='k')
        axs.axvline(color='k')

        axs.legend()

        plt.show()

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


