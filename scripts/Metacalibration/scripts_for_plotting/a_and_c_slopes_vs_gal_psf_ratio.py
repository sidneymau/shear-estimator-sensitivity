import mcObjects
import sys
import numpy as np
import matplotlib.pyplot as plt
import pdb

# no change to galaxy profiles
true_psf_profiles = {'gaussian': 0}
shear_ests = ['REGAUSS']
pixel_scales = [0.2]
# calibration shears unchanged
true_psf_fwhms = [0.7]

# WIDE RANGE OF GAL_PSF_RATIOS
# gal_psf_ratios = [1.0]
gal_psf_ratios = np.arange(0.5, 5.1, 0.5)


# offsets unchanged
wrong_psf_fwhms = np.arange(true_psf_fwhms[0] - 0.005, true_psf_fwhms[0] + 0.006, 0.001)
#wrong_psf_fwhms = np.arange(0.68, 0.721, 0.001)
# wrong_psf_fwhms = [0.7]

def generate():
    # DESCRIPTION OF WHAT'S BEING GENERATED

    all_ratios = mcObjects.comboObject()

    all_ratios.true_psf_profiles = true_psf_profiles
    all_ratios.shape_mes_algs = shear_ests
    all_ratios.pixel_scales = pixel_scales
    all_ratios.true_psf_fwhms = true_psf_fwhms
    all_ratios.gal_psf_ratios = gal_psf_ratios
    all_ratios.wrong_psf_fwhms = wrong_psf_fwhms


    import pdb; pdb.set_trace()

    all_ratios.generate_combinations('gaussian_wrong_psfs_FWHM=0.7_scale=0.2_all_ratios_deltaFWHM_pm0.005', 4)



    pass

def plot():
    # DESCRIPTION OF WHAT'S BEING PLOTTED

    # What we want:
    # slopes of the plots of c for different gal_psf_ratios

    obj = mcObjects.mcSummaryObject('pickles/gaussian_wrong_psfs_FWHM=0.7_scale=0.2_all_ratios_deltaFWHM_pm0.005.pickle')


    def get_c(obj, dataframe):

        a1, b1, c1 = obj.get_fit_parameters(dataframe)

        return c1 


    fig, axs = plt.subplots(1, 2, figsize=(16, 7)) 


    ratio_tuple = {}

    for ratio in gal_psf_ratios:
        sliced_by_gal = obj.slice('gal_psf_ratio', value=ratio, table_in=obj.df)
        c_list = []
        for wrong_fwhm in wrong_psf_fwhms:
            subset = obj.slice('deconv_psf_fwhm', value=wrong_fwhm, table_in=sliced_by_gal)
            c = get_c(obj, subset)
            
            c_list.append(c)


        diff = np.asarray(wrong_psf_fwhms) - 0.7 * np.ones(len(wrong_psf_fwhms))

        # normalizing diff to make everything dimensionless
        norm_diff = diff / 0.7

        slope, intercept = np.polyfit(norm_diff, c_list, 1)

        axs[0].scatter(norm_diff, c_list)
        axs[0].plot(norm_diff, slope * norm_diff + intercept, label=f"ratio = {ratio}")
        ratio_tuple[ratio] = (slope, intercept)


    ratios = np.asarray(list(ratio_tuple.keys()))
    slopes = np.asarray(list(map(lambda tup: tup[0], ratio_tuple.values())))

    print(ratios)
    print(slopes)


    axs[1].scatter(ratios, slopes)

    axs[0].legend(title=r"$\frac{FWHM_{gal}}{FWHM_{true\;PSF}}$")

    axs[0].set_xlabel('delta_FWHM / FWHM')
    axs[0].set_ylabel('c (intercept of quadratic fit for m)')

    axs[1].set_xlabel('gal_psf_ratio')
    axs[1].set_xticks(ratios)
    axs[1].set_ylabel('rate of change of c with fractional error on PSF size')

    axs[0].set_title('intercept of quadratic fit for m as a function of delta_FWHM / FWHM')
    axs[1].set_title('rate of change of c with fractional error on PSF size as a function of gal_psf_ratio')

    plt.show()

    # pdb.set_trace()











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


