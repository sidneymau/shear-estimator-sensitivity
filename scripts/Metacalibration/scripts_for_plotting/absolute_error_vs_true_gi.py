import mcObjects
import sys
import matplotlib.pyplot as plt


def generate():
    # DESCRIPTION OF WHAT'S BEING GENERATED
    pass

def plot():
    # DESCRIPTION OF WHAT'S BEING PLOTTED

    ratio_half = mcObjects.mcSummaryObject('pickles/all_gaussian_wrong_psfs_trueFWHM=0.7_ratio=0.5_scale=0.2_miniscule_errors.pickle')
    ratio_one  = mcObjects.mcSummaryObject('pickles/all_gaussian_wrong_psfs_trueFWHM=0.7_ratio=1.0_scale=0.2_miniscule_errors.pickle')
    ratio_two  = mcObjects.mcSummaryObject('pickles/all_gaussian_wrong_psfs_trueFWHM=0.7_ratio=2.0_scale=0.2_miniscule_errors.pickle')

    ratio_half.plot_absolute_error(color_column='deconv_psf_fwhm', show=False, plotname='wrong_psfs_FWHM=0.7_ratio=0.5_scale=0.2_abs_error')
    ratio_one.plot_absolute_error(color_column='deconv_psf_fwhm', show=False, plotname='wrong_psfs_FWHM=0.7_ratio=1.0_scale=0.2_abs_error')
    ratio_two.plot_absolute_error(color_column='deconv_psf_fwhm', show=False, plotname='wrong_psfs_FWHM=0.7_ratio=2.0_scale=0.2_abs_error')

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


