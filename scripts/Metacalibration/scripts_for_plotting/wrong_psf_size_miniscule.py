import mcObjects
import sys
import galsim
import numpy as np

def generate():
    # DESCRIPTION OF WHAT'S BEING GENERATED
    

    # no change to galaxy profiles
    true_psf_profiles = {'gaussian': 0}
    shear_ests = ['REGAUSS']
    pixel_scales = [0.2]
    # calibration shears unchanged
    true_psf_fwhms = [0.7]
    gal_psf_ratios = [2.0]
    # offsets unchanged
    wrong_psf_fwhms = np.arange(0.68, 0.721, 0.001)
    # wrong_psf_fwhms = [0.7]

    comboObj = mcObjects.comboObject()
    
    comboObj.true_psf_profiles = true_psf_profiles
    comboObj.shape_mes_algs = shear_ests
    comboObj.pixel_scales = pixel_scales
    comboObj.true_psf_fwhms = true_psf_fwhms
    comboObj.gal_psf_ratios = gal_psf_ratios
    comboObj.wrong_psf_fwhms = wrong_psf_fwhms

    import pdb; pdb.set_trace()
    comboObj.generate_combinations('all_gaussian_wrong_psfs_trueFWHM=0.7_miniscule_errors', 4)


def plot():
    # DESCRIPTION OF WHAT'S BEING PLOTTED
    obj = mcObjects.mcSummaryObject('pickles/all_gaussian_wrong_psfs_trueFWHM=0.7_miniscule_errors.pickle')
    # obj.slice('deconv_psf_fwhm', lambda fwhm: fwhm == 0.7)

    wrong_psf_fwhms = np.arange(0.68, 0.721, 0.001)
    true_fwhm = 0.7

    for i in range(len(wrong_psf_fwhms)):
        # obj.slice('deconv_psf_fwhm', lambda dummy: abs(wrong_psf_fwhms[i] - true_fwhm) < 0.005)
        obj.slice('deconv_psf_fwhm', lambda dummy: True)
        obj.plot_quadratic_m(color_column='deconv_psf_fwhm', show=False, plotname=f"fwhm_diff={wrong_psf_fwhms[i] - true_fwhm}")
        obj.reset()


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


