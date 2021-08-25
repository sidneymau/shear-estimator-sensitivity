import mcObjects
import sys
import numpy as np


def generate():
    # DESCRIPTION OF WHAT'S BEING GENERATED

    # no change to galaxy profiles
    true_psf_profiles = {'gaussian': 0}
    shear_ests = ['REGAUSS']
    pixel_scales = [0.2]
    # calibration shears unchanged
    true_psf_fwhms = [0.7]
    gal_psf_ratios = [0.5, 2.0]
    wrong_psf_fwhms = None
    
    # number data points
    n = 10

    # offset amplification
    amp = 4

    offsets = [(amp * np.random.random_sample(), amp * np.random.random_sample()) for i in range(n)] 
    # offsets unchanged

    comboObj = mcObjects.comboObject()
    
    comboObj.true_psf_profiles = true_psf_profiles
    comboObj.shape_mes_algs = shear_ests
    comboObj.pixel_scales = pixel_scales
    comboObj.true_psf_fwhms = true_psf_fwhms
    comboObj.gal_psf_ratios = gal_psf_ratios
    comboObj.wrong_psf_fwhms = wrong_psf_fwhms
    comboObj.offsets = offsets

    import pdb; pdb.set_trace()
    comboObj.generate_combinations('gaussian_PSF_FWHM=0.7_more_ratios_varying_offsets_big_offset', 4)

def plot():
    # DESCRIPTION OF WHAT'S BEING PLOTTED

    obj = mcObjects.mcSummaryObject('pickles/gaussian_PSF_FWHM=0.7_more_ratios_varying_offsets_big_offset.pickle')
    obj.plot_quadratic_m(color_column='gal_psf_ratio')

    import pdb; pdb.set_trace()



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


