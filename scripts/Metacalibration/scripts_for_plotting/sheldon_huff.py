import mcObjects
import sys
import numpy as np


def generate():
    # DESCRIPTION OF WHAT'S BEING GENERATED

    true_psf_profiles = {'moffat': 3.5}
    shape_mes_algs = ['REGAUSS']
    pixel_scales = [0.2, 0.02]
    # oshear_dgs = [-0.01, 0.01]
    oshear_dgs = np.arange(-0.05, 0.06, 0.01)
    true_psf_fwhms = [0.9]
    # gal_psf_ratios = [2.0]
    offsets = [None]

    comboObject = mcObjects.comboObject()

    comboObject.true_psf_profiles = true_psf_profiles
    comboObject.shape_mes_algs = shape_mes_algs
    comboObject.pixel_scales = pixel_scales
    comboObject.oshear_dgs = oshear_dgs
    comboObject.true_psf_fwhms = true_psf_fwhms
    # comboObject.gal_psf_ratios = gal_psf_ratios
    comboObject.print_parameters()

    comboObject.generate_combinations('sheldon_huff', 4)


def plot():
    # DESCRIPTION OF WHAT'S BEING PLOTTED
    obj = mcObjects.mcSummaryObject('pickles/sheldon_huff(1).pickle')

    # looking at the failure points
    obj.slice('pixel_scale', lambda scale: scale == 0.2)
    # obj.slice_min_or_max('gal_psf_ratio', 'max')
    # obj.plot_quadratic_m(color_column='gal_fwhm')
    # obj.plot_absolute_error(color_column='gal_fwhm')

    obj.plot_row_images(97, axes=True)
    obj.plot_row_images(319, axes=True) 

    obj.slice('oshear_g1', lambda g1: abs(g1 - 0.0) < 0.001)
    obj.slice('oshear_g2', lambda g2: abs(g2 - 0.0) < 0.001)

    obj.plot_row_images(30, axes=True, plotname='no_shear_ratio2.0.png')
    obj.plot_row_images(0, axes=True, plotname='no_shear_ratio0.5.png')
    import pdb; pdb.set_trace()
    
    # obj.plot_quadratic_m(color_column='gal_psf_ratio')# , plotname='moffat_PSFs_beta=3.5_FWHM=0.9')



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


