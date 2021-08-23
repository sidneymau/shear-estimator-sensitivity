from matplotlib.pyplot import plot
import mcObjects


def example_plot():

    # Using a pickle file that I have already created
    obj = mcObjects.mcSummaryObject('pickles/gaussian_data.pickle')

    # slicing for pixel_scale == 0.2 only
    obj.slice('pixel_scale', lambda scale: scale == 0.2)

    # slicing for shear_estimator == 'REGAUSS' only
    obj.slice('shear_estimator', lambda shear_est: shear_est == 'REGAUSS')

    # slicing to only get rows where the calibration shear in either g1 or g2 is 0.01
    obj.slice('cshear_dg1', lambda dg: dg == 0.01 or dg == 0.0)
    obj.slice('cshear_dg2', lambda dg: dg == 0.01 or dg == 0.0)

    obj.change_fontsize(12)
    # obj.plot_quadratic_m_simplified(color_column='gal_psf_ratio', axis_equal=True, plotname='all_gaussian_scale=0.2_REGAUSS_allratios_comparing_wwo_metacal_simplified_EQUAL_Y_SCALE')
    obj.plot_absolute_error(color_column='gal_psf_ratio') #, plotname='all_gaussian_scale=0.2_REGAUSS_allratios_comparing_wwo_metacal_absolute_error')

    # import pdb; pdb.set_trace()


def comparing():

    # Using a pickle file that I have already created
    obj = mcObjects.mcSummaryObject('pickles/gaussian_data.pickle')

    # slicing for pixel_scale == 0.2 only
    obj.slice('pixel_scale', lambda scale: scale == 0.2)

    # slicing for shear_estimator == 'REGAUSS' only
    obj.slice('shear_estimator', lambda shear_est: shear_est == 'REGAUSS')

    # slicing to only get rows where the calibration shear in either g1 or g2 is 0.01
    obj.slice('cshear_dg1', lambda dg: dg == 0.01 or dg == 0.0)
    obj.slice('cshear_dg2', lambda dg: dg == 0.01 or dg == 0.0)

    obj.slice('gal_psf_ratio', lambda ratio: abs(ratio - 2.0) < 0.001)

    obj.change_fontsize(10)
    obj.plot_quadratic_m(color_column='gal_fwhm', plotname='all_gaussian_scale=0.2_REGAUSS_biggestratios_comparing_wwo_metacal')
    
    # import pdb; pdb.set_trace()



def replicate_Sheldon_Huff():

    true_psf_profiles = {'moffat': 3.5}
    shape_mes_algs = ['REGAUSS']
    pixel_scales = [0.2, 0.02]
    # oshear_dgs = [-0.01, 0.01]
    true_psf_fwhms = [0.9]
    # gal_psf_ratios = [2.0]

    comboObject = mcObjects.comboObject()

    comboObject.true_psf_profiles = true_psf_profiles
    comboObject.shape_mes_algs = shape_mes_algs
    comboObject.pixel_scales = pixel_scales
    # comboObject.oshear_dgs = oshear_dgs
    comboObject.true_psf_fwhms = true_psf_fwhms
    # comboObject.gal_psf_ratios = gal_psf_ratios
    comboObject.print_parameters()

    import pdb; pdb.set_trace()

    comboObject.generate_combinations('potato', 4)

def display_Sheldon_Huff():

    obj = mcObjects.mcSummaryObject('pickles/potato.pickle')

    # looking at the failure points
    obj.slice_min_or_max('gal_psf_ratio', 'min')
    obj.plot_quadratic_m(color_column='gal_fwhm')
    obj.plot_absolute_error(color_column='gal_fwhm')


    import pdb; pdb.set_trace()
    
    obj.plot_quadratic_m(color_column='gal_psf_ratio')# , plotname='moffat_PSFs_beta=3.5_FWHM=0.9')


def generate_offset_Sheldon_Huff():
    
    true_psf_profiles = {'moffat': 3.5}
    shape_mes_algs = ['REGAUSS']
    pixel_scales = [0.2, 0.02]
    # oshear_dgs = [-0.01, 0.01]
    true_psf_fwhms = [0.9]
    # gal_psf_ratios = [2.0]
    

    comboObject = mcObjects.comboObject()

    comboObject.true_psf_profiles = true_psf_profiles
    comboObject.shape_mes_algs = shape_mes_algs
    comboObject.pixel_scales = pixel_scales
    # comboObject.oshear_dgs = oshear_dgs
    comboObject.true_psf_fwhms = true_psf_fwhms
    # comboObject.gal_psf_ratios = gal_psf_ratios

    comboObject.offset = (0.5, 0.5)


    comboObject.print_parameters()

    import pdb; pdb.set_trace()

    comboObject.generate_combinations('potato_offset', 4)


def display_offset_Sheldon_Huff():
    obj = mcObjects.mcSummaryObject('pickles/potato_offset.pickle')
    obj.plot_quadratic_m(color_column='gal_psf_ratio')# , plotname='moffat_PSFs_beta=3.5_FWHM=0.9_offset')


def main():

    # generate_offset_Sheldon_Huff()
    # display_offset_Sheldon_Huff()
    display_Sheldon_Huff()
    # replicate_Sheldon_Huff()
    # example_plot()
    # comparing()

if __name__ == '__main__':
    main()
