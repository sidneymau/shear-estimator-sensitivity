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

    obj.plot_quadratic_m(color_column='gal_psf_ratio')
    
    # import pdb; pdb.set_trace()

def example_generation_and_plot():

    true_psf_profiles = {'gaussian': 0}
    shape_mes_algs = ['REGAUSS']
    pixel_scales = [0.2]
    oshear_dgs = [-0.01, 0.01]
    true_psf_fwhms = [1.0]
    gal_psf_ratios = [2.0]

    comboObject = mcObjects.comboObject()

    comboObject.true_psf_profiles = true_psf_profiles
    comboObject.shape_mes_algs = shape_mes_algs
    comboObject.pixel_scales = pixel_scales
    comboObject.oshear_dgs = oshear_dgs
    comboObject.true_psf_fwhms = true_psf_fwhms
    comboObject.gal_psf_ratios = gal_psf_ratios

    comboObject.generate_combinations('potato', 4)


def main():

    example_plot()

if __name__ == '__main__':
    main()