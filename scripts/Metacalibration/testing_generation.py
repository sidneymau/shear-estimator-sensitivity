import mcObjects

def main():

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

    import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()