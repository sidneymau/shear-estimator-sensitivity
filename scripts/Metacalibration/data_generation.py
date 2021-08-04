"""
Using the metacalibration function, this script loops over a variety of
different parameters to measure the response of the shear response matrix.

It pickles a list of tuples with the galsim objects and relevant parameters to disk.

Step 1. Generate a combination of galaxies to feed into the metacalibration() function from metacal.py
Step 2. Use multiprocessing/Pool to feed the list of combinations into the metacalibration function and generate a list of tuples
Step 3. pickle the resulting list of tuples to disk for data processing/plotting in the metacal_plotting.py script

"""
import galsim
import numpy as np
import metacal
import pickle
import sys
from multiprocessing import Pool
import pandas as pd
import matplotlib.pyplot as plt
import os.path


# START COMBINATION-GENERATION FUNCTIONS
def all_gaussian_combinations():
    """

    Generates combinations with:

    - all gaussian source galaxies
    - all gaussian PSFs
    - NO INITIAL COSMIC SHEAR
    - Calibration shear magnitude of 0.01
    - constant dilation factor of 1.2 * size of deconv PSF
    - pixel scale of 0.2
    - 'REGAUSS' shape measurement algorithm.
   
    """

    gal_flux = 1.e5
    dg = [0.01]
    dilation_factor = 1.2

    gal_psf_ratios = np.arange(0.5, 2.1, 0.1)
    true_psf_sigmas = 1/2.355 * np.arange(0.5, 1.3, 0.1)  # 0.5 - 1.3 arcseconds fwhm converted to sigma
    reconv_psf_sigmas = dilation_factor * true_psf_sigmas

    combinations = []
    for ratio in gal_psf_ratios:        # gal_sigma / psf_sigma should be equal to ratio
        for i in range(len(true_psf_sigmas)):
            for delta_g in dg:
                original_gal = galsim.Gaussian(flux=gal_flux, sigma=true_psf_sigmas[i] * ratio)
                true_psf = galsim.Gaussian(flux=gal_flux, sigma=true_psf_sigmas[i])
                deconv_psf = true_psf
                reconv_psf = galsim.Gaussian(flux=gal_flux, sigma=reconv_psf_sigmas[i])

                combinations.append((original_gal, 0.0, 0.0, true_psf, deconv_psf, reconv_psf, reconv_psf, delta_g, delta_g, 'REGAUSS', 0.2))

    return combinations


def moffat_psf_combinations():
    """

    Generates combinations with:

    - all gaussian source galaxies
    - all Moffat PSFs
    - NO INITIAL COSMIC SHEAR
    - Calibration shear magnitude of 0.01
    - constant dilation factor of 1.2 * size of deconv PSF
    - pixel scale of 0.2
    - 'REGAUSS' shape measurement algorithm.

    """

    gal_flux = 1.e5
    dg = [0.01]
    dilation_factor = 1.2
    psf_beta = 5.

    gal_psf_ratios = np.arange(0.5, 2.1, 0.1)
    true_psf_fwhm =  np.arange(0.5, 1.3, 0.1)
    reconv_psf_fwhm = dilation_factor * true_psf_fwhm

    combinations = []
    for ratio in gal_psf_ratios:        # gal_sigma / psf_sigma should be equal to ratio
        for i in range(len(true_psf_fwhm)):
            for delta_g in dg:
                original_gal = galsim.Gaussian(flux=gal_flux, fwhm=true_psf_fwhm[i] * ratio)
                true_psf = galsim.Moffat(flux=gal_flux, beta=psf_beta, fwhm=true_psf_fwhm[i])
                deconv_psf = true_psf
                reconv_psf = galsim.Moffat(flux=gal_flux, beta=psf_beta, fwhm=reconv_psf_fwhm[i])

                combinations.append((original_gal, 0.0, 0.0, true_psf, deconv_psf, reconv_psf, reconv_psf, delta_g, delta_g, 'REGAUSS', 0.2))

    return combinations


def response_accuracy_test_0(pixel_scale):
    """

    Generates combinations with:

    - all Gaussian source galaxies
    - all Gaussian PSFs
    - initial cosmic shear from -0.05 to 0.05 for g1 and g2, one at a time
    - Calibration shear magnitude of 0.01 or 0.05
    - variable dilation factor (from Huff & Mandelbaum 2017)
    - variable pixel_scale (parameter)
    - 'REGAUSS' shape measurement algorithm.

    takes ~ 30 seconds for pixel_scale = 0.2
    takes ~ 5 minutes for pixel_scale = 0.02
    """
    # going back to all-Gaussian galaxies

    gal_flux = 1.e5
    # cshear_dg = [0.05]
    cshear_dg = [0.01]


    gal_psf_ratios = np.arange(0.5, 2.1, 0.1)
    true_psf_sigmas = 1/2.355 * np.arange(0.5, 1.3, 0.1)  # 0.5 - 1.3 arcseconds fwhm converted to sigma
    

    combinations = []

    # oshear_dgs = np.arange(-0.05, 0.06, 0.01)
    oshear_dgs = [i for i in np.arange(-0.05, 0.06, 0.01) if not abs(i) < 0.001]

    for ratio in gal_psf_ratios:        # gal_sigma / psf_sigma should be equal to ratio
        for i in range(len(true_psf_sigmas)):
            for cshear_delta_g in cshear_dg:
                for oshear_dg in oshear_dgs: #

                    dilation_factor = dilation_factor = 1 / (1 - 2 * cshear_delta_g)

                    original_gal = galsim.Gaussian(flux=gal_flux, sigma=true_psf_sigmas[i] * ratio)
                    true_psf = galsim.Gaussian(flux=gal_flux, sigma=true_psf_sigmas[i])
                    deconv_psf = true_psf
                    reconv_psf = galsim.Gaussian(flux=gal_flux, sigma=dilation_factor * true_psf_sigmas[i])
                    
                    combinations.append((original_gal, oshear_dg, 0.0, true_psf, deconv_psf, reconv_psf, reconv_psf, cshear_delta_g, cshear_delta_g, 'REGAUSS', pixel_scale))
                    combinations.append((original_gal, 0.0, oshear_dg, true_psf, deconv_psf, reconv_psf, reconv_psf, cshear_delta_g, cshear_delta_g, 'REGAUSS', pixel_scale))

    return combinations


def response_accuracy_test_better():
    """

    Generates combinations with:

    - all Gaussian source galaxies
    - all Gaussian PSFs
    - initial cosmic shear from -0.05 to 0.05 for g1 and g2, one at a time
    - Calibration shear magnitudes from 0.01 to 0.10
    - variable dilation factor (from Huff & Mandelbaum 2017)
    - variable pixel_scale (0.2 and 0.02))
    - 'REGAUSS' shape measurement algorithm.

    runtime: ~ 1 hour, needs to be done on Sherlock

    """

    gal_flux = 1.e5
    cshear_dg = np.arange(0.01, 0.11, 0.01)

    gal_psf_ratios = np.arange(0.5, 2.1, 0.1)
    true_psf_sigmas = 1/2.355 * np.arange(0.5, 1.3, 0.1)  # 0.5 - 1.3 arcseconds fwhm converted to sigma

    combinations = []

    # oshear_dgs = np.arange(0.01, 0.11, 0.01)
    oshear_dgs = [i for i in np.arange(-0.05, 0.06, 0.01) if not abs(i) < 0.001]

    for ratio in gal_psf_ratios:        # gal_sigma / psf_sigma should be equal to ratio
        for i in range(len(true_psf_sigmas)):
            for cshear_delta_g in cshear_dg:
                for oshear_dg in oshear_dgs:
                    for pixel_scale in [0.2, 0.02]:

                        dilation_factor = 1 / (1 - 2 * cshear_delta_g)
                    
                        original_gal = galsim.Gaussian(flux=gal_flux, sigma=true_psf_sigmas[i] * ratio)
                        true_psf = galsim.Gaussian(flux=gal_flux, sigma=true_psf_sigmas[i])
                        deconv_psf = true_psf
                        reconv_psf = galsim.Gaussian(flux=gal_flux, sigma=dilation_factor*true_psf_sigmas[i])

                        combinations.append((original_gal, oshear_dg, 0.0, true_psf, deconv_psf, reconv_psf, reconv_psf, cshear_delta_g, cshear_delta_g, 'REGAUSS', pixel_scale))
                        combinations.append((original_gal, 0.0, oshear_dg, true_psf, deconv_psf, reconv_psf, reconv_psf, cshear_delta_g, cshear_delta_g, 'REGAUSS', pixel_scale))

    return combinations



def generate_combinations():
    """
    Generates a list of different combinations of
    (observed galaxy, deconv_psf, reconv_psf, delta_g, delta_g)

    to run metacalibration over. Feeds into vary_parameters()
    """

    # TODO will eventually need to change this function to vary the observed galaxy used as well

    # Loop over one observed galaxy only (to test)
    gal_flux = 1.e5
    gal_sigma = 2.
    gal = galsim.Gaussian(flux=gal_flux, sigma=gal_sigma,)

    # initial shear TODO change this to use an "intrinsic ellipticity" parameter on generation instead of manually shearing?
    dg1 = 0.00
    dg2 = 0.00

    psf1 = galsim.Gaussian(flux=1., sigma=gal_sigma)

    observed_galaxy = metacal.generate_observed_galaxy(gal, psf1, dg1, dg2)

    # Creating lists of different parameters to loop through
    psf_beta = 5.

    # Deconvolution PSF type and size variations
    deconv_Gaussian_size_variation = [galsim.Gaussian(flux=1., sigma=sig) for sig in 1/2.355 * np.arange(0.5, 1.3, 0.1)]
    deconv_Moffat_size_variation = [galsim.Moffat(beta=psf_beta, flux=1., half_light_radius=r0) for r0 in
                                    np.arange(0.8, 2.0, 0.2)]

    # Reconvolution PSF type and size variations
    reconv_Gaussian_size_variation = [galsim.Gaussian(flux=1., sigma=sig) for sig in 1/2.355 * np.arange(0.5, 1.3, 0.1)]
    reconv_Moffat_size_variation = [galsim.Moffat(beta=psf_beta, flux=1., half_light_radius=r0) for r0 in
                                    np.arange(0.8, 2.0, 0.2)]

    # different sized calibration shears
    # dg = np.arange(0.01, 0.11, 0.01)
    dg = [0.01]  # same as Sheldon and Huff Value

    # Creating long master list of all combinations to loop through
    combination_list = []
    for deconv_psf in deconv_Gaussian_size_variation + deconv_Moffat_size_variation:
        for reconv_psf in reconv_Gaussian_size_variation + reconv_Moffat_size_variation:
            for delta_g in dg:
                combination_list.append((observed_galaxy, deconv_psf, reconv_psf, delta_g, delta_g))

    return combination_list

# END COMBINATION-GENERATION FUNCTIONS
def vary_parameters(combo_list, storage_file):
    """

    This function is called in main after the desired combinations are generated.
    Uses multiprocessing to feed combinations into metacalibration and generate
    a list of tuples of results, which is then dumped into a .pickle file

    """

    # Using multiprocessing to generate shear response matrices for all combinations
    with Pool(4) as p:
        results = p.starmap(metacal.metacalibration, combo_list)

    # Storing metacalibration results to disk
    with open(storage_file, 'wb') as f:
        pickle.dump(results, f)

    print("\n" * 4)
    print(f"Result stored to {storage_file}")
    print("\n" * 4)


def main():

    args = sys.argv[1:]


    if args[0] == '-generate':

        # parse filename for .pickle file from cmd line
        filename_to_create = args[1]

        # Generates combinations using the one function not commented out

        # combinations = generate_combinations()
        # combinations = sanity_check1()
        # combinations = all_gaussian_combinations()
        # combinations = moffat_psf_combinations()
        combinations = response_accuracy_test_0(0.2)
        # combinations = response_accuracy_test_better()


        # Stuff for my sanity
        # naming the pickle file according to cmd line argument
        # making sure pickle files aren't overwritten

        if not os.path.exists('pickles'):
            vary_parameters(combinations, filename_to_create + '.pickle')
            return 0

        if not os.path.exists('pickles/' + filename_to_create + '.pickle'):
            vary_parameters(combinations, 'pickles/' + filename_to_create + '.pickle')

        else:
            version = 1
            while os.path.exists('pickles/' + filename_to_create + '(' + str(version) + ').pickle'):
                version += 1

            vary_parameters(combinations, 'pickles/' + filename_to_create + '(' + str(version) + ').pickle')

        return 0

    if args[0] == '-sherlock':

        combinations = response_accuracy_test_better()

        vary_parameters(combinations, 'data.pickle')

    ## Old Code ##



    # if args[0] == '-filter':
    #     pd_table = generate_df(stored_results)
    #     pd_table.to_csv('table.csv')
    #     # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     #     print(pd_table)

    # return 0


if __name__ == '__main__':
    main()
