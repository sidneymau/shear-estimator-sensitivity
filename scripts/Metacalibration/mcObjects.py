"""
Goal: Create a framework to refactor the three scripts to make things easier
"""

from scripts.Metacalibration.metacal import generate_observed_galaxy
import numpy as np
import galsim
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
import sys
import metacal
from multiprocessing import Pool

class mcSummaryObject:
    """
    A wrapper for the master dataframe and all the slicing/plotting that goes into it
    """
    # Attributes
    # dataframe
    # pickle file name used to create

    # Initialization Methods

    def __init__(self, pickle_file):
        
        self.folder = 'plots/'
        self.fontsize = 14
        plt.rcParams.update({'font.size': self.fontsize})

        self.pickle_file = pickle_file
        self._results_array  = self.__unpickle__()
        self.__generate_df__()
        self.dropNan() # TODO put somewhere else?
        self.__element_columns__()
        self.__add_size_ratio__()
        self.original_df = self.df

    def __unpickle__(self):
        with open(self.pickle_file, 'rb') as f:
            results_array =  pickle.load(f)
            return results_array


    def __element_columns__(self):
        """
        Adds as columns the 4 individual elements of the shear response matrix

        Parameters:
            
            dataframe:      pandas dataframe    The master pandas dataframe with the galsim objects and parameters for each iteration of metacalibration

        Returns:

            dataframe:      pandas dataframe    The same dataframe but with four new columns, one for each of the elements of the shear response matrix R

        """
        for i in range(0, 2):
            for j in range(0, 2):
                self.df['R_' + str(i + 1) + str(j + 1)] = list(map(lambda r: r[i][j], self.df['R']))


    def __add_size_ratio__(self):

        # USE FWHM FOR BOTH
        self.df['gal_fwhm'] = [gal.fwhm for gal in self.df['original_gal']]
        self.df['psf_fwhm'] = [psf.fwhm for psf in self.df['true_psf']]
        self.df['gal_psf_ratio'] = self.df['gal_fwhm'] / self.df['psf_fwhm']
        

    def __generate_df__(self):

        # Loading the results table into a Pandas DataFrame
        column_names = ['original_gal', 'oshear_g1', 'oshear_g2', 'true_psf', 'deconv_psf', 'reconv_psf', 'shear_estimation_psf', 'cshear_dg1', 'cshear_dg2', 'shear_estimator', 'pixel_scale', 'R', 'reconvolved_noshear', 'reconvolved_noshear_e1', 'reconvolved_noshear_e2']
        self.df = pd.DataFrame(self._results_array, columns=column_names)



    # Modifier Methods

    def dropNan(self):
        self.df = self.df.dropna(axis=0, how='any')
        self.df = self.df.reset_index(drop=True)


    def slice(self, by, boolean_criterion):
        """
        by : what column to slice by
        values: the values of the column to keep
        """ 
        
        sliced_df = self.df

        criterion = sliced_df[by].map(boolean_criterion)
        sliced_df = sliced_df[criterion]

        self.df = sliced_df.reset_index(drop=True)
    

    def reset(self):
        self.df = self.original_df


    # Plotting Methods
    def save_fig_to_plots(self, figname):
        """
        Function for my own sanity, used for saving files to a specific directory
        without overwriting any old ones.

        Parameters:

            figname:        string      the desired name for the saved file (without the extension)

        Returns:

            VOID

        """
        # finding file version
        version = 1
        if not os.path.exists(self.folder + figname + '.png'):
            plt.savefig(self.folder + figname + '.png')

        else:
            while os.path.exists(self.folder + figname + '(' + str(version) + ').png'):
                version += 1
            plt.savefig(self.folder + figname + '(' + str(version) + ').png')


    def estimated_gi(self):
        """
        """


        R_inv_list = [np.linalg.inv(R) for R in self.df['R']]
        R_inv_array = np.asarray(R_inv_list)
        
        estimated_ellip_vec_list = []
        for i in range(len(self.df['R'])):
            e1 = self.df['reconvolved_noshear_e1'][i]
            e2 = self.df['reconvolved_noshear_e2'][i]
            estimated_ellip_vec_list.append(np.array([[e1],[e2]]))
        
        estimated_ellip_vec_array = np.asarray(estimated_ellip_vec_list)
    
        estimated_shear_array = R_inv_array @ estimated_ellip_vec_array

        estimated_g1 = estimated_shear_array[:,0, 0]
        estimated_g2 = estimated_shear_array[:,1, 0]

        # # Accounting for the fact that e ~ 2g --> g ~ e/2
        # estimated_g1 = estimated_e1 / 2
        # estimated_g2 = estimated_e2 / 2

        return estimated_g1, estimated_g2


    def plot_quadratic_m(self, color_column=None, plotname=None, blind=False):

        true_g1 = self.df['oshear_g1'].to_numpy()
        true_g2 = self.df['oshear_g2'].to_numpy()
        print(true_g1, true_g2)
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))

        axs[0].set_xlabel('true_g1')
        axs[0].set_ylabel('[(estimated_g1 - true_g1)/true_g1]' )
        axs[0].set_ylabel(r'$m = (\frac{{g_1}_{est} - {g_1}_{true}}{{g_1}_{true}})$')
        axs[0].set_title('g1')
        # axs[1].scatter(true_g2, estimated_g2 - true_g2)
        axs[1].set_xlabel('true_g2')
        axs[1].set_ylabel(r'$m = (\frac{{g_2}_{est} - {g_2}_{true}}{{g_2}_{true}})$')
        axs[1].set_title('g2')

        estimated_g1, estimated_g2 = self.estimated_gi()

        y1 = (estimated_g1 - true_g1)/true_g1
        y2 = (estimated_g2 - true_g2)/true_g2

        if color_column is not None:
            vmin = np.min(self.df[color_column])
            vmax = np.max(self.df[color_column])
            im0 = axs[0].scatter(true_g1, y1, c=self.df[color_column], cmap='cividis', vmin=vmin, vmax=vmax)
            im1 = axs[1].scatter(true_g2, y2, c=self.df[color_column], cmap='cividis', vmin=vmin, vmax=vmax)
            cbaxes0 = fig.add_axes([0.2, 0.05, 0.6, 0.01])
            # cbaxes1 = fig.add_axes([0.2, 0.95, 0.6, 0.01])
            cb0 = fig.colorbar(im0, ax=axs[:], orientation='horizontal', shrink=0.45, cax=cbaxes0)
            # cb1 = fig.colorbar(im1, ax=axs[:], orientation='horizontal', shrink=0.45, cax=cbaxes1)
            cb0.set_label(color_column)
            # cb1.set_label(color_column)
        
        else:
            im = axs[0].scatter(true_g1, y1, c='r', label='m')
            im = axs[1].scatter(true_g2, y2, c='r', label='m')

        plt.subplots_adjust(hspace=2.0, wspace=0.3)


        idx1 = np.nonzero(true_g1) 
        idx2 = np.nonzero(true_g2)

        a1, b1, c1 = np.polyfit(true_g1[idx1], y1[idx1], 2)
        a2, b2, c2 = np.polyfit(true_g2[idx2], y2[idx2], 2)


        x = np.linspace(-0.05, 0.05, 20)


        axs[0].plot(x, a1*x*x + b1*x + c1, c='k', label=f"a1 = {a1:.2f} \n b1 = {b1} \n c1 = {c1}", zorder=0)
        axs[1].plot(x, a2*x*x + b2*x + c2, c='k', label=f"a2 = {a2:.2f} \n b2 = {b2} \n c2 = {c2}", zorder=0)

        axs[0].legend()
        axs[1].legend()



        fig.suptitle(r'$m = (\frac{{g_i,}_{est} - {g_i,}_{true}}{{g_i,}_{true}})$ by element')

        if blind:
            for ax in axs:
                ax.tick_params(bottom=False, top=False, left=False, right=False,
                            labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        if plotname is not None:
            self.save_fig_to_plots(plotname)
        

        plt.show() 




    # plot quadratic_m
    # plot_R_elements
    # generate image


class comboObject:


    def __init__(self):

        self.folder = 'pickles'

        self.gal_profiles = {'gaussian': 0}
        self.true_psf_profiles = {'gaussian': 0, 'moffat': 3.5, 'moffat': 5}
        self.shape_mes_algs= ['REGAUSS', 'LINEAR', 'BJ'] #TODO figure out how to deal with KSB
        self.pixel_scales = [0.2, 0.02]
        self.gal_fluxes = [1.e5]
        self.oshear_dgs = [i for i in np.arange(-0.05, 0.06, 0.01) if not abs(i) < 0.001]
        self.cshear_dgs = [0.01]
        self.true_psf_fwhms = np.arange(0.5, 1.3, 0.1)
        self.gal_psf_ratios = np.arange(0.5, 2, 0.1)


    # TODO CHANGE THIS TO REFLECT ACTUAL WRONGNESS
    def make_wrong(self, true_psf):
        return true_psf


    def __create_combinations__(self):
        """
        all things we could possibly want to iterate over

        gal_flux
        cshear_dg
        gal_psf_ratio
        oshear_dg
        shape_mes_alg
        pixel_scale

        """

        combinations = []

        for gal_profile in self.gal_profiles.keys():
            for true_psf_profile in self.true_psf_profiles.keys():
                for shape_mes_alg in self.shape_mes_algs:
                    for pixel_scale in self.pixel_scales:
                        for gal_flux in self.gal_fluxes:
                            for oshear_dg in self.oshear_dgs:
                                for cshear_dg in self.cshear_dgs:
                                    for gal_psf_ratio in self.gal_psf_ratios:
                                        for true_psf_fwhm in self.true_psf_fwhms:

                                        
                                            dilation_factor = 1 / (1 - 2 * cshear_dg)

                                            if gal_profile == 'gaussian':
                                                original_gal = galsim.Gaussian(flux=gal_flux, fwhm=gal_psf_ratio * true_psf_fwhm)

                                            elif gal_profile == 'moffat':
                                                # don't forget beta
                                                original_gal = galsim.Moffat(flux=gal_flux, fwhm=gal_psf_ratio * true_psf_fwhm, beta=self.gal_profiles[gal_profile])
                                            
                                            else:
                                                raise Exception('Invalid galaxy profile!')
                                            

                                            if true_psf_profile == 'gaussian':
                                                true_psf = galsim.Gaussian(flux=1.0, fwhm=true_psf_fwhm)
            
                                            
                                            elif true_psf_profile == 'moffat':
                                                # don't forget beta
                                                true_psf = galsim.Moffat(flux=1.0, fwhm=true_psf_fwhm, beta=self.true_psf_profiles[true_psf_profile])
                                            
                                            else:
                                                raise Exception('Invalid PSF profile!')
                                            
                                            # making psf wrong
                                            deconv_psf = self.make_wrong(true_psf)

                                            # dilating reconv_psf
                                            reconv_psf = deconv_psf.dilate(dilation_factor)

                                            combinations.append((original_gal, oshear_dg, 0.0, true_psf, deconv_psf, reconv_psf, reconv_psf, cshear_dg, cshear_dg, shape_mes_alg, pixel_scale))
                                            combinations.append((original_gal, 0.0, oshear_dg, true_psf, deconv_psf, reconv_psf, reconv_psf, cshear_dg, cshear_dg, shape_mes_alg, pixel_scale))

        return combinations 


    def __pickle_dont_overwrite__(self, results, storage_file):

        name = storage_file

        if not os.path.exists(self.folder):
            os.mkdir(self.folder)

        name = f"{self.folder}/{storage_file}.pickle"
        if not os.path.exists(name):
            with open(name, 'wb') as f:
               pickle.dump(results, f) 
            return name
        
        else:
            version = 1
            while os.path.exists(f"{self.folder}/{storage_file}({version}).pickle"):
                version += 1
            
            with open(f"{self.folder}/{storage_file}({version}).pickle", 'wb') as f:
                pickle.dump(results, f)

            return f"{self.folder}/{storage_file}({version}).pickle"


    def generate_combinations(self, storage_file, num_workers):

        combinations = self.__create_combinations__()
        
        with Pool(num_workers) as p:
            results = p.starmap(metacal.metacalibration, combinations)

        filename = self.__pickle_dont_overwrite__(results, storage_file)
        
        print(f"Results stored to {filename}")




class metacalObject:

    def __init__(self, original_gal, oshear_g1, oshear_g2, true_psf,
                        deconv_psf, reconv_psf, shear_estimation_psf,
                        cshear_dg1, cshear_dg2, shear_estimator, pixel_scale):

        self.original_gal = original_gal
        self.oshear_g1 = oshear_g1
        self.oshear_g2 = oshear_g2
        self.true_psf = true_psf
        self.deconv_psf = deconv_psf
        self.reconv_psf = reconv_psf
        self.shear_estimation_psf = shear_estimation_psf
        self.cshear_dg1 = cshear_dg1
        self.cshear_dg2 = cshear_dg2
        self.shear_estimator = shear_estimator
        self.pixel_scale = pixel_scale


    def __generate_observed_galaxy__(self):
        """
        Returns:

            observed:      galsim object   The galaxy as would be seen through a telescope with no corrections
                                        (cosmic shear applied, PSF applied)

        """
        # shearing the original galaxy
        sheared = self.original_gal.shear(g1=self.oshear_g1, g2=self.oshear_g2)

        # Convolving the sheared galaxy with the PSF
        observed = galsim.Convolve(sheared, self.true_psf)

        self._observed_gal = observed


    def __delta_shear__(self):
        """
        Takes in an observed galaxy object, two PSFs for metacal (deconvolving
        and re-convolving), and the amount by which to shift g1 and g2, and returns
        a tuple of tuples of modified galaxy objects.
        ((g1plus, g1minus), (g2plus, g2minus))

        Parameters:

            observed_gal:   galsim object   The observed galaxy (cosmic shear and true_psf already applied)

            psf_deconvolve: galsim object   The PSF chosen for deconvolution in metacal (\Gamma 2)

            psf_reconvolve: galsim object   The reconvolution PSF (\Gamma 3) 

            delta_g1:       float           Calibration shear g1

            delta_g2:       float           Calibration shear g2


        Returns:

            g1_plus_minus:          tuple of galsim objects     (sheared with +dg1, sheared with -dg1)
            
            g2_plus_minus:          tuple of galsim objects     (sheared with +dg2, sheared with -dg2)

            reconvolved_noshear:    galsim_object               (unsheared, for accuracy tests) 

        """
        # Deconvolving by psf_deconvolve
        inv_psf = galsim.Deconvolve(self.deconv_psf)
        self._deconvolved = galsim.Convolve(self._observed_gal, inv_psf)

        # Applying second shear in g1
        self._sheared_plus_g1 = self._deconvolved.shear(g1=self.cshear_dg1, g2=0)
        self._sheared_minus_g1 = self._deconvolved.shear(g1= -self.cshear_dg1, g2=0)

        # Applying second shear in g2
        self._sheared_plus_g2 = self._deconvolved.shear(g1=0, g2=self.cshear_dg2)
        self._sheared_minus_g2 = self._deconvolved.shear(g1=0, g2= -self.cshear_dg2)

        # Reconvolving by psf_reconvolve for g1
        self._reconvolved_plus_g1 = galsim.Convolve(self._sheared_plus_g1, self.reconv_psf)
        self._reconvolved_minus_g1 = galsim.Convolve(self._sheared_minus_g1, self.reconv_psf)

        # Reconvolving by psf_reconvolve for g2
        self._reconvolved_plus_g2 = galsim.Convolve(self._sheared_plus_g2, self.reconv_psf)
        self._reconvolved_minus_g2 = galsim.Convolve(self._sheared_minus_g2, self.reconv_psf)

        self._reconvolved_noshear = galsim.Convolve(self._deconvolved, self.reconv_psf)


    def __shear_response__(self): 

        """
        Returns:

            R:              2D numpy array      The calculated shear response matrix 

            noshear_e1:     float               The measured shape (distortion, first component) of the galaxy to which no calibration shear was applied

            noshear_e2:     float               The measured shape (distortion, second component) of the galaxy to which no calibration shear was applied

        """

        plus_g1_gal = g1_plus_minus[0]
        minus_g1_gal = g1_plus_minus[1]
        plus_g2_gal = g2_plus_minus[0]
        minus_g2_gal = g2_plus_minus[1]

        # Measuring galaxy shape parameters
        # We want to measure the shapes of reconvolved_plus_galaxy and reconvolved_minus_galaxy
        # the documentation recommends that we use the method='no_pixel' on the images

        plus_g1 = self._reconvolved_plus_g1.drawImage(scale=self.pixel_scale, method='no_pixel')
        minus_g1 = self._reconvolved_minus_g1.drawImage(scale=self.pixel_scale, method='no_pixel')

        plus_g2 = self._reconvolved_plus_g2.drawImage(scale=self.pixel_scale, method='no_pixel')
        minus_g2 = self._reconvolved_minus_g2.drawImage(scale=self.pixel_scale, method='no_pixel')

        psf_shearestimator_image = self.shear_estimation_psf.drawImage(scale=self.pixel_scale)

        plus_moments_g1 = galsim.hsm.EstimateShear(plus_g1, psf_shearestimator_image, shear_est=self.shear_estimator)
        minus_moments_g1 = galsim.hsm.EstimateShear(minus_g1, psf_shearestimator_image, shear_est=self.shear_estimator)
        plus_moments_g2 = galsim.hsm.EstimateShear(plus_g2, psf_shearestimator_image, shear_est=self.shear_estimator)
        minus_moments_g2 = galsim.hsm.EstimateShear(minus_g2, psf_shearestimator_image, shear_est=self.shear_estimator)


        e1_plus_g1 = plus_moments_g1.corrected_e1
        e2_plus_g1 = plus_moments_g1.corrected_e2

        e1_minus_g1 = minus_moments_g1.corrected_e1
        e2_minus_g1 = minus_moments_g1.corrected_e2

        e1_plus_g2 = plus_moments_g2.corrected_e1
        e2_plus_g2 = plus_moments_g2.corrected_e2

        e1_minus_g2 = minus_moments_g2.corrected_e1
        e2_minus_g2 = minus_moments_g2.corrected_e2

        # calculating the shear response matrix R
        R_11 = (e1_plus_g1 - e1_minus_g1) / (2 * self.cshear_dg1)
        R_12 = (e2_plus_g1 - e2_minus_g1) / (2 * self.cshear_dg1)
        R_21 = (e1_plus_g2 - e1_minus_g2) / (2 * self.cshear_dg2)
        R_22 = (e2_plus_g2 - e2_minus_g2) / (2 * self.cshear_dg2)

        R = np.array([[R_11, R_12],[R_21, R_22]])

        # Calculating shape of reconvolved_no_shear to test accuracy of shear response
        noshear_image = self._reconvolved_noshear.drawImage(scale=self.pixel_scale, method='no_pixel')
        noshear_moments = galsim.hsm.EstimateShear(noshear_image, psf_shearestimator_image, shear_est=self.shear_estimator)
        noshear_e1 = noshear_moments.corrected_e1
        noshear_e2 = noshear_moments.corrected_e2

        self.R = R
        self.noshear_e1 = noshear_e1
        self.noshear_e2 = noshear_e2 

    def metacalibration(self):
        self.__generate_observed_galaxy__()
        self.__delta_shear__()
        self.__shear_response__()

        return self

        return (self.original_gal,
                self.oshear_g1,
                self.oshear_g2,
                self.true_psf,
                self.deconv_psf,
                self.reconv_psf,
                self.shear_estimation_psf,
                self.cshear_dg1,
                self.cshear_dg2,
                self.shear_estimator,
                self.pixel_scale,
                self.R,
                self._reconvolved_noshear,
                self.noshear_e1,
                self.noshear_e2)