"""
Goal: Create a framework to refactor the three scripts to make things easier
"""

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