"""
Goal: Create a framework to refactor the three scripts to make things easier
"""

from re import I
import numpy as np
import galsim
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool

class mcSummaryObject:
    """
    A wrapper for the master dataframe and all the slicing/plotting that goes into it
    """

    # Initialization Methods

    def __init__(self, pickle_file, dropNan=True):
        """
        Initialize an mcSummaryObject. 
        
        Parameters:

            pickle_file:    string        The .pickle file from which to create the object

            dropNan:        bool          whether or not to drop lines with Nan from the initial table
        
        """ 
        # the folder to which plots will be saved
        self.folder = 'plots/'

        # the default fontsize for generated plots
        self.fontsize = 10
        plt.rcParams.update({'font.size': self.fontsize})

        # the pickle file
        self.pickle_file = pickle_file

        # list of tuples containing the results 
        self._results_array  = self._unpickle()

        # generating the main dataframe
        self._generate_df()
        
        if dropNan:
            self.dropNan() # TODO put somewhere else?
            self._element_columns()

        # post-adding this column for convenience later
        self._add_size_ratio()

        # original_df should never be modified
        self.original_df = self.df


    def _unpickle(self):
        """
        Unpickles the pickle file and loads it into a list of tuples. Used by __init__ only

        Returns:

            List of tuples of results from iterations run through a comboObj

        """

        with open(self.pickle_file, 'rb') as f:
            results_array =  pickle.load(f)
            return results_array


    def _element_columns(self):
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


    def _add_size_ratio(self):
        """
        
        Adds the columns 'gal_fwhm', 'true_psf_fwhm', 'gal_psf_ratio', and 'deconv_psf_fwhm' to the master dataframe.
        
        """
        # USE FWHM FOR BOTH
        self.df['gal_fwhm'] = [gal.fwhm for gal in self.df['original_gal']]
        self.df['true_psf_fwhm'] = [psf.fwhm for psf in self.df['true_psf']]
        self.df['gal_psf_ratio'] = self.df['gal_fwhm'] / self.df['true_psf_fwhm']

        self.df['deconv_psf_fwhm'] = [obj.deconv_psf.fwhm for obj in self.df['mcObject']]
        

    def _generate_df(self):
        
        """
        Creates the .df attribute of the mcSummaryObject from the ._results_array attribute        
        """

        def expand_object(mcObject):
            """
            Helper function to expand _results_array into the necessary shape to create a dataframe from
            """

            return (mcObject, mcObject.original_gal, mcObject.oshear_g1, mcObject.oshear_g2,
                    mcObject.true_psf, mcObject.deconv_psf, mcObject.reconv_psf, mcObject.shear_estimation_psf,
                    mcObject.cshear_dg1, mcObject.cshear_dg2, mcObject.shear_estimator,
                    mcObject.pixel_scale, mcObject.R, mcObject.reconvolved_noshear,
                    mcObject.noshear_e1, mcObject.noshear_e2)

        # make a dataframe-shaped list from the results_array

        dfshape_array = [expand_object(obj) for obj in self._results_array]


        # Loading the results table into a Pandas DataFrame
        column_names = ['mcObject', 'original_gal', 'oshear_g1', 'oshear_g2', 'true_psf', 'deconv_psf', 'reconv_psf', 'shear_estimation_psf', 'cshear_dg1', 'cshear_dg2', 'shear_estimator', 'pixel_scale', 'R', 'reconvolved_noshear', 'reconvolved_noshear_e1', 'reconvolved_noshear_e2']
        # self.df = pd.DataFrame(self._results_array, columns=column_names)
        self.df = pd.DataFrame(dfshape_array, columns=column_names)



    # Modifier Methods

    def dropNan(self):
        """

        Drops all rows with any NaN values in the entire table, and resets the table indexes as well.

        """

        self.df = self.df.dropna(axis=0, how='any')
        self.df = self.df.reset_index(drop=True)


    def slice(self, by, boolean_criterion=None, table_in=None, value=None):
        """

        by : what column to slice by
        values: the values of the column to keep
        """ 
        

        if table_in is None:
            sliced_df = self.df
        else:
            sliced_df = table_in

        if boolean_criterion is not None:
            criterion = sliced_df[by].map(boolean_criterion)
        
        else:
            criterion = sliced_df[by].map(lambda x: abs(x - value) < 0.0001)

        sliced_df = sliced_df[criterion]

        if table_in is None:
            self.df = sliced_df.reset_index(drop=True)
            
        else:
            return sliced_df
    

    def slice_min_or_max(self, column, choice):

        # Finding the max and min of the particular column
        max_value = self.df[column].max()
        min_value = self.df[column].min()

        max_criterion = self.df[column].map(lambda val: abs(val - max_value) < 0.001)
        min_criterion = self.df[column].map(lambda val: abs(val - min_value) < 0.001)

        max_index = self.df.index[max_criterion]
        min_index = self.df.index[min_criterion]

        if choice == 'min':
            self.df = self.df[min_criterion]
        
        elif choice == 'max':
            self.df = self.df[max_criterion]
        
        else:
            raise Exception("'min' or 'max' only")
            
        pass


    def reset(self):
        self.df = self.original_df


    def change_fontsize(self, fontsize):
        self.fontsize = fontsize
        plt.rcParams.update({'font.size': self.fontsize})


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


    def estimated_gi(self, table=None):
        """
        """

        if table is None:
            dataframe = self.df
        else:
            dataframe = table

        R_inv_list = [np.linalg.inv(R) for R in dataframe['R']]
        R_inv_array = np.asarray(R_inv_list)
        
        estimated_ellip_vec_list = []
        for i in range(len(dataframe['R'])):
            e1 = dataframe['reconvolved_noshear_e1'].to_numpy()[i]
            e2 = dataframe['reconvolved_noshear_e2'].to_numpy()[i]
            estimated_ellip_vec_list.append(np.array([[e1],[e2]]))
        
        estimated_ellip_vec_array = np.asarray(estimated_ellip_vec_list)
    
        estimated_shear_array = R_inv_array @ estimated_ellip_vec_array

        estimated_g1 = estimated_shear_array[:,0, 0]
        estimated_g2 = estimated_shear_array[:,1, 0]

        # NOW WITHOUT METACAL
        R_inv_NO_METACAL_array = [0.5 * np.eye(2) for R in dataframe['R']]
        # R_inv_NO_METACAL_array = [np.linalg.inv(2 * np.eye(2)) for R in self.df['R']]
        estimated_shear_array_NO_METACAL = R_inv_NO_METACAL_array @ estimated_ellip_vec_array

        estimated_g1_NO_METACAL = estimated_shear_array_NO_METACAL[:,0, 0]
        estimated_g2_NO_METACAL = estimated_shear_array_NO_METACAL[:,1, 0]

        return estimated_g1, estimated_g2, estimated_g1_NO_METACAL, estimated_g2_NO_METACAL


    
    def ax_plot_quadratic_m(self, ax, true_gi_array, m, i=None, color_column=None, blind=False, cmap='cividis', quad_fit=False, legend=True, plot_x_axis=True, plot_y_axis=True, ylims=None): 
        
        yi = m
        
        ax.scatter(true_gi_array, m)
    

        idxi = np.nonzero(true_gi_array)
        valid_gis = true_gi_array[idxi] 
        valid_yis = yi[idxi]

    
        if color_column is not None:

            # color_column_i = self.df[color_column].to_numpy()[idxi]
            # print(color_column_i)

            vmin = np.min(color_column)
            vmax = np.max(color_column)
            
            im = ax.scatter(true_gi_array, yi, c=color_column, cmap=cmap, vmin=vmin, vmax=vmax)
            
        
        else:
            im = ax.scatter(true_gi_array, yi, c='r', label='m')
        
        # do colorbar outside this function

        
        if quad_fit:
            a, b, c = np.polyfit(valid_gis, valid_yis, 2) 

            x = np.linspace(np.min(true_gi_array), np.max(true_gi_array), 20)
            
            ax.plot(x, a*x*x + b*x + c,
                    c='k', label=f"a = {a:.2f} \n b = {b:.2f} \n c = {c}", zorder=0)
        
        # plotting error limit of 0.001    
        ax.axhline(y=0.001, zorder=0, color='r')

        if i is not None:
            ax.set_xlabel(f'true_g{i}')
            ax.set_ylabel(fr'$m = (\frac{{{{g_2}}_{{est}} - {{g_2}}_{{true}}}}{{{{g_2}}_{{true}}}})$')
            ax.set_title(f'g{i}')

        if legend:
            ax.legend()
        
        if plot_x_axis:
            ax.axhline(zorder=0, color='k')
        
        if plot_y_axis:
            ax.axvline(zorder=0, color='k')

        if ylims is not None:
            ax.set_ylim(ylims)
        
        if quad_fit:
            return im, c, a

        return im


    def get_fit_parameters(self, dataframe_subset):


        true_g1 = dataframe_subset['oshear_g1'].to_numpy()
        true_g2 = dataframe_subset['oshear_g2'].to_numpy()



        estimated_g1, estimated_g2, estimated_g1_NOMETACAL, estimated_g2_NOMETACAL = self.estimated_gi(table=dataframe_subset)

        idx1 = np.nonzero(true_g1)
        idx2 = np.nonzero(true_g2)

        true_g1 = true_g1[idx1]
        true_g2 = true_g2[idx2]

        estimated_g1 = estimated_g1[idx1]
        estimated_g2 = estimated_g2[idx2]
        estimated_g1_NOMETACAL = estimated_g1_NOMETACAL[idx1]
        estimated_g2_NOMETACAL = estimated_g2_NOMETACAL[idx2]

        m1_nomc = (estimated_g1_NOMETACAL - true_g1) / true_g1
        m2_nomc = (estimated_g2_NOMETACAL - true_g2) / true_g2
        m1_mc = (estimated_g1 - true_g1) / true_g1
        m2_mc = (estimated_g2 - true_g2) / true_g2
    
        a1, b1, c1 = np.polyfit(true_g1, m1_mc, 2) 

        return a1, b1, c1


    def with_without_metacal(self, color_column=None, plotname=None, blind=False, show=True, cmap='cividis', nomc_ylims=None, mc_ylims=None):

        true_g1 = self.df['oshear_g1'].to_numpy()
        true_g2 = self.df['oshear_g2'].to_numpy()



        estimated_g1, estimated_g2, estimated_g1_NOMETACAL, estimated_g2_NOMETACAL = self.estimated_gi()

        idx1 = np.nonzero(true_g1)
        idx2 = np.nonzero(true_g2)

        true_g1 = true_g1[idx1]
        true_g2 = true_g2[idx2]

        estimated_g1 = estimated_g1[idx1]
        estimated_g2 = estimated_g2[idx2]
        estimated_g1_NOMETACAL = estimated_g1_NOMETACAL[idx1]
        estimated_g2_NOMETACAL = estimated_g2_NOMETACAL[idx2]

        if color_column is not None:
            color_column_1 = self.df[color_column].to_numpy()[idx1]
            color_column_2 = self.df[color_column].to_numpy()[idx2]
        
        else:
            color_column_1 = None
            color_column_2 = None


        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        # plotting to each individual ax 

        m1_nomc = (estimated_g1_NOMETACAL - true_g1) / true_g1
        m2_nomc = (estimated_g2_NOMETACAL - true_g2) / true_g2
        m1_mc = (estimated_g1 - true_g1) / true_g1
        m2_mc = (estimated_g2 - true_g2) / true_g2

        factor = 0.2

        if nomc_ylims is None:
            # without metacal ylims
            mmax_nomc = np.max([np.max(m1_nomc), np.max(m2_nomc)])
            mmin_nomc = np.min([np.min(m1_nomc), np.min(m2_nomc)])
            nomc_ylims = (mmin_nomc - factor*abs(mmin_nomc), mmax_nomc + factor*abs(mmax_nomc))

        if mc_ylims is None:
            mmax_mc = np.max([np.max(m1_mc), np.max(m2_mc)])
            mmin_mc = np.min([np.min(m1_mc), np.min(m2_mc)])
            mc_ylims = (mmin_mc - factor*abs(mmin_mc), mmax_mc + factor*abs(mmax_mc))

        #g1 without metacal
        im = self.ax_plot_quadratic_m(axs[0][0], true_g1, m1_nomc, i=1, color_column=color_column_1, blind=blind, ylims=nomc_ylims,
                                cmap=cmap, quad_fit=False, legend=True)

        #g2 without metacal
        im =self.ax_plot_quadratic_m(axs[0][1], true_g2, m2_nomc, i=2, color_column=color_column_2, blind=blind, ylims=nomc_ylims,
                                cmap=cmap, quad_fit=False, legend=True)
        
        #g1 with metacal
        im, c1_mc, a1 = self.ax_plot_quadratic_m(axs[1][0], true_g1, m1_mc, i=1, color_column=color_column_1, blind=blind, ylims=mc_ylims,
                                cmap=cmap, quad_fit=True, legend=True)

        #g2 with metacal
        im, c2_mc, a2 = self.ax_plot_quadratic_m(axs[1][1], true_g2, m2_mc, i=2, color_column=color_column_2, blind=blind, ylims=mc_ylims,
                                cmap=cmap, quad_fit=True, legend=True)

        plt.subplots_adjust(hspace=0.5, wspace=0.3)

        if color_column is not None:
            cbaxes0 = fig.add_axes([0.2, 0.05, 0.6, 0.01])
            cb0 = fig.colorbar(im, ax=axs[:], orientation='horizontal', shrink=0.45, cax=cbaxes0)
            cb0.set_label(color_column)

        if plotname is not None:
            self.save_fig_to_plots(plotname)

        if show:
            plt.show() 


        return nomc_ylims, mc_ylims, c1_mc, c2_mc, a1, a2


    def plot_quadratic_m(self, color_column=None, plotname=None, blind=False, show=True, ylims=None):

        true_g1 = self.df['oshear_g1'].to_numpy()
        true_g2 = self.df['oshear_g2'].to_numpy()
        
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        for i in range(2):
            axs[i][0].set_xlabel('true_g1')
            axs[i][0].set_ylabel(r'$m = (\frac{{g_1}_{est} - {g_1}_{true}}{{g_1}_{true}})$')
            axs[i][0].set_title('g1')
            axs[i][1].set_xlabel('true_g2')
            axs[i][1].set_ylabel(r'$m = (\frac{{g_2}_{est} - {g_2}_{true}}{{g_2}_{true}})$')
            axs[i][1].set_title('g2')

        estimated_g1, estimated_g2, estimated_g1_NOMETACAL, estimated_g2_NOMETACAL = self.estimated_gi()

        y1 = (estimated_g1 - true_g1)/true_g1
        y2 = (estimated_g2 - true_g2)/true_g2

        y1_nm = (estimated_g1_NOMETACAL - true_g1) / true_g1
        y2_nm = (estimated_g2_NOMETACAL - true_g2) / true_g2

        if color_column is not None:
            vmin = np.min(self.df[color_column])
            vmax = np.max(self.df[color_column])

            im00 = axs[0][0].scatter(true_g1, y1, c=self.df[color_column], cmap='cividis', vmin=vmin, vmax=vmax)
            im01 = axs[0][1].scatter(true_g2, y2, c=self.df[color_column], cmap='cividis', vmin=vmin, vmax=vmax)

            im10 = axs[1][0].scatter(true_g1, y1_nm, c=self.df[color_column], cmap='cividis', vmin=vmin, vmax=vmax)
            im11 = axs[1][1].scatter(true_g2, y2_nm, c=self.df[color_column], cmap='cividis', vmin=vmin, vmax=vmax)

            cbaxes0 = fig.add_axes([0.2, 0.05, 0.6, 0.01])
            # cbaxes1 = fig.add_axes([0.2, 0.95, 0.6, 0.01])
            cb0 = fig.colorbar(im00, ax=axs[:], orientation='horizontal', shrink=0.45, cax=cbaxes0)
            # cb1 = fig.colorbar(im1, ax=axs[:], orientation='horizontal', shrink=0.45, cax=cbaxes1)
            cb0.set_label(color_column)
            # cb1.set_label(color_column)
        
        else:
            im = axs[0][0].scatter(true_g1, y1, c='r', label='m')
            im = axs[0][1].scatter(true_g2, y2, c='r', label='m')
            im = axs[1][0].scatter(true_g1, y1_nm, c='r', label='m')
            im = axs[1][1].scatter(true_g2, y2_nm, c='r', label='m')

        plt.subplots_adjust(hspace=0.5, wspace=0.3)


        idx1 = np.nonzero(true_g1) 
        idx2 = np.nonzero(true_g2)

        # all valid points
        valid_g1s = true_g1[idx1]
        valid_g2s = true_g2[idx2]
        valid_y1s = y1[idx1]
        valid_y2s = y2[idx2]


        # With metacal
        a1, b1, c1 = np.polyfit(valid_g1s, valid_y1s, 2)
        a2, b2, c2 = np.polyfit(valid_g2s, valid_y2s, 2)

        # Without metacal

        # a1_nm, b1_nm, c1_nm = np.polyfit(true_g1[idx1], y1_nm[idx1], 2)
        # a2_nm, b2_nm, c2_nm = np.polyfit(true_g2[idx2], y2_nm[idx2], 2)

        x = np.linspace(-0.05, 0.05, 20)

        axs[0][0].plot(x, a1*x*x + b1*x + c1, c='k', label=f"a1 = {a1:.2f} \n b1 = {b1:.2f} \n c1 = {c1}", zorder=0)
        axs[0][1].plot(x, a2*x*x + b2*x + c2, c='k', label=f"a2 = {a2:.2f} \n b2 = {b2:.2f} \n c2 = {c2}", zorder=0)

        # axs[1][0].plot(x, a1_nm*x*x + b1_nm*x + c1_nm, c='k', label=f"a1_nm = {a1_nm:.2f} \n b1_nm = {b1_nm} \n c1_nm = {c1_nm}", zorder=0)
        # axs[1][1].plot(x, a2_nm*x*x + b2_nm*x + c2_nm, c='k', label=f"a2_nm = {a2_nm:.2f} \n b2_nm = {b2_nm} \n c2_nm = {c2_nm}", zorder=0)

        for i in range(2):
            for j in range(2):
                axs[i][j].legend()
                axs[i][j].axhline(zorder=0, color='k')
                axs[i][j].axhline(y=0.001, zorder=0, color='r')
                
                if ylims is not None:
                    axs[i][j].set_ylim(ylims)

        # axs[0].legend()
        # axs[1].legend()



        fig.suptitle(r'$m = (\frac{{g_i,}_{est} - {g_i,}_{true}}{{g_i,}_{true}})$ by element')

        if blind:
            for ax in axs:
                ax.tick_params(bottom=False, top=False, left=False, right=False,
                            labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        if plotname is not None:
            self.save_fig_to_plots(plotname)
        
        if show:
            plt.show() 

        return axs[0][0].get_ylim(), c1


    def plot_quadratic_m_simplified(self, color_column=None, plotname=None, blind=False, axis_equal=False):

        true_g1 = self.df['oshear_g1'].to_numpy()
        true_g2 = self.df['oshear_g2'].to_numpy()
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 6))

        for i in range(2):
            axs[i].set_xlabel('True Cosmic Shear')
            axs[i].set_ylabel('Fractional Error')
            # axs[i].set_title('g1')

        axs[1].set_title('With Metacalibration')
        axs[0].set_title('Without Metacalibration')

        estimated_g1, estimated_g2, estimated_g1_NOMETACAL, estimated_g2_NOMETACAL = self.estimated_gi()

        y1 = (estimated_g1 - true_g1)/true_g1

        y1_nm = (estimated_g1_NOMETACAL - true_g1) / true_g1

        if color_column is not None:
            vmin = np.min(self.df[color_column])
            vmax = np.max(self.df[color_column])

            im0 = axs[1].scatter(true_g1, y1, c=self.df[color_column], cmap='cividis', vmin=vmin, vmax=vmax)
            axs[1].axhline(zorder=0, color='k')

            im1 = axs[0].scatter(true_g1, y1_nm, c=self.df[color_column], cmap='cividis', vmin=vmin, vmax=vmax)
            axs[0].axhline(zorder=0, color='k')

            cbaxes0 = fig.add_axes([0.2, 0.1, 0.6, 0.02])
            # cbaxes1 = fig.add_axes([0.2, 0.95, 0.6, 0.01])
            cb0 = fig.colorbar(im0, ax=axs[:], orientation='horizontal', shrink=0.45, cax=cbaxes0)
            # cb1 = fig.colorbar(im1, ax=axs[:], orientation='horizontal', shrink=0.45, cax=cbaxes1)
            cb0.set_label(color_column)
            # cb1.set_label(color_column)
        
        else:
            im = axs[1].scatter(true_g1, y1, c='r', label='m')
            im = axs[0].scatter(true_g1, y1_nm, c='r', label='m')

        if axis_equal:

            maxlim = np.max([axs[0].get_ylim()[1], axs[1].get_ylim()[1]])
            minlim = np.min([axs[0].get_ylim()[0], axs[1].get_ylim()[0]])

            for ax in axs: 
                ax.set_ylim(bottom=minlim, top=maxlim)
        # Adding zero line
        # plt.hlines([0.0], xmin=np.min(true_g1), xmax=np.max(true_g1))

        plt.subplots_adjust(hspace=0.5, wspace=0.3, bottom=0.25)


        idx1 = np.nonzero(true_g1) 
        idx2 = np.nonzero(true_g2)

        # With metacal
        a1, b1, c1 = np.polyfit(true_g1[idx1], y1[idx1], 2)

        # Without metacal

        # a1_nm, b1_nm, c1_nm = np.polyfit(true_g1[idx1], y1_nm[idx1], 2)

        x = np.linspace(-0.05, 0.05, 20)


        axs[1].plot(x, a1*x*x + b1*x + c1, c='k', label=f"a1 = {a1:.2f} \n b1 = {b1:.2f} \n c1 = {c1:.2f}", zorder=0)

        # axs[1][0].plot(x, a1_nm*x*x + b1_nm*x + c1_nm, c='k', label=f"a1_nm = {a1_nm:.2f} \n b1_nm = {b1_nm} \n c1_nm = {c1_nm}", zorder=0)
        # axs[1][1].plot(x, a2_nm*x*x + b2_nm*x + c2_nm, c='k', label=f"a2_nm = {a2_nm:.2f} \n b2_nm = {b2_nm} \n c2_nm = {c2_nm}", zorder=0)

        axs[1].legend()

        fig.suptitle(r'Fractional Error $m = (\frac{{g,}_{est} - {g,}_{true}}{{g,}_{true}})$')

        if blind:
            for ax in axs:
                ax.tick_params(bottom=False, top=False, left=False, right=False,
                            labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        if plotname is not None:
            self.save_fig_to_plots(plotname)
        

        plt.show() 




    def ax_plot_absolute_error(self, ax, estimated_gi_array, true_gi_array, i, color_array=None, color='r', cmap='cividis', ylims=None):
        
        abs_error = estimated_gi_array - true_gi_array

        ax.set_xlabel(f'true_g{i}')
        ax.set_ylabel(rf'$({{g_{i}}}_{{est}} - {{g_{i}}}_{{true}})$')
        ax.set_title(f'g{i}')

        if ylims is not None:
            ax.set_ylim(ylims)

        if color_array is not None:
            vmin = np.min(color_array)
            vmax = np.max(color_array)

            im = ax.scatter(true_gi_array, abs_error, c=color_array, cmap=cmap, vmin=vmin, vmax=vmax)
        
        else:
            im = ax.scatter(true_gi_array, abs_error, c=color)


        pass


    def plot_absolute_error(self, color_column=None, plotname=None, blind=False, show=True):

        true_g1 = self.df['oshear_g1'].to_numpy()
        true_g2 = self.df['oshear_g2'].to_numpy()
        
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        for i in range(2):
            axs[i][0].set_xlabel('true_g1')
            axs[i][0].set_ylabel(r'$m = ({g_1}_{est} - {g_1}_{true})$')
            axs[i][0].set_title('g1')
            axs[i][1].set_xlabel('true_g2')
            axs[i][1].set_ylabel(r'$m = ({g_2}_{est} - {g_2}_{true})$')
            axs[i][1].set_title('g2')

        estimated_g1, estimated_g2, estimated_g1_NOMETACAL, estimated_g2_NOMETACAL = self.estimated_gi()

        y1 = (estimated_g1 - true_g1)
        y2 = (estimated_g2 - true_g2)

        y1_nm = (estimated_g1_NOMETACAL - true_g1) 
        y2_nm = (estimated_g2_NOMETACAL - true_g2) 

        if color_column is not None:
            vmin = np.min(self.df[color_column])
            vmax = np.max(self.df[color_column])

            im00 = axs[1][0].scatter(true_g1, y1, c=self.df[color_column], cmap='cividis', vmin=vmin, vmax=vmax)
            im01 = axs[1][1].scatter(true_g2, y2, c=self.df[color_column], cmap='cividis', vmin=vmin, vmax=vmax)

            im10 = axs[0][0].scatter(true_g1, y1_nm, c=self.df[color_column], cmap='cividis', vmin=vmin, vmax=vmax)
            im11 = axs[0][1].scatter(true_g2, y2_nm, c=self.df[color_column], cmap='cividis', vmin=vmin, vmax=vmax)

            cbaxes0 = fig.add_axes([0.2, 0.05, 0.6, 0.01])
            # cbaxes1 = fig.add_axes([0.2, 0.95, 0.6, 0.01])
            cb0 = fig.colorbar(im00, ax=axs[:], orientation='horizontal', shrink=0.45, cax=cbaxes0)
            # cb1 = fig.colorbar(im1, ax=axs[:], orientation='horizontal', shrink=0.45, cax=cbaxes1)
            cb0.set_label(color_column)
            # cb1.set_label(color_column)
        
        else:
            im = axs[0][0].scatter(true_g1, y1, c='r', label='m')
            im = axs[0][1].scatter(true_g2, y2, c='r', label='m')
            im = axs[1][0].scatter(true_g1, y1_nm, c='r', label='m')
            im = axs[1][1].scatter(true_g2, y2_nm, c='r', label='m')

        plt.subplots_adjust(hspace=0.5, wspace=0.4)


        idx1 = np.nonzero(true_g1) 
        idx2 = np.nonzero(true_g2)

        # With metacal
        a1, b1, c1 = np.polyfit(true_g1[idx1], y1[idx1], 2)
        a2, b2, c2 = np.polyfit(true_g2[idx2], y2[idx2], 2)

        # Without metacal

        # a1_nm, b1_nm, c1_nm = np.polyfit(true_g1[idx1], y1_nm[idx1], 2)
        # a2_nm, b2_nm, c2_nm = np.polyfit(true_g2[idx2], y2_nm[idx2], 2)

        x = np.linspace(-0.05, 0.05, 20)


        # axs[0][0].plot(x, a1*x*x + b1*x + c1, c='k', label=f"a1 = {a1:.2f} \n b1 = {b1:.2f} \n c1 = {c1:.2f}", zorder=0)
        # axs[0][1].plot(x, a2*x*x + b2*x + c2, c='k', label=f"a2 = {a2:.2f} \n b2 = {b2:.2f} \n c2 = {c2:.2f}", zorder=0)

        # axs[1][0].plot(x, a1_nm*x*x + b1_nm*x + c1_nm, c='k', label=f"a1_nm = {a1_nm:.2f} \n b1_nm = {b1_nm} \n c1_nm = {c1_nm}", zorder=0)
        # axs[1][1].plot(x, a2_nm*x*x + b2_nm*x + c2_nm, c='k', label=f"a2_nm = {a2_nm:.2f} \n b2_nm = {b2_nm} \n c2_nm = {c2_nm}", zorder=0)

        # for i in range(2):
        #     for j in range(2):
        #         axs[i][j].legend()

        # axs[0].legend()
        # axs[1].legend()



        fig.suptitle(r'$({g_i,}_{est} - {g_i,}_{true})$ by element')

        if blind:
            for ax in axs:
                ax.tick_params(bottom=False, top=False, left=False, right=False,
                            labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        if plotname is not None:
            self.save_fig_to_plots(plotname)
        
        if show:
            plt.show() 


    def plot_row_images(self, row_index, height=4.5, axes=False, plotname=None):
        
        row_dict = self.df.loc[row_index, :].to_dict()

        # true_g1 = row_dict['oshear_g1']
        # true_g2 = row_dict['oshear_g2']
        # R = row_dict['R']
        # e1 = row_dict['reconvolved_noshear_e1']
        # e2 = row_dict['reconvolved_noshear_e2']

        # g1g2_est = np.linalg.inv(R) @ np.array([[e1], [e2]])

        # g1_est = g1g2_est[0]
        # g2_est = g1g2_est[1]

        # m1 = (g1_est - true_g1) / true_g1
        # m2 = (g2_est - true_g2) / true_g2

        # print(m1, m2)


        # original_gal = row_dict['original_gal']
        # true_psf = row_dict['true_psf']
        # pixel_scale = row_dict['pixel_scale']

        # original_gal_image = original_gal.drawImage(scale=pixel_scale)
        # true_psf_image = true_psf.drawImage(scale=pixel_scale)

        mcObject = row_dict['mcObject']

        reconvolved_no_shear_image = mcObject.reconvolved_noshear.drawImage(scale=mcObject.pixel_scale, offset=mcObject.offset)

        nx = reconvolved_no_shear_image.array.shape[1]
        ny = reconvolved_no_shear_image.array.shape[0]

        original_gal_image = mcObject.original_gal.drawImage(nx=nx, ny=ny, scale=mcObject.pixel_scale, offset=mcObject.offset)
        true_psf_image = mcObject.true_psf.drawImage(nx=nx, ny=ny, scale=mcObject.pixel_scale, offset=mcObject.offset)
        cosmic_sheared_image = mcObject._cosmic_sheared.drawImage(nx=nx, ny=ny, scale=mcObject.pixel_scale, offset=mcObject.offset)
        deconvolved_image = mcObject._deconvolved.drawImage(nx=nx, ny=ny, scale=mcObject.pixel_scale, offset=mcObject.offset)

        plus_g1 = mcObject._reconvolved_plus_g1.drawImage(nx=nx, ny=ny, scale=mcObject.pixel_scale, method='no_pixel', offset=mcObject.offset)
        minus_g1 = mcObject._reconvolved_minus_g1.drawImage(nx=nx, ny=ny, scale=mcObject.pixel_scale, method='no_pixel', offset=mcObject.offset)

        plus_g2 = mcObject._reconvolved_plus_g2.drawImage(nx=nx, ny=ny, scale=mcObject.pixel_scale, method='no_pixel', offset=mcObject.offset)
        minus_g2 = mcObject._reconvolved_minus_g2.drawImage(nx=nx, ny=ny, scale=mcObject.pixel_scale, method='no_pixel', offset=mcObject.offset)

        number = 4
        numrows = 3
        width = height * number

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig, axs = plt.subplots(numrows, number, figsize = (width, numrows * height))

        axs[0][0].imshow(original_gal_image.array)
        axs[0][1].imshow(true_psf_image.array, cmap='plasma')
        axs[0][2].imshow(deconvolved_image.array)
        axs[0][3].imshow(reconvolved_no_shear_image.array)

        # difference between original sheared and deconvolved
        conv_diff = deconvolved_image.array - cosmic_sheared_image.array
        vmax_conv_diff = np.max(conv_diff)
        im = axs[2][0].imshow(conv_diff, cmap='PiYG', vmax=vmax_conv_diff, vmin=-vmax_conv_diff)
        divider = make_axes_locatable(axs[2][0])
        cax = divider.append_axes('right', size='5%', pad=0)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label('Count difference')

        g1_plus_diff = plus_g1.array - reconvolved_no_shear_image.array
        g1_minus_diff = minus_g1.array - reconvolved_no_shear_image.array
        g2_plus_diff = plus_g2.array - reconvolved_no_shear_image.array
        g2_minus_diff = minus_g2.array - reconvolved_no_shear_image.array

        print(g1_plus_diff.sum())
        print(g1_minus_diff.sum())
        print(g2_plus_diff.sum())
        print(g2_minus_diff.sum())
        
        vmax_g1_plus = np.max(g1_plus_diff)
        vmax_g1_minus = np.max(g1_minus_diff)
        vmax_g2_plus = np.max(g2_plus_diff)
        vmax_g2_minus = np.max(g2_minus_diff)        

        im = axs[1][0].imshow(g1_plus_diff, cmap='PiYG', vmax=vmax_g1_plus, vmin=-vmax_g1_plus)

        divider = make_axes_locatable(axs[1][0])
        cax = divider.append_axes('right', size='5%', pad=0)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label('Count difference')

        im = axs[1][1].imshow(g1_minus_diff, cmap='PiYG', vmax=vmax_g1_minus, vmin=-vmax_g1_minus)

        divider = make_axes_locatable(axs[1][1])
        cax = divider.append_axes('right', size='5%', pad=0)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label('Count difference')

        im = axs[1][2].imshow(g2_plus_diff, cmap='PiYG', vmax=vmax_g2_plus, vmin=-vmax_g2_plus)

        divider = make_axes_locatable(axs[1][2])
        cax = divider.append_axes('right', size='5%', pad=0)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label('Count difference')

        im = axs[1][3].imshow(g2_minus_diff, cmap='PiYG', vmax=vmax_g2_minus, vmin=-vmax_g2_minus)

        divider = make_axes_locatable(axs[1][3])
        cax = divider.append_axes('right', size='5%', pad=0)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label('Count difference')


        axs[0][0].set_title('Original Galaxy')
        axs[0][1].set_title('True PSF')
        axs[0][2].set_title('Deconvolved Galaxy')
        axs[0][3].set_title('Reconvolved No Shear')

        axs[1][0].set_title('rec+g1 - rec0')
        axs[1][1].set_title('rec-g1 - rec0')
        axs[1][2].set_title('rec+g2 - rec0')
        axs[1][3].set_title('rec-g2 - rec0')

        axs[2][0].set_title('deconvolved - pre-seeing galaxy')

        plt.subplots_adjust(hspace=0.3, wspace=0.3) 
        
        if not axes:
            axs_iter = axs.flatten()
            for ax in axs_iter:
                ax.set_xlabel('x')
                ax.set_ylabel('y')

                ax.tick_params(bottom=False, top=False, left=False, right=False,
                            labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        if plotname is not None:
            plt.savefig(plotname)
        plt.show()



    # plot quadratic_m
    # plot_R_elements
    # generate image


class comboObject:
    """
    A class that contains functionality for generating combinations of parameters to
    loop through, feeding those combinations into metacal using multiprocessing, and then
    pickling the results array to be read by an mcSummaryObject
    
    """

    def __init__(self):
        """
        Initializing different lists and maps of parameters to loop over when combinations are generated. If a user wishes to customize the lists of parameters
        to use (and they almost always will), they will need to reassign these attributes for the object post-initialization and before running the generate_combinations() method.
        
        """

        # Default folder to which generated pickle files will be saved
        self.folder = 'pickles'
        
        # Default dict of original_gal profiles to loop through. The value "0" for the gaussian
        # is never read. 
        self.gal_profiles = {'gaussian': 0}

        # Default list of original_gal fluxes to loop through
        self.gal_fluxes = [1.e5]

        # Default dict of true PSF profiles to loop through. The value "0" for the gaussian
        # is never read. For moffat profiles, the value of the dict is the desired "beta" parameter. 
        self.true_psf_profiles = {'gaussian': 0, 'moffat': 3.5, 'moffat': 5}

        # Default list of shear estimators to use (REGAUSS, LINEAR, and BJ currently supported)
        self.shape_mes_algs= ['REGAUSS', 'LINEAR', 'BJ'] #TODO figure out how to deal with KSB

        # Default list of pixel scales that one desires to loop through
        self.pixel_scales = [0.2, 0.02]

        # Default list of true shears to apply in both g1 and g2
        self.oshear_dgs = [i for i in np.arange(-0.05, 0.06, 0.01) if not abs(i) < 0.001]

        # Defaut list of calibration shear magnitudes to apply in both g1 and g2
        self.cshear_dgs = [0.01]

        # Default list of true PSF FWHMs to loop through
        self.true_psf_fwhms = np.arange(0.5, 1.3, 0.1)

        # Default list of galaxy to PSF ratios for which to create galaxies
        self.gal_psf_ratios = np.arange(0.5, 2.1, 0.1)

        # Default list of tuples of offsets from center to test (e.g. [(0.5, 0.5)])
        self.offsets = [None]

        # Default list of wrong_psf_fwhms to use (default is to not use wrong PSFs, indicated by using None here)
        self.wrong_psf_fwhms = None


    def print_parameters(self):
        """
        Prints the parameters that will be generated -- good for double checking that everything is as expected
        """

        print('\n' + 20 * '-')
        print(f"Galaxy Profiles: {self.gal_profiles}")
        print(f"True PSF Profiles: {self.true_psf_profiles}")
        print(f"Shape Measurement Algorithms: {self.shape_mes_algs}")
        print(f"Pixel Scales {self.pixel_scales}")
        print(f"Galaxy fluxes: {self.gal_fluxes}")
        print(f"Cosmic shears: {self.oshear_dgs}")
        print(f"Calibration shear magnitudes: {self.cshear_dgs}")
        print(f"True PSF fwhms: {self.true_psf_fwhms}")
        print(f"Gal/PSF ratios: {self.gal_psf_ratios}")
        print(f"Offsets: {self.offsets}")
        print(f"Wrong PSF fwhms: {self.wrong_psf_fwhms}")
        print(20 * '-' + '\n')


    def _create_object_list(self):
        """
        Creates a list metacalObject instances, looping through the desired parameters indicated in __init__ and 
        any explicit changes to parameters made by the user

        Returns:

            combinations:       list of metacalObject instances      Used in the generate_combinations method
        """

        combinations = []

        for gal_profile in self.gal_profiles.keys():
            for true_psf_profile in self.true_psf_profiles.keys():
                for shape_mes_alg in self.shape_mes_algs:
                    for pixel_scale in self.pixel_scales:
                        for gal_flux in self.gal_fluxes:
                            for oshear_dg in self.oshear_dgs:
                                for cshear_dg in self.cshear_dgs:
                                    for offset in self.offsets:
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
                                                
                                                deconv_psf = true_psf
                                                # dilating reconv_psf
                                                reconv_psf = deconv_psf.dilate(dilation_factor)

                                                combinations.append(metacalObject(original_gal, oshear_dg, 0.0, true_psf, deconv_psf, reconv_psf, reconv_psf, cshear_dg, cshear_dg, shape_mes_alg, pixel_scale, offset))
                                                combinations.append(metacalObject(original_gal, 0.0, oshear_dg, true_psf, deconv_psf, reconv_psf, reconv_psf, cshear_dg, cshear_dg, shape_mes_alg, pixel_scale, offset))

        return combinations 


    def _create_object_list_wrong_psf(self):
        """
        Used in the case where the user desires to use the wrong PSF. Function is called when the attribute self.wrong_psf_fwhms is not None.

        Returns:

            combinations:       list of metacalObject instances     Used in the generate_combinations() method
        """

        combinations = []

        for gal_profile in self.gal_profiles.keys():
            for true_psf_profile in self.true_psf_profiles.keys():
                for shape_mes_alg in self.shape_mes_algs:
                    for pixel_scale in self.pixel_scales:
                        for gal_flux in self.gal_fluxes:
                            for oshear_dg in self.oshear_dgs:
                                for cshear_dg in self.cshear_dgs:
                                    for offset in self.offsets:
                                        for gal_psf_ratio in self.gal_psf_ratios:
                                            for true_psf_fwhm in self.true_psf_fwhms:
                                                for wrong_psf_fwhm in self.wrong_psf_fwhms:
                                                
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
                                                        wrong_psf = galsim.Gaussian(flux=1.0, fwhm=wrong_psf_fwhm)
                                                    
                                                    elif true_psf_profile == 'moffat':
                                                        # don't forget beta
                                                        true_psf = galsim.Moffat(flux=1.0, fwhm=true_psf_fwhm, beta=self.true_psf_profiles[true_psf_profile])
                                                        wrong_psf = galsim.Moffat(flux=1.0, fwhm=wrong_psf_fwhm, beta=self.true_psf_profiles[true_psf_profile])

                                                    else:
                                                        raise Exception('Invalid PSF profile!')
                                                    
                                                    deconv_psf = wrong_psf


                                                    # dilating reconv_psf
                                                    reconv_psf = deconv_psf.dilate(dilation_factor)

                                                    g1_only = metacalObject(original_gal, 0.0, oshear_dg, true_psf, deconv_psf, reconv_psf, reconv_psf, cshear_dg, cshear_dg, shape_mes_alg, pixel_scale, offset)
                                                    g2_only = metacalObject(original_gal, oshear_dg, 0.0, true_psf, deconv_psf, reconv_psf, reconv_psf, cshear_dg, cshear_dg, shape_mes_alg, pixel_scale, offset)

                                                    psf_var_frac_err = (true_psf_fwhm**2 - wrong_psf_fwhm**2) / true_psf_fwhm**2
                                                    g1_only.psf_var_frac_err = psf_var_frac_err
                                                    g2_only.psf_var_frac_err = psf_var_frac_err

                                                    combinations.append(g1_only)
                                                    combinations.append(g2_only)
                                                    


            return combinations 


    def _pickle_dont_overwrite(self, results, storage_file):
        """
        Save "results" to the file "storage_file" in the pickles folder without risk of overwriting.

        Parameters:

            results:        any datatype (usually a list of tuples)

            storage_file:   string                                          Name of the pickle file to create (without suffix .pickle)


        """

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


    def _apply_metacalibration_lambda(self, object):
        """
        
        Run self.metacalibration() on a metacalObject instance and return the new object

        Parameters:

            object:     a metacalObject instance

        Returns:

            new:        the metacalObject instance after self.metacalibration() has been run

        """

        new = object.metacalibration()
        print(object.R)
        return new
        

    def generate_combinations(self, storage_file, num_workers):
        """
        Run metacalibration on a combination of parameters using a certain number of workers and
        save the results to a pickle file with name storage_file

        Parameters:

            storage_file        string      The name of the pickle file to which to store the results. Do not include the .pickle suffix

            num_workers         int         The number of workers to use in multiprocessing

        Returns:

            VOID

        """
        
        if self.wrong_psf_fwhms is not None:
            combinations = self._create_object_list_wrong_psf()

        else:
            combinations = self._create_object_list()

        

        with Pool(num_workers) as p:
            results = p.map(self._apply_metacalibration_lambda, combinations)

        filename = self._pickle_dont_overwrite(results, storage_file)
        
        print(f"Results stored to {filename}")


class metacalObject:
    """

    An object containing parameters 

    """
    

    def __init__(self, original_gal, oshear_g1, oshear_g2, true_psf,
                        deconv_psf, reconv_psf, shear_estimation_psf,
                        cshear_dg1, cshear_dg2, shear_estimator, pixel_scale, offset):
        """
        Constructor for metacalObject instances.

        Parameters:

            original_gal            Galsim object               The source galaxy object

            oshear_g1               float                       The cosmic shear to apply in g1

            oshear_g2               float                       The cosmic shear to apply in g2

            true_psf                Galsim object               The True PSF Galsim object

            deconv_psf              Galsim object               The deconvolution PSF (Gamma 2) Galsim object

            reconv_psf              Galsim object               The reconvolution PSF (Gamma 3) Galsim object

            shear_estimation_psf    Galsim object               The PSF used by the shear estimator (Gamma 4), also a Galsim object

            cshear_dg1              float                       The calibration shear magnitude (in g1). Sheldon and Huff uses 0.01

            cshear_dg2              float                       The calibration shear magnitude (in g2). Sheldon and Huff uses 0.01    

            shear_estimator         string                      REGAUSS, LINEAR, or BJ (KSB not yet supported), see Galsim documentation on galsim.EstimateShear()

            pixel_scale             float                       The pixel scale to draw images with.

            offset                  length-2 tuple (x, y)       The offset by which to draw galaxy images. Default is (0, 0), which draws the center of the object at the corner of four pixels


        """
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
        self.offset = offset


    def _generate_observed_galaxy(self):
        """

        Creates a new attribute for the metacalObject called _observed_gal, which is the source galaxy
        plus cosmic shear plus convolution by the true PSF.

        """
        # shearing the original galaxy
        sheared = self.original_gal.shear(g1=self.oshear_g1, g2=self.oshear_g2)
        self._cosmic_sheared = sheared
        # Convolving the sheared galaxy with the PSF
        observed = galsim.Convolve(sheared, self.true_psf)

        self._observed_gal = observed


    def _delta_shear(self):
        """
        Using the _observed_galaxy object, two PSFs for metacal (deconvolving
        and re-convolving), and the amount by which to shift g1 and g2, creates "private" attributes
        for the shears in +-g1 and +-g2
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

        self.reconvolved_noshear = galsim.Convolve(self._deconvolved, self.reconv_psf)


    def _shear_response(self): 

        """

        Measures the shear response of a particular image.
        
        Creates the attributes R, noshear_e1, and noshear_e2 for the metacalObject based on previously created attributes.

        Created attributes:

            R:              2D numpy array      The calculated shear response matrix 

            noshear_e1:     float               The measured shape (distortion, first component) of the galaxy to which no calibration shear was applied

            noshear_e2:     float               The measured shape (distortion, second component) of the galaxy to which no calibration shear was applied

        """

        # Measuring galaxy shape parameters
        # We want to measure the shapes of reconvolved_plus_galaxy and reconvolved_minus_galaxy
        # the documentation recommends that we use the method='no_pixel' on the images

        plus_g1 = self._reconvolved_plus_g1.drawImage(scale=self.pixel_scale, method='no_pixel', offset=self.offset)
        minus_g1 = self._reconvolved_minus_g1.drawImage(scale=self.pixel_scale, method='no_pixel', offset=self.offset)

        plus_g2 = self._reconvolved_plus_g2.drawImage(scale=self.pixel_scale, method='no_pixel', offset=self.offset)
        minus_g2 = self._reconvolved_minus_g2.drawImage(scale=self.pixel_scale, method='no_pixel', offset=self.offset)

        noshear_image = self.reconvolved_noshear.drawImage(scale=self.pixel_scale, method='no_pixel', offset=self.offset)

        psf_shearestimator_image = self.shear_estimation_psf.drawImage(scale=self.pixel_scale, offset=self.offset)

        try:
            plus_moments_g1 = galsim.hsm.EstimateShear(plus_g1, psf_shearestimator_image, shear_est=self.shear_estimator)

        except galsim.GalSimError as e:
            print(f'plus_g1 moments failed: {str(e)}')
             
            self.R = np.nan
            self.noshear_e1 = np.nan
            self.noshear_e2 = np.nan 
            return

        try:
            minus_moments_g1 = galsim.hsm.EstimateShear(minus_g1, psf_shearestimator_image, shear_est=self.shear_estimator)

        except galsim.GalSimError as e:
            print(f'minus_g1 moments failed: {str(e)}')
        
            self.R = np.nan
            self.noshear_e1 = np.nan
            self.noshear_e2 = np.nan 
            return

        try:
            plus_moments_g2 = galsim.hsm.EstimateShear(plus_g2, psf_shearestimator_image, shear_est=self.shear_estimator)
            
        except galsim.GalSimError as e:
            print(f'plus_g2 moments failed: {str(e)}')

            self.R = np.nan
            self.noshear_e1 = np.nan
            self.noshear_e2 = np.nan 
            return

        try:
            minus_moments_g2 = galsim.hsm.EstimateShear(minus_g2, psf_shearestimator_image, shear_est=self.shear_estimator)
        
        except galsim.GalSimError as e:
            print(f'minus_g2 moments failed: {str(e)}')

            self.R = np.nan
            self.noshear_e1 = np.nan
            self.noshear_e2 = np.nan 
            return

        try:
            noshear_moments = galsim.hsm.EstimateShear(noshear_image, psf_shearestimator_image, shear_est=self.shear_estimator)
        
        except galsim.GalSimError as e:
            print(f'no_shear moments failed: {str(e)}')
             
            self.R = np.nan
            self.noshear_e1 = np.nan
            self.noshear_e2 = np.nan 
            return



        e1_plus_g1 = plus_moments_g1.corrected_e1
        e2_plus_g1 = plus_moments_g1.corrected_e2

        e1_minus_g1 = minus_moments_g1.corrected_e1
        e2_minus_g1 = minus_moments_g1.corrected_e2

        e1_plus_g2 = plus_moments_g2.corrected_e1
        e2_plus_g2 = plus_moments_g2.corrected_e2

        e1_minus_g2 = minus_moments_g2.corrected_e1
        e2_minus_g2 = minus_moments_g2.corrected_e2

        # Calculating shape of reconvolved_no_shear to test accuracy of shear response
        noshear_e1 = noshear_moments.corrected_e1
        noshear_e2 = noshear_moments.corrected_e2

        # calculating the shear response matrix R
        R_11 = (e1_plus_g1 - e1_minus_g1) / (2 * self.cshear_dg1)
        R_12 = (e2_plus_g1 - e2_minus_g1) / (2 * self.cshear_dg1)
        R_21 = (e1_plus_g2 - e1_minus_g2) / (2 * self.cshear_dg2)
        R_22 = (e2_plus_g2 - e2_minus_g2) / (2 * self.cshear_dg2)

        R = np.array([[R_11, R_12],[R_21, R_22]])


        self.R = R
        self.noshear_e1 = noshear_e1
        self.noshear_e2 = noshear_e2 


    def metacalibration(self):
        """
        Perform metacalibration on the metacalObject instance, and return the new object.        
        """
        
        self._generate_observed_galaxy()
        self._delta_shear()
        self._shear_response()

        return self
        return (self,
                self.original_gal,
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
                self.reconvolved_noshear,
                self.noshear_e1,
                self.noshear_e2)



    def show_images(self, height, fontsize=20):
        """
        Shows images of the different objects used in this object's instance of metacal.

        Parameters:

            height      float       Height of the desired image

            fontsize    float       Fontsize of the labels in the images (Default: 20)
        """    
        number = 5
        width = height * number


        original_gal_image = self.original_gal.drawImage(scale=self.pixel_scale, offset=self.offset)
        true_psf_image = self.true_psf.drawImage(scale=self.pixel_scale, offset=self.offset)
        observed_gal_image = self._observed_gal.drawImage(scale=self.pixel_scale, offset=self.offset)
        deconvolved_image = self._deconvolved.drawImage(scale=self.pixel_scale, offset=self.offset)
        reconvolved_noshear_image = self.reconvolved_noshear.drawImage(scale=self.pixel_scale, offset=self.offset)
        
        fig, axs = plt.subplots(1, number, figsize=(width, height))

        axs[0].imshow(original_gal_image.array)
        axs[1].imshow(true_psf_image.array, cmap='plasma')
        axs[2].imshow(observed_gal_image.array)
        axs[3].imshow(deconvolved_image.array)
        axs[4].imshow(reconvolved_noshear_image.array)
        
        title_dict = {0: 'Original Galaxy', 1: 'True PSF', 2: 'Observed Galaxy', 3: 'Deconvolved Galaxy', 4: 'Reconvolved (no shear)'}

        for i in range(number):
            axs[i].set_title(title_dict[i], fontsize=fontsize)
            axs[i].set_xlabel('x', fontsize=fontsize)
            axs[i].set_ylabel('y', fontsize=fontsize) 

        plt.show()
