import mcObjects
import sys
import numpy as np
import pdb
import matplotlib.pyplot as plt

true_psf_profiles = {'gaussian': 0}
shear_ests = ['REGAUSS']
pixel_scales = [0.2]
# calibration shears unchanged
true_psf_fwhms = [0.7]

# WIDE RANGE OF GAL_PSF_RATIOS
# gal_psf_ratios = [1.0]
gal_psf_ratios = np.arange(0.5, 5.1, 0.5)


# offsets unchanged
wrong_psf_fwhms = np.arange(true_psf_fwhms[0] - 0.005, true_psf_fwhms[0] + 0.006, 0.001)
#wrong_psf_fwhms = np.arange(0.68, 0.721, 0.001)
# wrong_psf_fwhms = [0.7]

def generate():
    # DESCRIPTION OF WHAT'S BEING GENERATED
    pass

def plot():
    # DESCRIPTION OF WHAT'S BEING PLOTTED

    obj = mcObjects.mcSummaryObject('pickles/gaussian_wrong_psfs_FWHM=0.7_scale=0.2_all_ratios_deltaFWHM_pm0.005.pickle')

    def get_c(obj, dataframe):

        a1, b1, c1 = obj.get_fit_parameters(dataframe)

        return c1 

    deconv_psf_fwhm_array = obj.df['deconv_psf_fwhm'].to_numpy()
    true_psf_fwhm_array = 0.7 * np.ones(len(deconv_psf_fwhm_array))

    obj.df['d_rpsf^2'] = true_psf_fwhm_array*true_psf_fwhm_array - deconv_psf_fwhm_array*deconv_psf_fwhm_array 

    obj.df['d_rpsf^2 / rpsf^2'] = obj.df['d_rpsf^2'] / (true_psf_fwhm_array * true_psf_fwhm_array)


    delta_to_c = {}

    # need to generate a list of all unique d_rpsf^2 / rpsf^2
    err_set = set()
    
    for err in obj.df['d_rpsf^2 / rpsf^2'].to_numpy():
        err_set.add(err)

    print(err_set)

    for err in err_set:
    # for wrong_fwhm in wrong_psf_fwhms:

        sliced_by_drpsf = obj.slice('d_rpsf^2 / rpsf^2', value=err, table_in=obj.df)

        c_list = []
        for ratio in gal_psf_ratios:
            subset = obj.slice('gal_psf_ratio', value=ratio, table_in=sliced_by_drpsf)

            print(subset.head())

            c = get_c(obj, subset)
            c_list.append(c) 

        delta_to_c[err] = c_list
     

    gal_psf_ratios_array = np.asarray(gal_psf_ratios)
    psf_gal_ratios_array = 1/gal_psf_ratios_array

    def plot_for_err(ax, err):
        c_array = delta_to_c[err]
        ax.scatter(psf_gal_ratios_array, c_array, color='m', label=fr"$\frac{{\Delta (r_{{psf}}^2)}}{{r_{{psf}}^2}}$ = {err}")

        x = np.linspace(0, np.max(psf_gal_ratios_array)* 1.1)

        ax.plot(x, -err*x*x)
        # ax.plot(psf_gal_ratios_array, -psf_gal_ratios_array*err)

        ax.set_xlabel('psf_gal_ratio')
        ax.set_ylabel('c')
        # ax.set_xticks(psf_gal_ratios_array)

        ax.legend()

    # Calculations to figure out the number of rows and columns
    total_plots = len(err_set)
    num_cols = int(np.sqrt(total_plots))

    num_rows = num_cols
    if total_plots % num_cols != num_cols**2:
        num_rows += 1
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10), constrained_layout=True)


    print(f"Total number of plots: {total_plots}") 
    print(f"num_rows: {num_rows}")
    print(f"num_cols: {num_cols}")
        
         
    row = 0
    col = 0
    for err in err_set:


        print(row, col)
        plot_for_err(axs[row][col], err)

        col += 1

        if col >= num_cols:
            col = 0
            row += 1
        


        # axs.scatter(ratios_array, example_c_array)


    plt.show()

    pdb.set_trace()



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


