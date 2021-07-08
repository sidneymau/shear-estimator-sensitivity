"""
Implements the basic process of metacalibration.
Calculates the R shear response matrix from a galaxy
"""

import galsim
import numpy as np
import matplotlib.pyplot as plt


# Defining parameters
gal_flux = 1.e5
gal_r0 = 2.7
gal_sigma = 2.

# Using Moffat profile for PSF
psf_beta = 5
psf_re = 1.0

pixel_scale = 0.2

# shear "exaggerator"
scalar = 5

# first shear
shear1_g1 = 0.01 
shear1_g2 = 0.01

# calibration shears
shear2_g1 = 0.001
shear2_g2 = 0.001


# Defining galaxy profile
galaxy = galsim.Gaussian(flux=gal_flux, sigma=gal_sigma)
# galaxy = galsim.Exponential(flux=gal_flux, scale_radius=gal_r0)

# Defining PSFs
psf1 = galsim.Gaussian(flux=1., sigma=1)
# psf1 = galsim.Moffat(flux=1., beta=psf_beta, half_light_radius=psf_re)
psf2 = psf1
psf3 = psf1


def metacalibration(gal, psf_real, psf_deconvolve, psf_reconvolve):

	# Shearing the galaxy
	sheared = gal.shear(g1=shear1_g1, g2=shear1_g2)

	# Convolving the sheared galaxy with the PSF
	convolved = galsim.Convolve(sheared, psf_real)

	# Deconvolving by psf_deconvolve
	inv_psf = galsim.Deconvolve(psf_deconvolve)
	deconvolved = galsim.Convolve(convolved, inv_psf)

	# Applying second shear in g1
	sheared_plus_g1 = deconvolved.shear(g1=shear2_g1, g2=0)
	sheared_minus_g1 = deconvolved.shear(g1= -shear2_g1, g2=0)

	# Applying second shear in g2
	sheared_plus_g2 = deconvolved.shear(g1=0, g2=shear2_g2)
	sheared_minus_g2 = deconvolved.shear(g1=0, g2=-shear2_g2)

	# Reconvolving by psf_reconvolve for g1
	reconvolved_plus_g1 = galsim.Convolve(sheared_plus_g1, psf_reconvolve)
	reconvolved_minus_g1 = galsim.Convolve(sheared_minus_g1, psf_reconvolve)

	# Reconvolving by psf_reconvolve for g2
	reconvolved_plus_g2 = galsim.Convolve(sheared_plus_g2, psf_reconvolve)
	reconvolved_minus_g2 = galsim.Convolve(sheared_minus_g2, psf_reconvolve)

	# Measuring galaxy shape parameters
	# We want to measure the shapes of reconvolved_plus_galaxy and reconvolved_minus_galaxy
	# the documentation recommends that we use the method='no_pixel' on the images

	plus_galaxy_g1 = reconvolved_plus_g1.drawImage(scale=pixel_scale, method='no_pixel')
	minus_galaxy_g1 = reconvolved_minus_g1.drawImage(scale=pixel_scale, method='no_pixel')

	plus_galaxy_g2 = reconvolved_plus_g2.drawImage(scale=pixel_scale, method='no_pixel')
	minus_galaxy_g2 = reconvolved_minus_g2.drawImage(scale=pixel_scale, method='no_pixel')

	plus_moments_g1 = plus_galaxy_g1.FindAdaptiveMom()
	minus_moments_g1 = minus_galaxy_g1.FindAdaptiveMom()

	plus_moments_g2 = plus_galaxy_g2.FindAdaptiveMom()
	minus_moments_g2 = minus_galaxy_g2.FindAdaptiveMom()

	plus_shape_g1 = plus_moments_g1.observed_shape
	minus_shape_g1 = minus_moments_g1.observed_shape

	plus_shape_g2 = plus_moments_g2.observed_shape
	minus_shape_g2 = minus_moments_g2.observed_shape

	e1_plus_g1 = plus_shape_g1.e1
	e2_plus_g1 = plus_shape_g1.e2

	e1_minus_g1 = minus_shape_g1.e1
	e2_minus_g1 = minus_shape_g1.e2

	e1_plus_g2 = plus_shape_g2.e1
	e2_plus_g2 = plus_shape_g2.e2

	e1_minus_g2 = minus_shape_g2.e1
	e2_minus_g2 = minus_shape_g2.e2

	# calculating the shear response matrix R
	R_11 = (e1_plus_g1 - e1_minus_g1) / (2 * shear2_g1)
	R_12 = (e2_plus_g1 - e2_minus_g1) / (2 * shear2_g1)
	R_21 = (e1_plus_g2 - e1_minus_g2) / (2 * shear2_g2)
	R_22 = (e2_plus_g2 - e2_minus_g2) / (2 * shear2_g2)

	R = np.array([[R_11, R_12],[R_21, R_22]])
	print("Shear Response matrix R:\n", R)


metacalibration(galaxy, psf1, psf2, psf3)

