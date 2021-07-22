"""
Implements the basic process of metacalibration.
Calculates the R shear response matrix from a galaxy
"""

import galsim
import numpy as np



def generate_observed_galaxy(source_galaxy, psf_blur, lensing_g1, lensing_g2):
	"""
	Takes in a source_galaxy and psf_blur GalSim objects as well as the parameters
	g1 and g2 of the lensing shear and generates a new GalSim object representing
	an observed galaxy
	"""
	# shearing the original galaxy
	sheared = source_galaxy.shear(g1=lensing_g1, g2=lensing_g2)

	# Convolving the sheared galaxy with the PSF
	observed = galsim.Convolve(sheared, psf_blur)

	return observed


def delta_shear(observed_gal, psf_deconvolve, psf_reconvolve, delta_g1, delta_g2):
	"""
	Takes in an observed galaxy object, two PSFs for metacal (deconvolving
	and re-convolving), and the amount by which to shift g1 and g2, and returns
	a tuple of tuples of modified galaxy objects.
	((g1plus, g1minus), (g2plus, g2minus))
	"""
	# Deconvolving by psf_deconvolve
	inv_psf = galsim.Deconvolve(psf_deconvolve)
	deconvolved = galsim.Convolve(observed_gal, inv_psf)

	# Applying second shear in g1
	sheared_plus_g1 = deconvolved.shear(g1=delta_g1, g2=0)
	sheared_minus_g1 = deconvolved.shear(g1=-delta_g1, g2=0)

	# Applying second shear in g2
	sheared_plus_g2 = deconvolved.shear(g1=0, g2=delta_g2)
	sheared_minus_g2 = deconvolved.shear(g1=0, g2=-delta_g2)

	# Reconvolving by psf_reconvolve for g1
	reconvolved_plus_g1 = galsim.Convolve(sheared_plus_g1, psf_reconvolve)
	reconvolved_minus_g1 = galsim.Convolve(sheared_minus_g1, psf_reconvolve)

	g1_plus_minus = (reconvolved_plus_g1, reconvolved_minus_g1)


	# Reconvolving by psf_reconvolve for g2
	reconvolved_plus_g2 = galsim.Convolve(sheared_plus_g2, psf_reconvolve)
	reconvolved_minus_g2 = galsim.Convolve(sheared_minus_g2, psf_reconvolve)

	g2_plus_minus = (reconvolved_plus_g2, reconvolved_minus_g2)

	# g1_plus_minus = (sheared_plus_g1, sheared_minus_g1)
	# g2_plus_minus = (sheared_plus_g2, sheared_minus_g2)

	return g1_plus_minus, g2_plus_minus


def shear_response(g1_plus_minus, g2_plus_minus, delta_g1, delta_g2, psf_reconvolve, pixel_scale=0.2):
	"""
	Takes in the a tuple of the g1 plus/minus objects and
	a tuple of the g2 plus/minus objects and returns the
	shear response matrix R
	"""

	plus_g1_gal = g1_plus_minus[0]
	minus_g1_gal = g1_plus_minus[1]
	plus_g2_gal = g2_plus_minus[0]
	minus_g2_gal = g2_plus_minus[1]

	# Measuring galaxy shape parameters
	# We want to measure the shapes of reconvolved_plus_galaxy and reconvolved_minus_galaxy
	# the documentation recommends that we use the method='no_pixel' on the images

	plus_g1 = plus_g1_gal.drawImage(scale=pixel_scale, method='no_pixel')
	minus_g1 = minus_g1_gal.drawImage(scale=pixel_scale, method='no_pixel')

	plus_g2 = plus_g2_gal.drawImage(scale=pixel_scale, method='no_pixel')
	minus_g2 = minus_g2_gal.drawImage(scale=pixel_scale, method='no_pixel')

	# plus_moments_g1 = plus_g1.FindAdaptiveMom()
	# minus_moments_g1 = minus_g1.FindAdaptiveMom()
	#
	# plus_moments_g2 = plus_g2.FindAdaptiveMom()
	# minus_moments_g2 = minus_g2.FindAdaptiveMom()

	psf_reconvolve_image = psf_reconvolve.drawImage(scale=pixel_scale)

	plus_moments_g1 = galsim.hsm.EstimateShear(plus_g1, psf_reconvolve_image)
	minus_moments_g1 = galsim.hsm.EstimateShear(minus_g1, psf_reconvolve_image)
	plus_moments_g2 = galsim.hsm.EstimateShear(plus_g2, psf_reconvolve_image)
	minus_moments_g2 = galsim.hsm.EstimateShear(minus_g2, psf_reconvolve_image)

	# plus_shape_g1 = plus_moments_g1.observed_shape
	# minus_shape_g1 = minus_moments_g1.observed_shape
	#
	# plus_shape_g2 = plus_moments_g2.observed_shape
	# minus_shape_g2 = minus_moments_g2.observed_shape

	e1_plus_g1 = plus_moments_g1.corrected_e1
	e2_plus_g1 = plus_moments_g1.corrected_e2

	e1_minus_g1 = minus_moments_g1.corrected_e1
	e2_minus_g1 = minus_moments_g1.corrected_e2

	e1_plus_g2 = plus_moments_g2.corrected_e1
	e2_plus_g2 = plus_moments_g2.corrected_e2

	e1_minus_g2 = minus_moments_g2.corrected_e1
	e2_minus_g2 = minus_moments_g2.corrected_e2

	# calculating the shear response matrix R
	R_11 = (e1_plus_g1 - e1_minus_g1) / (2 * delta_g1)
	R_12 = (e2_plus_g1 - e2_minus_g1) / (2 * delta_g1)
	R_21 = (e1_plus_g2 - e1_minus_g2) / (2 * delta_g2)
	R_22 = (e2_plus_g2 - e2_minus_g2) / (2 * delta_g2)

	R = np.array([[R_11, R_12],[R_21, R_22]])

	return R


def metacalibration(observed_galaxy_profile, true_psf, psf_deconvolve, psf_reconvolve, delta_g1, delta_g2):
	"""
	Takes in an observed galaxy profile, the deconvolution and reconvolution PSFs,
	and the amounts by which to vary g1 and g2, then performs metacalibration based
	on these parameters. The function prints and returns the shear response matrix R
	as a numpy array.

	tuples are added to the results list
	"""
	g1pm, g2pm = delta_shear(observed_galaxy_profile, psf_deconvolve, psf_reconvolve, delta_g1, delta_g2)
	R = shear_response(g1pm, g2pm, delta_g1, delta_g2, psf_reconvolve)

	# helps to see that the multiprocessing is running
	print(R)

	return (observed_galaxy_profile, true_psf, psf_deconvolve, psf_reconvolve, delta_g1, delta_g2, R)


def main():

	# Defining parameters
	gal_flux = 1.e5
	gal_r0 = 2.7
	gal_sigma = 2.

	# first shear
	lensing_shear_g1 = 0.1
	lensing_shear_g2 = 0.2

	# calibration shears
	d_g1 = 0.001
	d_g2 = 0.001

	# Defining PSFs
	psf1 = galsim.Gaussian(flux=1., sigma=1)
	# psf1 = galsim.Moffat(flux=1., beta=psf_beta, half_light_radius=psf_re)
	psf2 = psf1
	psf3 = psf1

	# Defining galaxy profile
	galaxy = galsim.Gaussian(flux=gal_flux, sigma=gal_sigma)
	# galaxy = galsim.Exponential(flux=gal_flux, scale_radius=gal_r0)

	observed = generate_observed_galaxy(galaxy, psf1, lensing_shear_g1, lensing_shear_g2)
	metacalibration(observed, psf2, psf3, d_g1, d_g2)

if __name__ == '__main__':
	main()