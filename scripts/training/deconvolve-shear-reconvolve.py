"""
1. Creates a galaxy, shears it, convolves it with the PSF
2. Deconvolves by PSF, shears again, then reconvolves by PSF
"""

import galsim
import numpy as np
import matplotlib.pyplot as plt

#----------------

# Defining parameters
gal_flux = 1.e5
gal_r0 = 2.7

# Using Moffat profile for PSF
psf_beta = 5
psf_re = 1.0

pixel_scale = 0.2

# shear "exaggerator"
scalar = 5

# first shear
shear1_g1 = 0.01 
shear1_g2 = 0.01

# second shear
shear2_g1 = 0.001
shear2_g2 = 0.001


# Defining PSFs
psf1 = galsim.Moffat(flux=1., beta=psf_beta, half_light_radius=psf_re)
psf2 = psf1
psf3 = psf1


def center_crop_array(array, shape_x, shape_y):
	y, x = array.shape
	
	x_center = (x)//2
	x_delta = (shape_x)//2

	y_center = (y)//2
	y_delta = (shape_y)//2

	print(x_center, x_delta)
	print(y_center, y_delta)
	return array[x_center - x_delta:x_center + x_delta, y_center - y_delta:y_center + y_delta]


def double_shear(psf_real, psf_deconvolve, psf_reconvolve):

	# Creating galaxy from a brightness profile
	gal = galsim.Exponential(flux=gal_flux, scale_radius=gal_r0)

	# Shearing the galaxy
	sheared = gal.shear(g1=shear1_g1, g2=shear1_g2)

	
	# Convolving the sheared galaxy with the PSF
	convolved = galsim.Convolve(sheared, psf_real)

	# Deconvolving by psf_deconvolve
	inv_psf = galsim.Deconvolve(psf_deconvolve)
	deconvolved = galsim.Convolve(convolved, inv_psf)

	# Applying second shears
	sheared_plus = deconvolved.shear(g1=shear2_g1, g2=shear2_g2)
	sheared_minus = deconvolved.shear(g1= -shear2_g1, g2= -shear2_g2)

	# Reconvolving by psf_reconvolve
	reconvolved_plus = galsim.Convolve(sheared_plus, psf_reconvolve)
	reconvolved_minus = galsim.Convolve(sheared_minus, psf_reconvolve)


	source_galaxy = gal.drawImage(scale=pixel_scale)
	sheared_galaxy = sheared.drawImage(scale=pixel_scale)
	convolved_galaxy = convolved.drawImage(scale=pixel_scale)
	deconvolved_galaxy = deconvolved.drawImage(scale=pixel_scale)

	sheared_plus_galaxy = sheared_plus.drawImage(scale=pixel_scale)
	sheared_minus_galaxy = sheared_minus.drawImage(scale=pixel_scale)
	reconvolved_plus_galaxy = reconvolved_plus.drawImage(scale=pixel_scale)
	reconvolved_minus_galaxy = reconvolved_minus.drawImage(scale=pixel_scale)

	# Measuring galaxy shape parameters
	# We want to measure the shapes of reconvolved_plus_galaxy and reconvolved_minus_galaxy
	# the documentation recommends that we use the method='no_pixel' on the images

	plus_galaxy = reconvolved_plus.drawImage(scale=pixel_scale, method='no_pixel')
	minus_galaxy = reconvolved_minus.drawImage(scale=pixel_scale, method='no_pixel')

	plus_moments = plus_galaxy.FindAdaptiveMom()
	minus_moments = minus_galaxy.FindAdaptiveMom()

	plus_shape = plus_moments.observed_shape
	minus_shape = minus_moments.observed_shape

	e1_plus = plus_shape.e1
	e2_plus = plus_shape.e2

	e1_minus = minus_shape.e1
	e2_minus = minus_shape.e2

	print('plus ellipticities: ', e1_plus, e2_plus)
	print('minus ellipticities: ', e1_minus, e2_minus)

	# calculating the shear response matrix R
	R_11 = (e1_plus - e1_minus) / (2 * shear2_g1)
	R_12 = (e2_plus - e2_minus) / (2 * shear2_g1)
	R_21 = (e1_plus - e1_minus) / (2 * shear2_g2)
	R_22 = (e2_plus - e2_minus) / (2 * shear2_g2)

	R = np.array([[R_11, R_12],[R_21, R_22]])
	print("Shear Response matrix R:\n", R)


	# Displaying with matplotlib ------------------------
	images = [source_galaxy, sheared_galaxy, convolved_galaxy, deconvolved_galaxy, 
	sheared_plus_galaxy, sheared_minus_galaxy, reconvolved_plus_galaxy, reconvolved_minus_galaxy]

	vmax = max([np.max(image.array) for image in images])
	vmin = 0


	fig, axs = plt.subplots(2, 4, figsize=(12, 4), constrained_layout=True)

	im = axs[0][0].imshow(source_galaxy.array, vmin=vmin, vmax=vmax)
	axs[0][0].set_title('Source Galaxy')

	im = axs[0][1].imshow(sheared_galaxy.array, vmin=vmin, vmax=vmax)
	axs[0][1].set_title('Sheared Galaxy')

	im = axs[0][2].imshow(convolved_galaxy.array, vmin=vmin, vmax=vmax)
	axs[0][2].set_title('Sheared and Convolved Galaxy')

	im = axs[0][3].imshow(deconvolved_galaxy.array, vmin=vmin, vmax=vmax)
	axs[0][3].set_title('Deconvolved Galaxy')

	im = axs[1][0].imshow(sheared_plus_galaxy.array, vmin=vmin, vmax=vmax)
	axs[1][0].set_title('Second Shear +')

	im = axs[1][1].imshow(reconvolved_plus_galaxy.array, vmin=vmin, vmax=vmax)
	axs[1][1].set_title('Reconvolved Shear +')

	im = axs[1][2].imshow(sheared_minus_galaxy.array, vmin=vmin, vmax=vmax)
	axs[1][2].set_title('Second Shear -')

	im = axs[1][3].imshow(reconvolved_minus_galaxy.array, vmin=vmin, vmax=vmax)
	axs[1][3].set_title('Reconvolved Shear -')

	for height in range(2):
		for width in range(4):
			axs[height][width].set_xlabel('x')
			axs[height][width].set_ylabel('y')

	cb = fig.colorbar(im, ax=axs[:], location='bottom', shrink=0.6)
	cb.set_label('Flux')

	plt.savefig('deconvolve-shear-reconvolve.png')


	# Plots that show pixel-by-pixel differences between relevant plots

	# Difference between sheared_galaxy (pre-psf convolution) and deconvolved_galaxy (post-deconvolution)

	plt.figure()

	#reshaping deconvolved_galaxy.array
	deconvolved_galaxy_reshaped_array = center_crop_array(deconvolved_galaxy.array, 232, 232)

	plt.imshow(sheared_galaxy.array - deconvolved_galaxy_reshaped_array, vmin=vmin, vmax=vmax)
	plt.title('Pixel-by-pixel comparison of pre-PSF sheared galaxy and the convolved-deconvolved version')

	plt.show()

double_shear(psf1, psf2, psf3)

