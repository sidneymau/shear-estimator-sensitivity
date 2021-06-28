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

# first shear
shear1_g1 = 0.1
shear1_g2 = 0.2

# second shear
shear2_g1 = -0.2
shear2_g2 = 0.1


# Creating galaxy from a brightness profile
gal = galsim.Exponential(flux=gal_flux, scale_radius=gal_r0)

# Shearing the galaxy
sheared = gal.shear(g1=shear1_g1, g2=shear1_g2)

# Defining PSF
psf = galsim.Moffat(flux=1., beta=psf_beta, half_light_radius=psf_re)

# Convolving the sheared galaxy with the PSF
convolved = galsim.Convolve(sheared, psf)

source_galaxy = gal.drawImage(scale=pixel_scale)
sheared_galaxy = sheared.drawImage(scale=pixel_scale)
convolved_galaxy = convolved.drawImage(scale=pixel_scale)



# Deconvolving with the PSF
inv_psf = galsim.Convolve(psf)
deconvolved = galsim.Convolve(convolved, inv_psf)

# Applying second shear
sheared_2 = deconvolved.shear(g1=shear2_g1, g2=shear2_g2)

# Reconvolving with the PSF
reconvolved = galsim.Convolve(sheared_2, psf)

deconvolved_galaxy = deconvolved.drawImage(scale=pixel_scale)
second_sheared_galaxy = sheared_2.drawImage(scale=pixel_scale)
reconvolved_galaxy = reconvolved.drawImage(scale=pixel_scale)

# # Writing all to a .fits file
# galsim.fits.writeMulti([source_galaxy, sheared_galaxy, convolved_galaxy, 
# 	deconvolved_galaxy, second_sheared_galaxy, reconvolved_galaxy], file_name='steps.fits')

# Displaying with matplotlib ------------------------
images = [source_galaxy, sheared_galaxy, convolved_galaxy, deconvolved_galaxy, second_sheared_galaxy, reconvolved_galaxy]

# images = {'Source Galaxy': source_galaxy, 'Sheared Galaxy': sheared_galaxy, 'Convolved Galaxy': convolved_galaxy, 
# 'Deconvolved Galaxy': deconvolved_galaxy, 'Second Shear': second_sheared_galaxy, 'Reconvolved Galaxy': reconvolved_galaxy}

vmax = max([np.max(image.array) for image in images])
vmin = 0


fig, axs = plt.subplots(2, 3, figsize=(12, 4), constrained_layout=True)

im = axs[0][0].imshow(source_galaxy.array, vmin=vmin, vmax=vmax)
axs[0][0].set_title('Source Galaxy')

im = axs[0][1].imshow(sheared_galaxy.array, vmin=vmin, vmax=vmax)
axs[0][1].set_title('Sheared Galaxy')

im = axs[0][2].imshow(convolved_galaxy.array, vmin=vmin, vmax=vmax)
axs[0][2].set_title('Sheared and Convolved Galaxy')

im = axs[1][0].imshow(deconvolved_galaxy.array, vmin=vmin, vmax=vmax)
axs[1][0].set_title('Deconvolved Galaxy')

im = axs[1][1].imshow(second_sheared_galaxy.array, vmin=vmin, vmax=vmax)
axs[1][1].set_title('Second Shear')

im = axs[1][2].imshow(reconvolved_galaxy.array, vmin=vmin, vmax=vmax)
axs[1][2].set_title('Reconvolved galaxy')

for height in range(2):
	for width in range(3):
		axs[height][width].set_xlabel('x')
		axs[height][width].set_ylabel('y')

cb = fig.colorbar(im, ax=axs[:], location='bottom', shrink=0.6)
cb.set_label('Flux')

plt.savefig('deconvolve-shear-reconvolve.png')
plt.show()



