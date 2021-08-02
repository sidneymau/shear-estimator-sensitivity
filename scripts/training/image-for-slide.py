import galsim
import numpy as np
import matplotlib.pyplot as plt

#----------------
# Defining parameters
gal_flux = 1.e5
gal_r0 = 2.7

# Moffat profile for PSF
psf_beta = 5
psf_re = 1.0
psf = galsim.Moffat(flux=1., beta=psf_beta, half_light_radius=psf_re)


pixel_scale = 0.2

# defining shear parameters
g1 = 0.2
g2 = 0.3

#-------------------
gal = galsim.Exponential(flux=gal_flux, scale_radius=gal_r0)
gal = gal.shear(g1=0.1, g2=0.1)
sheared = gal.shear(g1=g1, g2=g2)
convolved = galsim.Convolve(sheared, psf)

source_galaxy = gal.drawImage(scale=pixel_scale)
sheared_galaxy = sheared.drawImage(scale=pixel_scale)
convolved_galaxy = convolved.drawImage(scale=pixel_scale)

images = [source_galaxy, sheared_galaxy, convolved_galaxy]

vmax = max([np.max(image.array) for image in images])
vmin = 0

fix, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

im = axs[0].imshow(source_galaxy.array, vmin=vmin, vmax=vmax)
axs[0].set_title('Source Galaxy')

im = axs[1].imshow(sheared_galaxy.array, vmin=vmin, vmax=vmax)
axs[1].set_title('Sheared Galaxy')

im = axs[2].imshow(convolved_galaxy.array, vmin=vmin, vmax=vmax)
axs[2].set_title('Blurred Galaxy')

for ax in axs:
	ax.set_xlabel('x')
	ax.set_ylabel('y')

plt.show()