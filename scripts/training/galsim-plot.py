"""
Plot galaxy images at different stages
"""

import galsim
import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------

# Define various parameters
gal_flux = 1.e5    # counts
gal_r0 = 2.7       # arcsec
g1 = 0.1           #
g2 = 0.2           #
psf_beta = 5       #
psf_re = 1.0       # arcsec
pixel_scale = 0.2  # arcsec / pixel

# Fix random seed for reproducability
random_seed = 1534225
rng = galsim.BaseDeviate(random_seed+1)

# Define the brightness profile of some galaxy
gal = galsim.Exponential(flux=gal_flux, scale_radius=gal_r0)

# Shear the galaxy
sheared = gal.shear(g1=g1, g2=g2)

# Define the PSF
psf = galsim.Moffat(beta=psf_beta, flux=1., half_light_radius=psf_re)

# Convolve the galaxy with the PSF
convolved = galsim.Convolve(psf, sheared)

# Draw images
image1 = gal.drawImage(scale=pixel_scale)
image2 = sheared.drawImage(scale=pixel_scale)
image3 = convolved.drawImage(scale=pixel_scale)

# Find a dynamic range suitable for all images
# We can access the "raw" numbers of each image with the .array property
vmin = 0
vmax = np.max([np.max(image1.array), np.max(image2.array), np.max(image3.array)])

# Plot using matplotlib
fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

im = axs[0].imshow(image1.array, vmin=vmin, vmax=vmax)
axs[0].set_title('Source Galaxy')

im = axs[1].imshow(image2.array, vmin=vmin, vmax=vmax)
axs[1].set_title('Sheared Galaxy')

im = axs[2].imshow(image3.array, vmin=vmin, vmax=vmax)
axs[2].set_title('Convolved & Sheared Galaxy')

for ax in axs:
    ax.set_xlabel('x')
    ax.set_ylabel('y')

cb = fig.colorbar(im, ax=axs[:], location='bottom', shrink=0.6)
cb.set_label('Flux')

plt.savefig('galsim-plot.png')
plt.show()
