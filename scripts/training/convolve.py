"""
Convole a galaxy with a PSF
"""

import galsim

#-------------------------------------------------------------------------------

# Define various parameters
gal_flux = 1.e5    # counts
gal_r0 = 2.7       # arcsec
g1 = 0.1           #
g2 = 0.2           #
psf_beta = 5       #
psf_re = 1.0       # arcsec
pixel_scale = 0.2  # arcsec / pixel
sky_level = 2.5e3  # counts / arcsec^2

# Fix random seed for reproducability
random_seed = 1534225
rng = galsim.BaseDeviate(random_seed+1)

# Define the brightness profile of some galaxy
gal = galsim.Exponential(flux=gal_flux, scale_radius=gal_r0)

# Define the PSF
psf = galsim.Moffat(beta=psf_beta, flux=1., half_light_radius=psf_re)

# Convolve the galaxy with the PSF
convolved = galsim.Convolve(psf, gal)

# Draw images
image1 = gal.drawImage(scale=pixel_scale)
image2 = convolved.drawImage(scale=pixel_scale)
image3 = psf.drawImage(scale=pixel_scale)

# Write to a FITS file
galsim.fits.writeMulti([image1, image2, image3], file_name='convolve.fits')
