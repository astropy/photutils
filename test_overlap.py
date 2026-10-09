from astropy.table import QTable
from photutils.datasets import make_model_image
from photutils.psf import CircularGaussianPRF

model = CircularGaussianPRF(fwhm=2.0)
params = QTable({'x_0': [50.0, -16.0], 'y_0': [50.0, 50.0], 'flux': [100.0, 100.0]})
img = make_model_image((100, 100), model, params, model_shape=(31, 31))
print("Success!")
