import numpy as np
from scipy.interpolate import Akima1DInterpolator as Akima

class DensityVariations1976(object):
    def __init__(self):
        # All this stuff is in metric units
        altitiude = np.array([0.0, 4000, 8000, 10000, 16666.67, 20000, 26666.67])
        nominal_density = np.array([1.225, 0.819129, 0.525168, 0.412707, 0.148913, 0.0880349, 0.0303264])
        percent_departure = np.array([20.0, 6.0, 2.0, 6.0, 26.0, 20, 14])
        density_deviations = nominal_density * percent_departure / 100 # Since the values are percentages

        # Now the deviations indicate the 1% extremes. This means that they represent
        # 49% deviation in one dircetion, i.e.,
        sigma = 0.682689492137 / 2
        std_dev_density = sigma * density_deviations / 0.49

        # Now create an interploation usign Akima polynomial
        self.std_dev_density_interp = Akima(altitiude, std_dev_density)

    def get_density_deviations(self, altitude_profile):
        std_dev_arr = self.std_dev_density_interp(altitude_profile)
        return std_dev_arr

if __name__ == '__main__':
    density_deviation_obj = DensityVariations1976()
    densit_dev_val = density_deviation_obj.get_density_deviations(0)
    print('densit_dev_val = ', densit_dev_val)
