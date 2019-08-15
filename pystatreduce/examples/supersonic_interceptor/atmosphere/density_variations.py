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
    nominal_altitude = np.array([[  100.        ],
                                 [  100.        ],
                                 [  158.71154193],
                                 [  208.26354802],
                                 [  581.54820633],
                                 [ 1885.33065083],
                                 [ 2508.96598669],
                                 [ 3886.01157759],
                                 [ 5531.48403351],
                                 [ 5938.86520482],
                                 [ 6854.36076941],
                                 [ 8019.53950214],
                                 [ 8362.38092809],
                                 [ 8902.14566076],
                                 [ 9016.16316281],
                                 [ 8875.61192646],
                                 [ 8401.84220327],
                                 [ 7648.20166125],
                                 [ 7461.23794421],
                                 [ 7181.56651798],
                                 [ 7112.68294527],
                                 [ 7157.94912386],
                                 [ 7319.9111466 ],
                                 [ 7610.00221154],
                                 [ 7699.66347111],
                                 [ 7905.03567485],
                                 [ 8180.61093254],
                                 [ 8266.38511845],
                                 [ 8458.44557179],
                                 [ 8704.9284588 ],
                                 [ 8775.95403863],
                                 [ 8916.74233772],
                                 [ 9048.63089773],
                                 [ 9072.17357963],
                                 [ 9091.93587063],
                                 [ 9183.14534263],
                                 [ 9268.69147944],
                                 [ 9760.70248235],
                                 [11248.9293303 ],
                                 [11946.20490623],
                                 [13769.19026677],
                                 [16426.53699354],
                                 [17196.47753757],
                                 [18706.73938156],
                                 [19907.06151334],
                                 [20000.        ]])
    densit_dev_val = density_deviation_obj.get_density_deviations(np.squeeze(nominal_altitude, axis=1))
    print('densit_dev_val = ', densit_dev_val)
