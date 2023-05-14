import numpy as np
# from scipy.signal import lti
# import math

# class Filter:
#     def __init__(self, num, den):
#         """
#         Wrapper for the scipy LTI system class.
#         :param num: numerator of transfer function
#         :param den: denominator of transfer function
#         """
#         self.filter = lti(num, den)
#         self.x = None
#         self.y = None
#         self.t = None

#     def simulate(self, u, t):
#         """
#         Simulate filter
#         :param u: filter input
#         :param t: time steps for which to simulate
#         :return: filter output
#         """
#         if self.x is None:
#             x_0 = None
#         else:
#             x_0 = self.x[-1]

#         self.t, self.y, self.x = self.filter.output(U=u, T=t, X0=x_0)

#         return self.y

#     def reset(self):
#         """
#         Reset filter
#         :return:
#         """
#         self.x = None
#         self.y = None
#         self.t = None


# class DrydenGustModel:
#     def __init__(self, dt, b, V_a=15, intensity=None):
#         """
#         Python realization of the continuous Dryden Turbulence Model (MIL-F-8785C).
#         :param dt: (float) band-limited white noise input sampling time.
#         :param b: (float) wingspan of aircraft
#         :param h: (float) Altitude of aircraft
#         :param V_a: (float) Airspeed of aircraft
#         :param intensity: (str) Intensity of turbulence, one of ["light", "moderate", "severe"]
#         """
#         # For fixed (nominal) altitude and airspeed

#         if intensity == "light":
#           sigma_u = 1.06
#           sigma_w = 0.7
#         elif intensity == "moderate":
#           sigma_u = 2.12
#           sigma_w = 1.4
#         else:
#             raise ValueError("Intensity must be one of ['light', 'moderate']")          

#         # Convert meters to feet and follow MIL-F-8785C spec
#         # h = h * meters2feet
#         # b = b * meters2feet
#         # V_a = V_a * meters2feet
#         # W_20 = W_20 * meters2feet

#         # Turbulence intensities
#         # sigma_w = 0.1 * W_20
#         # sigma_u = sigma_w / (0.177 + 0.000823 * h) ** 0.4
        

#         # Turbulence length scales
#         # L_u = h / (0.177 + 0.000823 * h) ** 1.2
#         # L_w = h
#         L_u = 200
#         L_w = 50

#         K_u = sigma_u * math.sqrt((2 * L_u) / (math.pi * V_a))
#         K_w = sigma_w * math.sqrt((L_w) / (math.pi * V_a))

#         T_u = L_u / V_a
#         T_w1 = math.sqrt(3.0) * L_w / V_a
#         T_w2 = L_w / V_a

#         # T_q = T_p
#         # T_r = 3 * b / (math.pi * V_a)

#         self.filters = {"H_u": Filter( K_u, [T_u, 1]),
#                         "H_w": Filter([K_w * T_w1, K_w], [T_w2 ** 2, 2 * T_w2, 1]),
#                         }

#         self.np_random = None
#         self.seed(0)

#         self.dt = dt
#         self.sim_length = 0

#         self.noise = None

#         self.vel_lin = None

#     def seed(self, seed=None):
#         """
#         Seed the random number generator.
#         :param seed: (int) seed.
#         :return:
#         """
#         self.np_random = np.random.RandomState(seed)

#     def _generate_noise(self, size):
#         return np.sqrt(np.pi / self.dt) * self.np_random.standard_normal(size=(2, size))

#     def reset(self, noise=None):
#         """
#         Reset model.
#         :param noise: (np.array) Input to filters, should be four sequences of Gaussianly distributed numbers.
#         :return:
#         """
#         self.vel_lin = None
#         self.sim_length = 0

      
#         noise = noise * math.sqrt(math.pi / self.dt)
#         self.noise = noise

#         for filter in self.filters.values():
#             filter.reset()

#     def simulate(self, length):
#         """
#         Simulate turbulence by passing band-limited Gaussian noise of length length through the shaping filters.
#         :param length: (int) the number of steps to simulate.
#         :return:
#         """
#         t_span = [self.sim_length, self.sim_length + length]

#         t = np.linspace(t_span[0] * self.dt, t_span[1] * self.dt, length)

#         if self.noise is None:
#             noise = self._generate_noise(t.shape[0])
#         else:
#             if self.noise.shape[-1] >= t_span[1]:
#                 noise = self.noise[:, t_span[0]:t_span[1]]
#             else:
#                 noise_start_i = t_span[0] % self.noise.shape[-1]
#                 remaining_noise_length = self.noise.shape[-1] - noise_start_i
#                 if remaining_noise_length >= length:
#                     noise = self.noise[:, noise_start_i:noise_start_i + length]
#                 else:
#                     if length - remaining_noise_length > self.noise.shape[-1]:
#                         concat_noise = np.pad(self.noise,
#                                               ((0, 0), (0, length - remaining_noise_length - self.noise.shape[-1])),
#                                               mode="wrap")
#                     else:
#                         concat_noise = self.noise[:, :length - remaining_noise_length]
#                     noise = np.concatenate((self.noise[:, noise_start_i:], concat_noise), axis=-1)

#         vel_lin = np.array([self.filters["H_u"].simulate(noise[0], t),
#                             self.filters["H_w"].simulate(noise[1], t)])


#         if self.vel_lin is None:
#             self.vel_lin = vel_lin
#         else:
#             self.vel_lin = np.concatenate((self.vel_lin, vel_lin), axis=1)

#         self.sim_length += length
#     # def plot_dryden_gusts(self, duration):
#     #       """
#     #       Generate Dryden gusts and plot the results.
#     #       :param duration: (float) duration of the simulation in seconds.
#     #       """
#     #       num_steps = int(duration / self.dt)
#     #       self.simulate(num_steps)

#     #       t = np.arange(0, duration, self.dt)

#     #       fig, axs = plt.subplots(2, 1, figsize=(10, 8))

#     #       axs[0].plot(t, self.vel_lin[0], label="u")
#     #       # axs[0].plot(t, self.vel_lin[1], label="v")
#     #       axs[0].plot(t, self.vel_lin[1], label="w")
#     #       axs[0].set_title("Linear velocities")
#     #       axs[0].set_xlabel("Time [s]")
#     #       axs[0].set_ylabel("Velocity [m/s]")
#     #       axs[0].legend()

#     #       # axs[1].plot(t, self.vel_ang[0], label="p")
#     #       axs[1].plot(t, self.vel_ang[0], label="q")
#     #       # axs[1].plot(t, self.vel_ang[2], label="r")
#     #       axs[1].set_title("Angular velocities")
#     #       axs[1].set_xlabel("Time [s]")
#     #       axs[1].set_ylabel("Angular velocity [rad/s]")
#     #       axs[1].legend()

#     #       plt.tight_layout()
#     #       plt.show()

# class SteadyWind:

#     def __init__(self):
#         """
#         Steady wind model.
#         :param wind_speed: (float) wind speed in m/s.
#         :param wind_direction: (float) wind direction in radians.
#         :param wind_gradient: (float) wind gradient in 1/s.
#         """
#         self.wind_range = (0, 8)

        

#     def update(self):
#         """
#         Simulate wind.
#         :param t: (float) time in seconds.
#         :return: (np.array) wind velocity in NED frame.
#         """
#         self.wind_speed = np.random.uniform(*self.wind_range)


class DrydenGustModel:
    def __init__(self, dt = 0.01, V_a=15, intensity='light', total_time=60.0):
        self.dt = dt
        Va = V_a
        self.Va = Va
        self.total_time = total_time
        self.intensity = intensity
        self._gust_state = np.zeros((2, 1))
        self.gust = np.zeros((1, 2))
        self.np_random = np.random.default_rng()
        if self.intensity == 'light':
            self.sigma_w = 0.7
            self.sigma_u = 1.06
        elif self.intensity == 'moderate':
            self.sigma_u = 2.12
            self.sigma_w =  1.4
        else:
            self.sigma_w = 0.0
            self.sigma_u = 0.0

        self.Lu = 200
        self.Lw = 50

        self._A = np.array([[-Va/self.Lu, 0],
                            [0, -2*(Va/self.Lw)]])
        self._B = np.array([[1], [1]])
        self._C = np.array([[self.sigma_u * np.sqrt((2*Va)/(np.pi * self.Lu)), 0],
                            [0, self.sigma_w * np.sqrt((3*Va)/(np.pi * self.Lw))]])

        num_steps = int(total_time / dt)
        self.vel_lin = None
        self.precomputed_w = np.sqrt(np.pi / dt) * self.np_random.standard_normal((2, num_steps))

    def update(self, step):
        if self.intensity == 'none':
            return np.array([[0.0, 0.0]])

        w = self.precomputed_w[:, step].reshape(-1, 1)

        self._gust_state += self.dt * (self._A @ self._gust_state + self._B * w)
        _gusts = self._C @ self._gust_state

        self.gust = np.array([[_gusts.item(0), _gusts.item(1)]])

        return self.gust
    

    
    def simulate(self):
        """ Return an array of gusts """
        num_steps = int(self.total_time / self.dt)
        gusts = np.zeros((num_steps, 2))
        for i in range(num_steps):
            gusts[i] = self.update(i)
        self.vel_lin = gusts.T

