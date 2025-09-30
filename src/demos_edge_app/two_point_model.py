
# -----------------------------------------------------------------------------
# APP UNIT NOTE:
# The Streamlit app supplies "power" in MW and "upstream_density" in 10^19 m^-3.
# The app converts these to SI before calling this module.
# This model continues to expect SI units internally:
#   - power: W
#   - upstream_density: m^-3
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
from scipy.optimize import root_scalar, fsolve

BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
ELECTRIC_CHARGE = 1.602176634e-19  # C
EFFECTIVE_CHARGE = 1.
ION_MASS = 2. * 1.67e-27  # (kg)

cmap = "summer"
uncert_cmap = "autumn_r"

# Conversions
def convert_eV_to_K(temperature):
    return temperature * ELECTRIC_CHARGE / BOLTZMANN_CONSTANT

def convert_eV_to_J(temperature):
    return temperature * ELECTRIC_CHARGE

def convert_K_to_eV(temperature):
    return temperature * BOLTZMANN_CONSTANT / ELECTRIC_CHARGE

# Helper functions

def random_log_space(lwr, upr, size):
    return np.exp(np.random.uniform(*np.log([lwr, upr]), size))

def log_space(lwr, upr, num):
    return np.exp(np.linspace(*np.log([lwr, upr]), num))

class SputteringModel:

    def __init__(self,
                 S=0.042,
                 p=0.2,
                 q=1.,
                 E_th=1.e-4,  # eV
                 erosion_constant=10E24  # years m^2 s^-1
                 ):
        self.S = S  # The material relevant constant
        self.p = p
        self.q = q
        self.E_th = E_th  # The sputtering threshold energy
        self.erosion_constant = erosion_constant

    def __call__(self, te_t, ne_t,
                 request="EROSION LIFETIME",
                 ):
        if request == "EROSION LIFETIME":
            return self.erosion_lifetime(te_t=te_t, ne_t=ne_t)
        elif request == "ION FLUX":
            return self.ion_flux(te_t=te_t, ne_t=ne_t)
        elif request == "SPUTTERING RATE":
            return self.sputtering_rate(te_t=te_t, ne_t=ne_t)
        elif request == "SPUTTERING YIELD":
            return self.sputtering_yield(te_t=te_t)

    def erosion_lifetime(self, ne_t, te_t):
        ion_flux = self.ion_flux(te_t=te_t, ne_t=ne_t)
        return self.erosion_constant /ion_flux

    def ion_flux(self, ne_t, te_t):
        return ne_t * self.ion_speed(te_t)

    def ion_speed(self, te_t):
        return np.sqrt(2 * BOLTZMANN_CONSTANT * convert_eV_to_K(te_t) / ION_MASS)

    def sputtering_rate(self, ne_t, te_t):
        return self.sputtering_yield(te_t = te_t) * self.ion_flux(ne_t=ne_t, te_t=te_t)

    def sputtering_yield(self, te_t):
        e_i = self.calc_ion_energy(te_t)
        return self.S * (max(e_i - self.E_th, 0.) **self.p) / (self.E_th **self.q)

    def calc_ion_energy(self, te_t):
        return te_t

import numpy as np
from scipy.optimize import root_scalar

# Physical constants
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
ELECTRIC_CHARGE   = 1.602176634e-19  # C
ION_MASS          = 2.0 * 1.67e-27   # kg (D)
# Spitzer-like electron conduction prefactor in eV units:
KAPPA0_E_EV = 2000.0  # W·m^-1·eV^(-7/2)
    
class TwoPointModel:
    """
    Solve for target Te (eV) and nt (m^-3) using the modified two-point model:
      (1) n_u T_u = (2/f_mom) n_t T_t
      (2) T_u^(7/2) = T_t^(7/2) + (7/2) * f_cond * q_{||,t} * L_{||} / kappa0_e
      (3) q_{||} (upstream) given by geometry/power; q_{||,t} = (1 - f_pow) * q_{||}
      (4) sheath: q_{t,sheath} = gamma * n_t * T_t * c_s,t * e
    Residual: F(T_t) = q_{t,sheath}(T_t) - q_{||,t}  →  0
    """

    def __init__(self,
                 power: float = 50.0e6,                 # W
                 upstream_density: float = 7.3e19,     # m^-3
                 connection_length: float = 50.0,  # m
                 lambda_q: float = 5.0e-3,             # m (midplane heat-flux width)
                 R_m: float = 3.0,            # m
                 f_cond: float = 1.0,                  # f_c in your code
                 f_mom: float = 1.0,                   # f_mom
                 f_pow: float = 0.,           # if None, infer from n_u via logistic
                 gamma: float = 7.0,                   # sheath coefficient
                 **sputtering_kwargs
                 ):
        
        
        self.gamma = float(gamma)
        self.n_u   = float(upstream_density)
        self.L     = float(connection_length)
        self.lambda_q = float(lambda_q)
        self.major_radius = float(R_m)
        
        # knobs
        self.f_c = float(np.clip(f_cond, 1e-4, 1.0))
        self.f_m = float(np.clip(f_mom,  1e-4, 1.0))
        self.f_p = float(np.clip(f_pow, 0.0, 0.9999))

        self.q_up = self._calc_q_par(power)

        self.sputtering_model = SputteringModel(**sputtering_kwargs)

    def __call__(self):
        target_temperature, target_electron_density = self.solve_target()
        
        erosion_lifetime = self.sputtering_model(te_t=target_temperature, ne_t=target_electron_density,
                                                 request="EROSION LIFETIME")

        return erosion_lifetime

    def _calc_q_par(self, power_W: float) -> float:
        denom = max(2. * np.pi * self.lambda_q * self.major_radius, 1e-12)
        return float(power_W) / denom  # W/m^2

    # ion sound speed at target (m/s) with Te in eV
    @staticmethod
    def _c_s_t(Te_eV: float) -> float:
        return np.sqrt(ELECTRIC_CHARGE * max(Te_eV, 0.0) / ION_MASS)

    # upstream temperature from energy equation (eV)
    def _T_u_from_T_t(self, Tt_eV: float, q_t_Wm2: float) -> float:
        Tt = Tt_eV#max(Tt_eV, 0.0)
        add = (7.0/2.0) * (self.f_c * q_t_Wm2 * self.L) / KAPPA0_E_EV
        Tu_75 = Tt**3.5 + add
        return Tu_75**(2.0/7.0)
        # return max(Tu_75, 0.0)**(2.0/7.0)

    # target density from momentum relation (m^-3)
    def _n_t_from_Tt_Tu(self, Tt_eV: float, Tu_eV: float) -> float:
        Tt = Tt_eV
        return (self.f_m * self.n_u * Tu_eV) / (2.0 * Tt)

    # sheath heat flux from Tt via Tu→nt chain (W/m^2)
    def _q_sheath_from_Tt(self, Tt_eV: float, q_t_target: float) -> float:
        Tu = self._T_u_from_T_t(Tt_eV, q_t_target)
        nt = self._n_t_from_Tt_Tu(Tt_eV, Tu)
        cs = self._c_s_t(Tt_eV)
        return self.gamma * nt * Tt_eV * ELECTRIC_CHARGE * cs / (1.-self.f_p)

    # residual for root solve: q_sheath(Tt) - q_t_target = 0
    def _residual(self, Tt_eV: float, q_t_target: float) -> float:
        # print(self._q_sheath_from_Tt(Tt_eV, q_t_target)/1.e6, q_t_target/1.e6)

        return (self._q_sheath_from_Tt(Tt_eV, q_t_target) - q_t_target) / 1.e6

    def solve_target(self,
                 Tt_min: float = 1e-4,   # eV
                 Tt_max: float = 2e3,    # eV
        ) -> tuple[float, float]:
        """
        Simple solver: solve F(exp(y)) = 0 with fsolve on y = ln(T_t).
        This enforces T_t > 0 and greatly improves conditioning.
        """
        # target heat flux from power and f_pow
        q_t = (1.0 - self.f_p) * self.q_up  # W/m^2

        def G(y):
            # keep y within [ln(Tt_min), ln(Tt_max)] to avoid runaway
            y = float(np.clip(y, np.log(Tt_min), np.log(Tt_max)))
            T = float(np.exp(y))
            return self._residual(T, q_t)
        
        guesses = np.exp(np.linspace(np.log(1.e-3), np.log(1.e3), num=100))
        residual = [self._residual(g, q_t) for g in guesses]
        ind = np.argmin(np.abs(residual))
        Tt = guesses[ind]
        y_star = fsolve(G, Tt, xtol=1.e-4, maxfev=1000)[0]
        Tt = float(np.clip(np.exp(y_star), Tt_min, Tt_max))
        
        Tu = self._T_u_from_T_t(Tt, q_t)
        nt = self._n_t_from_Tt_Tu(Tt, Tu)
        return Tt, nt


def edge_simulator(**kwargs):
    # return 10E24/TwoPointModel(*args, **kwargs)()
    return TwoPointModel(
         **kwargs)()
