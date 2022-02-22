import astropy.units as u
import numpy as np
import radvel.orbit as rvo
from astropy.time import Time


class planet:
    """
    This interprets a block of text from radvel and creates the keplerian planet
    The format of the text input is

    ****** 14 Her ******
    Planet b:
    - Period: 1768.84 +/- 0.92/0.9 days
    - Eccentricity: 0.3647 +/- 0.0047
    - Argument of Periastron: 0.406+/- 0.011 radians
    - Time of Periastron: 2456675.2 +/- 2.5 JD
    - Time of Inferior Conjunction: 2456836.3 +/- 2.1 JD
    - Msini: 4.94 +/- 0.15 Mjup

    """

    def __init__(self, planet_text):
        self.planet_id = f'{planet_text[0].strip("*").strip()} {planet_text[1].strip(":").split()[1]}'
        period_vals = self.get_vals(planet_text[2])
        self.T = period_vals[0] * u.d
        self.T_err = period_vals[1] * u.d

        ecc_vals = self.get_vals(planet_text[3])
        self.e = ecc_vals[0]
        self.e_err = ecc_vals[1]

        w_vals = self.get_vals(planet_text[4])
        self.w_s = (w_vals[0] * u.rad) % (2 * np.pi * u.rad)
        self.w = (self.w_s + np.pi * u.rad) % (2 * np.pi * u.rad)
        self.w_err = w_vals[1] * u.rad

        # T_p_vals = self.get_vals(planet_text[5])
        # self.T_p = Time(T_p_vals[0], format="jd")
        # self.T_p_err = T_p_vals[1] * u.d

        T_c_vals = self.get_vals(planet_text[6])
        self.T_c = Time(T_c_vals[0], format="jd")
        self.T_c_err = T_c_vals[1] * u.d

        # Calculating the time of periastron since it is not always given
        self.T_p = rvo.timetrans_to_timeperi(self.T_c, self.T, self.e, self.w.value)
        self.T_p_err = self.T_c_err

        if "Mjup" in planet_text[7]:
            Msini_unit = u.M_jup
        elif "Mear" in planet_text[7]:
            Msini_unit = u.M_earth
        else:
            raise ValueError(f"No valid mass unit provided for planet {self.planet_id}")
        Msini_vals = self.get_vals(planet_text[7])
        self.Msini = Msini_vals[0] * Msini_unit
        self.Msini_err = Msini_vals[1] * Msini_unit

        a_vals = self.get_vals(planet_text[8])
        self.a = a_vals[0] * u.AU
        self.a_err = a_vals[1] * u.AU

    def get_vals(self, line):
        # Split up the line for analysis
        split_line = line.split()
        units = ["radians", "JD", "days", "Mjup", "AU", "Mear"]
        if all(unit not in line for unit in units):
            # When there isn't a unit the values are offset
            unit_offset = 1
        else:
            unit_offset = 0

        # Start by making sure that there's error included
        if "+/-" not in line:
            # If no error is reported then we return the fitted value and nans
            # for error
            value = float(split_line[-2 + unit_offset])
            errors = [np.nan, np.nan]
        else:
            # In this case he have an error given
            if "/" in split_line[-2 + unit_offset]:
                # When the error is not symmetric it reports them separated
                # by a slash
                error_plus = float(split_line[-2 + unit_offset].split("/")[0])
                error_minus = float(split_line[-2 + unit_offset].split("/")[1])
                errors = [error_plus, error_minus]
            else:
                error_sym = float(split_line[-2 + unit_offset])
                errors = [error_sym, error_sym]
            value = float(split_line[-4 + unit_offset])
        return value, errors

if __name__ == "__main__":
    raw_text = np.genfromtxt("planets_updated_param.txt", dtype=str, delimiter="\n")
    planets = []
    for line_num, line_text in enumerate(raw_text):
        if "******" in line_text:
            planet_text = raw_text[line_num : line_num + 9]
            planets.append(planet(planet_text))
