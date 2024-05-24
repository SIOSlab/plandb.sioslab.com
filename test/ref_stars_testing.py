from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import astropy.units as u
from find_ref_stars_coords import findRefStars

dec0 = 88.17
dec1 = 10.11
dec2 = -83.76
star0 = SkyCoord(
    "12h20m42.91s",
    "+17d47m35.71s",
    frame="icrs",
    unit=(u.hourangle, u.deg),
    obstime="J2000",
    pm_ra_cosdec=-109.24 * u.mas / u.yr,
    pm_dec=dec0 * u.mas / u.yr,
    distance=coord.Distance(parallax=10.71 * u.mas),
)
star1 = SkyCoord(
    "15h17m05.90s",
    "+71d49m26.19s",
    frame="icrs",
    unit=(u.hourangle, u.deg),
    obstime="J2000",
    pm_ra_cosdec=3.43 * u.mas / u.yr,
    pm_dec=dec1 * u.mas / u.yr,
    distance=coord.Distance(parallax=7.95 * u.mas),
)
star2 = SkyCoord(
    "23h31m17.80s",
    "+39d14m09.01s",
    frame="icrs",
    unit=(u.hourangle, u.deg),
    obstime="J2000",
    pm_ra_cosdec=287.29 * u.mas / u.yr,
    pm_dec=dec1 * u.mas / u.yr,
    distance=coord.Distance(parallax=13.23 * u.mas),
)


ref_coords = [star0, star1, star2]
target = SkyCoord(
    "14 39 36.49400", "-60 50 02.3737", frame="icrs", unit=(u.hourangle, u.deg)
)  # alpha centauri
stars = findRefStars(ref_stars=ref_coords, target_star=target, phi_max=150.0)
print("star indices: ", stars)
