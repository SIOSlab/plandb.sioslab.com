from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time
from pointing_code import pointingAngles

# main method - testing

tgt_ra = "08h00m00s"  # example coords
tgt_dec = "60d00m00s"  # example coords

# tgt_ra='12h00m00s'#example coords
# tgt_dec='40d00m00s' #example coords

# other cases tested
# tgt_ra='16h00m00s'#example coords
# tgt_dec='-50d00m10s' #example coords
# tgt_ra='22h00m00s'#example coords
# tgt_dec='70d00m00s' #example coords
tgt_dist = 10

tgt = SkyCoord(ra=tgt_ra, dec=tgt_dec, distance=tgt_dist * u.au, frame="icrs")
time = new_obstime = Time.now()
[r, p, y, sun_angle, sunx, suny, sunz] = pointingAngles(tgt, time)

# print outputs
print(" roll=", r, "deg")
print(" pitch=", p, "deg")
print(" yaw=", y, "deg")
print(" sun angle=", sun_angle, "deg")
print("sunx, suny, sunz=   %.1f  %.1f   %.1f" % (sunx, suny, sunz))
