import numpy as np
from astropy.coordinates import get_body

# set up matrices to convert from inertial coordinates to local SC coordinates

"""equatorial to ecliptic (about ICRS x)
C0= [1      0       0
    0       cos     sin
    0       -sin    cos]
"""

""" roll
C1= [1      0       0
    0       cos     sin
    0       -sin    cos]

    
"""

""" pitch
C2= [cos      0       -sin
    0         1        0
    sin       0        cos]

"""

""" yaw
C3= [cos      sin       0
    -sin      cos       0
     0        0         1]

"""


"""order of rotations:
    pitch1
    roll2
    yaw3
    pitch4"""


def pointingAngles(tgt_in, time, SC_loc=None):
    """Outputs rotation angles needed for spacecraft to line up the solar array with the sun and the boresight with the target

    Args:
        tgt: a SkyCoord instance representing
                    reference star coordinates
        time: astropy Time instance representing the time of observation
        SC_loc: a SkyCoord instance of spacecraft location (when set to none, default is L2 position)

    Returns: yaw, pitch, roll in degrees (measured from zero point orientation with local z direction aligned with sun-SC vector)
        angular separation between sun and target
        each component of sun-SC vector in body frame
    """

    # get bodies
    # obstime=new_obstime=Time.now() #time for testing
    obstime = time
    sun_coord = get_body("Sun", obstime)  # get sun location GCRS
    sun_coord = sun_coord.transform_to("icrs")  # sun location ICRS
    earth_coord = get_body("Earth", obstime)  # get earth location GCRS
    earth_coord = earth_coord.transform_to("icrs")  # sun location ICRS

    # convert tgt coordinates to Cartesian ecliptic
    tgt_ec = tgt_in.transform_to("geocentricmeanecliptic").represent_as(
        "cartesian"
    )  # tgt in geocentric mean ecliptic
    tgt_ec = np.array([tgt_ec.x.value, tgt_ec.y.value, tgt_ec.z.value])

    r_eo = earth_coord.transform_to("geocentricmeanecliptic").represent_as(
        "cartesian"
    )  # earth wrt mean earth vector tgt in geocentric mean ecliptic
    r_so = sun_coord.transform_to("geocentricmeanecliptic").represent_as(
        "cartesian"
    )  # sun wrt mean earth vector in geocentric mean ecliptic
    r_eo = np.array([r_eo.x.value, r_eo.y.value, r_eo.z.value])  # in ecliptic
    r_so = np.array([r_so.x.value, r_so.y.value, r_so.z.value])  # in ecliptic

    r_es = np.subtract(r_eo, r_so)  # get vector of earth wrt sun

    if SC_loc is None:  # if no SC location specified, assume at L2
        # calculate L2
        l2e_dist = 1.5e6 / 149597870.691  # 1.5 million kilometers to AU
        # could calculate more exactly using fsolve in matlab

        # get components of L2/S vector
        r_l2s_x = r_es[0] + (l2e_dist * r_es[0] / np.linalg.norm(r_es))
        r_l2s_y = r_es[1] + (l2e_dist * r_es[1] / np.linalg.norm(r_es))
        r_l2s_z = r_es[2] + (l2e_dist * r_es[2] / np.linalg.norm(r_es))
        r_l2s = np.array([r_l2s_x, r_l2s_y, r_l2s_z])
        r_l2o = r_l2s + r_so  # L2/O vector (O is mean Earth)
        r_sc_o = r_l2o  # SC/O vector (O is mean Earth)
    else:  # if SC location specified
        SC_loc = SC_loc.represent_as("cartesian").transform_to(
            "geocentricmeanecliptic"
        )  # SC location in ecliptic coordinates wrt to mean Earth
        r_sc_o = [SC_loc.x.value, SC_loc.y.value, SC_loc.z.value]

    # first 2 maneuvers: get to zero point orientation (z aligned with r_(sun/SC) line )- rotates around Y, then X’, then target pointing (rotate around Z” and Y’’’)

    r_sunsc_norm = (r_so - r_sc_o) / np.linalg.norm(
        r_so - r_sc_o
    )  # unit vector from sun to S/C

    # unit vectors - geocentric mean ecliptic frame coordinates
    xec = np.array([1, 0, 0])
    yec = np.array([0, 1, 0])
    zec = np.array([0, 0, 1])

    # angle of rotation about y hat (pitch)
    v1 = zec
    b = r_sunsc_norm - np.dot(r_sunsc_norm, yec) * yec
    v2 = b / np.linalg.norm(
        b
    )  # unit vector of sun-SC vector projected onto local x-z plane
    v3 = yec

    # make column vectors
    v1 = v1[:, np.newaxis]
    v2 = v2[:, np.newaxis]
    v3 = v3[:, np.newaxis]

    # a1-rotate about y by angle between e3 and projection of r_sun/SC onto e1-e3 plane
    a1 = np.arctan2(
        -1
        * np.linalg.norm(np.cross(np.transpose(v1), np.transpose(v2)))
        * np.sign(np.linalg.det(a=np.hstack((v1, v2, -1 * v3)))),
        np.dot(np.transpose(v1).flatten(), np.transpose(v2).flatten()),
    )
    ang = a1
    mat_C2 = np.array(
        [[np.cos(ang), 0, -np.sin(ang)], [0, 1, 0], [np.sin(ang), 0, np.cos(ang)]]
    )  # rotation matrix

    r_sunsc_normp = np.matmul(
        mat_C2, r_sunsc_norm
    )  # unit vector of sun-SC in current SC body coordinates

    # a2-rotate about x' by angle between r_sun/SC and z' for local z to point to sun
    v1 = np.array([0, 0, 1])
    v2 = r_sunsc_normp
    v3 = np.array([1, 0, 0])
    v1 = v1[:, np.newaxis]
    v2 = v2[:, np.newaxis]
    v3 = v3[:, np.newaxis]
    a2 = np.arctan2(
        -1
        * np.linalg.norm(np.cross(np.transpose(v1), np.transpose(v2)))
        * np.sign(np.linalg.det(a=np.hstack((v1, v2, -v3)))),
        np.dot(np.transpose(v1).flatten(), np.transpose(v2).flatten()),
    )

    # angle of rotation about x hat (roll)
    ang = a2
    """C1= [1      0       0
    0       cos     sin
    0       -sin    cos]"""
    mat_C1 = np.array(
        [[1, 0, 0], [0, np.cos(ang), np.sin(ang)], [0, -np.sin(ang), np.cos(ang)]]
    )

    # AT ZERO POINT ORIENTATION NOW
    # z hat is aligned with sun-SC line (as seen in Figure 3)
    # transition matrix (global to local) = C1(a2)C2(a1)

    #############################################################################################################

    # get observatory-star look vector
    # r_t/G=r_t/O-r_G/O
    r_obs_star = np.subtract(tgt_ec, r_sc_o)
    r_obs_starloc = np.matmul(mat_C1, np.matmul(mat_C2, r_obs_star)) / np.linalg.norm(
        r_obs_star
    )  # convert SC-star vector to local SC coordinates

    # a3 and a4: align boresight with look vector

    # a3: rotate about z by yaw angle (angle between x and projection of r_t/G onto x-y plane)
    v1 = np.array([1, 0, 0])
    b = r_obs_starloc - (
        np.dot(r_obs_starloc, np.array([0, 0, 1])) * np.array([0, 0, 1])
    )
    v2 = b / np.linalg.norm(
        b
    )  # unit vector of star-S/C vector projected onto local x-y plane
    v3 = np.array([0, 0, 1])
    v1 = v1[:, np.newaxis]
    v2 = v2[:, np.newaxis]
    v3 = v3[:, np.newaxis]
    a3 = -1 * np.arctan2(
        np.linalg.norm(np.cross(np.transpose(v1), np.transpose(v2)))
        * np.sign(np.linalg.det(a=np.hstack((v1, v2, -v3)))),
        np.dot(np.transpose(v1).flatten(), np.transpose(v2).flatten()),
    )
    ang = a3
    mat_C3 = np.array(
        [[np.cos(ang), np.sin(ang), 0], [-np.sin(ang), np.cos(ang), 0], [0, 0, 1]]
    )

    r_obs_starloc3 = np.matmul(mat_C3, r_obs_starloc)

    # a4: rotate about y_hat with pitch angle equal to angle between x hat and r_t/G
    v1 = np.array([1, 0, 0])
    v2 = r_obs_starloc3  # unit vector of sun-SC in current SC body coordinates
    v3 = np.array([0, 1, 0])
    v1 = v1[:, np.newaxis]
    v2 = v2[:, np.newaxis]
    v3 = v3[:, np.newaxis]
    a4 = np.arctan2(
        -1
        * np.linalg.norm(np.cross(np.transpose(v1), np.transpose(v2)))
        * np.sign(np.linalg.det(a=np.hstack((v1, v2, -v3)))),
        np.dot(np.transpose(v1).flatten(), np.transpose(v2).flatten()),
    )
    ang = a4
    mat_C4 = np.array(
        [[np.cos(ang), 0, -np.sin(ang)], [0, 1, 0], [np.sin(ang), 0, np.cos(ang)]]
    )

    # unit vectors in inertial coordinates for plotting and sun angles
    xpppp = np.matmul(
        np.transpose(mat_C2),
        np.matmul(
            np.transpose(mat_C1),
            np.matmul(
                np.transpose(mat_C3),
                np.matmul(np.transpose(mat_C4), np.array([1, 0, 0])),
            ),
        ),
    )
    ypppp = np.matmul(
        np.transpose(mat_C2),
        np.matmul(
            np.transpose(mat_C1),
            np.matmul(
                np.transpose(mat_C3),
                np.matmul(np.transpose(mat_C4), np.array([0, 1, 0])),
            ),
        ),
    )
    zpppp = np.matmul(
        np.transpose(mat_C2),
        np.matmul(
            np.transpose(mat_C1),
            np.matmul(
                np.transpose(mat_C3),
                np.matmul(np.transpose(mat_C4), np.array([0, 0, 1])),
            ),
        ),
    )
    ################################################################################

    # calculate norms for sun-SC vector and tgt-SC vector distances
    tnorm = np.linalg.norm(tgt_ec - r_sc_o)
    snorm = np.linalg.norm(r_so - r_sc_o)

    # calculate sun_angle wrt spacecraft
    sun_angle = (
        np.arccos(np.dot((r_so - r_sc_o) / snorm, (tgt_ec - r_sc_o) / tnorm))
        * 180
        / np.pi
    )

    # return roll, pitch, yaw post zero-point orientation and sun angles

    # zero point alignment angles
    pitch1 = a1
    roll2 = a2

    # manuevers to point at tgt
    roll = 0 * 180 / np.pi  # for now
    yaw3 = a3 * 180 / np.pi
    pitch4 = a4 * 180 / np.pi

    # sun angles - dot sun-SC vector with each SC body frame unit vector
    sunx = np.arccos(np.dot((r_so - r_sc_o) / snorm, xpppp)) * 180 / np.pi
    suny = np.arccos(np.dot((r_so - r_sc_o) / snorm, ypppp)) * 180 / np.pi
    sunz = np.arccos(np.dot((r_so - r_sc_o) / snorm, zpppp)) * 180 / np.pi

    return roll, pitch4, yaw3, sun_angle, sunx, suny, sunz



