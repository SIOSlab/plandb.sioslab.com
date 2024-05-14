from astropy.time import Time


def findRefStars(ref_stars, target_star, phi_max):
    """Args:
        ref_stars: list of SkyCoord instances representing
                   reference star coordinates
        target_star : a SkyCoord instance of target star coordinates
        phi_max : a float maximum angular separation in degrees

    Returns:
        a list of indices of usable reference stars in the ref_stars list"""
    valid_stars = []
    for star in ref_stars:
        new_star = star.apply_space_motion(new_obstime=Time.now())
        phi = new_star.separation(target_star)
        # print("phi with star ",ref_stars.index(star), " ",phi.degree) #output for testing
        if phi.value < phi_max:
            valid_stars.append(ref_stars.index(star))
    return valid_stars
