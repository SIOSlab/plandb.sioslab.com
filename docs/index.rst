.. Known Planets Database documentation master file, created by
   sphinx-quickstart on Tue Aug  7 10:56:42 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Known Planets Database's documentation!
==================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


KnownPlanets Table Generation
-------------------------------

The KnownPlanets table contains all of the target star and known planet properties that are needed to calculate direct imaging observables.  This describes the procedure for creating the table.

1. All data is grabbed from the IPAC-hosted Exoplanet Archive from the exoplanets and composite tables via the following API queries: ::

    https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=compositepars&select=*&format=csv"
    https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=exoplanets&select=*&format=csv"

2. The leading 'f' is stripped off all data column names for the composite table data to match the names in the exoplanets table data.
   
3. Composite table columns with different names from the exoplanets table are renamed to match the exoplanets table naming as follows: ::
    
    'pl_smax':'pl_orbsmax'
    'pl_smaxerr1':'pl_orbsmaxerr1'
    'pl_smaxerr2':'pl_orbsmaxerr2'
    'pl_smaxlim':'pl_orbsmaxlim'
    'pl_smaxreflink':'pl_orbsmaxreflink'
    'pl_eccen':'pl_orbeccen'
    'pl_eccenerr1':'pl_orbeccenerr1'
    'pl_eccenerr2':'pl_orbeccenerr2'
    'pl_eccenlim':'pl_orbeccenlim'
    'pl_eccenreflink':'pl_orbeccenreflink'
    'st_met':'st_metfe' 
    'st_meterr1':'st_metfeerr1' 
    'st_meterr2':'st_metfeerr2'
    'st_metreflink':'st_metfereflink'
    'st_metlim':'st_metfelim'

4. The two data sets are merged with the composite table values used preferentially when both are populated.

5. The data is filtered such that the remaining rows have:
    
    |   Star Distance AND
    |   (Planet Semi-Major Axis OR (Planet Period AND Star Mass)) AND 
    |   (Planet Radius OR Planet Mass OR Planet :math:`m\sin(I))`.


6. For the remaining rows that do not have a planet semi-major axis, it is filled in as follows: The `st_mass` value is taken to be the stellar mass (:math:`M_S`) in units of :math:`M_\odot` and the gravitational parameter is calculated as :math:`\mu = G M_S`. The `pl_orbper` value is taken to be the orbital period (:math:`T`) in units of days. The semi-major axis is then calculated as: 
   
   .. math::

    \left( \frac{ \mu T^2 }{ 4\pi^2 }\right)^\frac{1}{3}
    
  The values are converted to AU and copied to the `pl_orbsmax` column.  The `pl_orbsmaxreflink` for these rows is updated to: "Calculated from stellar mass and orbital period."


7. The angular separation (`pl_angsep`) is re-calculated for all rows based on the current entries of `pl_orbsmax` (:math:`a`) and `st_dist` (:math:`d`) as:
    
   .. math::
    
    \tan^{-1}\left( \frac{a}{d} \right)

  with the result converted to miliarcseconds. 


8. The composite data table has radius values computed from available masses using the Forecaster best-fit ([Chen2016]_).  This tends to overestimate the radius for many Jovian-size planets, so we provide two additional mass estimates (but leave the original column).  We will only carry along masses in units of :math:`M_J`, and so drop columns `pl_rade`,  `pl_radelim`, `pl_radserr2`, `pl_radeerr1`, `pl_rads`, `pl_radslim`, `pl_radeerr2`, and  `pl_radserr1`.


9. The first alternate mass fit is based on a modification of the Forecaster fit as follows:
   
   * The 'Terran Worlds' fit is unchanged.
   * The 'Neptunian Worlds' fit is modified to end at the Saturn mass/radius point.  
   * A new fit is added as a straight line (in log-log space) from Saturn mass/radius to Jupiter mass/radius.
   * The 'Jovian Worlds' fit is modified to be a constant Jupiter radius value from Jupiter Mass through 0.08 solar masses.
   * The 'Stellar Worlds' fit is unchanged. 

   The :math:`\mathcal{C}` and :math:`\mathcal{S}` parameters (following the nomencalture of the original Forecaster paper [Chen2016]_) of this new fit are therefore:

   .. math::
    
    \mathcal{C} = [ 0.00346053, -0.06613329,  0.48091861,  1.04956612, -2.84926757]

    \mathcal{S} = [0.279, 0.50376436, 0.22725968, 0., 0.881]

 for Earth mass/radius units.  The radius is calculated as :math:`10^{\mathcal{C} + \log_{10}(m)\mathcal{S}}` and the modified fit looks like this:

 .. image:: modified_forecaster_fit.png
    :scale: 50 %
    :alt: modified Forecaster fit


 A duplicate column is created from `pl_radj` called `pl_radj_forecastermod` and those rows that originally contained fit radii are overwritten with the modified fit.


10. The second radius fit is based on the planet density models from [Fortney2007]_. The relevant masses are converted to radii as follows:

    * For masses  :math:`\le 17_\oplus`, Eq. 8 (the rock/iron mixture density, as corrected in paper erratum) is used with a rock fraction of 0.67 (such that 1 Earth Mass gives 1 Earth radius). 
    * For masses :math:`> 17 M_\oplus`, Table 4 (mass-radius relationship for 4.5 Gyr planets) are interpolated, assuming a constant core mass fraction of :math:`< 10 M_\oplus`. The mean age of stars in the subset of rows being considered here (for which age is available) is 4.64 Gyr (mean of 4.27).  To avoid extrapolating beyond the available grid, semi-major values in the inputs are strictly restricted to the range (0.02, 9.5) AU.

   This approach consistently produces smaller radii for 'rocky' planets, as compared with the modified Forecaster fit, and consistently produces larger radii for giants, as can seen in the figure below:  

    .. image:: modified_forecaster_v_Fortney.png
        :scale: 50 %
        :alt: modified Forecaster fit vs. Fortney fit

11. For rows with eccentricity information available, a maximum angular separation is calculated as:

    .. math::
        
        tan^{-1}\left(\frac{a(1 + e)}{d}\right)
   
   where :math:`a` is the semi-major axis, :math:`d` is the target distance and  :math:`e` is the eccentricity. These values are assigned to a new column `pl_maxangsep`, which equals `pl_angsep` for rows with no eccentricity available. 

12. A minimum angular separation is calculated as:

    .. math::
        
        tan^{-1}\left(\frac{s}{d}\right)
   
   where :math:`d` is the target distance and  :math:`s` is the minimum projected separation.  :math:`s` is calculated by taking the semi-major axis and scaling it by :math:`s(1 + e)` for those rows with eccentricity available.  This product is then further scaled by :math:`\cos(I)` for those rows with inclination available.  For those rows without inclination values, :math:`I` is assumed to be 90 degrees and  :math:`s`  is set to zero.


At this point, the rows are dumped directly to the KnownPlanets table.

PlanetOrbits Table Generation
--------------------------------

The PlanetOrbits table contains orbital and photometric data for each known planet over 1 full orbital period.  

For photometry, we will using model grids by Nikole Lewis et al. (publication pending). These grids represent values of geometric albedo scaled by phase function value (:math:`p\Phi(\beta)`), parametrized by:

   * Metallicity (dex) for values [0. , 0.5, 1. , 1.5, 1.7, 2. ]
   * Star-planet separation (orbital radius) (AU) for values [0.5 , 0.6 , 0.7 , 0.85, 1.  , 1.5 , 2.  , 3.  , 4.  , 5.  ]
   * :math:`f_\textrm{sed}` for values [0.  , 0.01, 0.03, 0.1 , 0.3 , 1.  , 3.  , 6.  ] where 0 represents no clouds.
   * Phase (illuminant-object-observer), from 0 to 180 degrees in increments of 10 degrees
   * Wavelengths, from 0.3 to 0.9993 microns, in increments of 0.35 nm.
     
   As it is not technically correct to interpolate over the full 5D grid space, the nearest neighbor grid will be selected for stellar metallicity and orbit semi-major axis, and an interpolant used over phase angle and wavelength.  We will calculate flux ratios for all :math:`f_\textrm{sed}` levels for the  wavelengths of interest: 575 nm/10%,  660 nm/18%, 730 nm/18%, 760 nm/18%, and 825 nm/10%.


References
-----------------
.. [Chen2016] Chen, J. and Kipping, D. M. (2016) Probabilistic Forecasting of the Masses and Radii of Other Worlds, ApJ 834(1)
.. [Fortney2007] Fortney, J. J., Marley, M. S. and Barnes, J. W. (2007) Planetary Radii across Five Orders of Magnitude in Mass and Stellar Insolation: Application to Transits, ApJ 659, 2
.. [Brown2005] Brown, R. A. (2005) Single-visit photometric and obscurational completeness, ApJ 624
.. [Garrett2016] Garett, D. and Savransky, D. (2016) Analytical Formulation of the Single-visit Completeness Joint Probability Density Function, ApJ 828(1)

    

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
