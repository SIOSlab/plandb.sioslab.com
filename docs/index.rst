.. Known Planets Database documentation master file, created by
   sphinx-quickstart on Tue Aug  7 10:56:42 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Known Planets Database
==================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

The Known Planets Database is a merge of the information in the NASA Exoplanet Science Institute Exoplanet Archive (hosted at IPAC) with calculated photometric, orbital, and direct imaging completeness information for planets of interest to the direct imaging community.  This documentation is intended to unambiguously define exactly how the database is generated and list all assumptions made.  It will also (eventually) contain use cases and sample queries. 

Database Generation
=============================

.. _photometry:

Planet Photometry
------------------------------

For photometry, we will using model grids by Nikole Lewis et al. (publication pending). These grids represent values of geometric albedo scaled by phase function value (:math:`p\Phi(\beta)`), parametrized by:

   * Metallicity (dex) for values [0. , 0.5, 1. , 1.5, 1.7, 2. ]
   * Star-planet separation (orbital radius) (AU) for values [0.5 , 0.6 , 0.7 , 0.85, 1.  , 1.5 , 2.  , 3.  , 4.  , 5.  ]
   * :math:`f_\textrm{sed}` for values [0.  , 0.01, 0.03, 0.1 , 0.3 , 1.  , 3.  , 6.  ] where 0 represents no clouds.
   * Phase (illuminant-object-observer), from 0 to 180 degrees in increments of 10 degrees
   * Wavelengths, from 0.3 to 0.9993 microns, in increments of 0.35 nm.
     

It is important to remember that the :math:`f_\textrm{sed}` parametrization does not map monotonically to the resulting planet reflectivity and interacts in a complex fashion with the wavelength or interest and the separation distance (which itself affects the condensing cloud species).  As an illustration, we plot phase curves for two cases: 575 nm at 1 AU and 825 nm at 5 AU:

|pic1|  |pic2|

    .. |pic1| image:: phase_curves_575nm_1AU.png
        :width: 49%

    .. |pic2| image:: phase_curves_825nm_5AU.png
        :width: 49%

In the former case, the reflectivity increases steadily as :math:`f_\textrm{sed}` is reduced from 6.0 to 0.3, and then decreases as the efficiency drops, creating thicker cloud decks which scatter less efficiently.  In the latter case, the peak instead occurs at f1.0.  For shorter separations, the condensing species switches, yielding different results (description by Natasha Batalha).

Because of these inherent non-linearities, it is not technically correct to interpolate over the full 5D grid space. Instead, the nearest neighbor grid is selected for stellar metallicity and orbit semi-major axis, and an interpolant is used over phase angle and wavelength.  For the current database, we calculate fluxes for all :math:`f_\textrm{sed}` levels but only for for the wavelengths/bandpasses of interest to WFIRST: 

* 575 nm/10%
* 660 nm/18%
* 730 nm/18%
* 760 nm/18%
* 825 nm/10%

This list may be extended in the future to support additional filters.


KnownPlanets Table
-------------------------------

The KnownPlanets table contains all of the target star and known planet properties that are needed to calculate direct imaging observables.  This describes the procedure for creating the table.

1. All data is grabbed from the Exoplanet Archive from the exoplanets and composite tables via the following API queries: ::

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


6. For the remaining rows that do not have a planet semi-major axis, it is filled in as follows: The `st_mass` value is taken to be the stellar mass (:math:`M_S`) in units of :math:`M_\odot` and the gravitational parameter is calculated as :math:`\mu = G M_S`. The ``pl_orbper`` value is taken to be the orbital period (:math:`T`) in units of days. The semi-major axis is then calculated as: 
   
   .. math::

    \left( \frac{ \mu T^2 }{ 4\pi^2 }\right)^\frac{1}{3}
    
  The values are converted to AU and copied to the `pl_orbsmax`` column.  The ``pl_orbsmaxreflink`` for these rows is updated to: "Calculated from stellar mass and orbital period."


7. The angular separation (``pl_angsep``) is re-calculated for all rows based on the current entries of ``pl_orbsmax`` (:math:`a`) and ``st_dist`` (:math:`d`) as:
    
   .. math::
    
    \tan^{-1}\left( \frac{a}{d} \right)

  with the result converted to miliarcseconds. 


8. The composite data table has radius values computed from available masses using the Forecaster best-fit ([Chen2016]_).  This tends to overestimate the radius for many Jovian-size planets, so we provide two additional mass estimates (but leave the original column).  We will only carry along masses in units of :math:`M_J`, and so drop columns ``pl_rade,  pl_radelim, pl_radserr2, pl_radeerr1, pl_rads, pl_radslim, pl_radeerr2, pl_radserr1``.


9. The original Forecaster ([Chen2016]_) best fit is composed of linear (in log-log space) segments of the form  :math:`R = 10^{\mathcal{C} + \log_{10}(M)\mathcal{S}}` where :math:`R` and :math:`M` are the radius and mass, respectively, and  :math:`\mathcal{C}` and :math:`\mathcal{S}` are fit coefficients defined in four intervals of mass (Terran, Neptunian, Jovian and Stellar Worlds).  We modify the Forecaster fit as follows:
   
   * The 'Terran Worlds' fit is unchanged.
   * The 'Neptunian Worlds' fit is modified to end at the Saturn mass/radius point.  
   * A new fit is added as a straight line (in log-log space) from Saturn mass/radius to Jupiter mass/radius.
   * The 'Jovian Worlds' fit is modified to be a constant Jupiter radius value from Jupiter Mass through 0.08 solar masses.
   * The 'Stellar Worlds' fit is unchanged. 

   
   The :math:`\mathcal{C}` and :math:`\mathcal{S}` parameters of this new fit are therefore:

   .. math::
    
    \begin{align*}
    \mathcal{C} &= [ 0.00346053, -0.06613329,  0.48091861,  1.04956612, -2.84926757] \\
    \mathcal{S} &= [0.279, 0.50376436, 0.22725968, 0., 0.881]
    \end{align*}

 for Earth mass/radius units with mass intervals:

 .. math::
    
    \mathcal{T} = [0, 2.04, 95.16, 317.828407, 26635.6863, \infty] M_\oplus
 
 The radius is calculated as :math:`10^{\mathcal{C} + \log_{10}(M)\mathcal{S}}` and the modified fit looks like this:

 .. image:: modified_forecaster_fit.png
    :scale: 50 %
    :alt: modified Forecaster fit


 A duplicate column is created from ``pl_radj`` called ``pl_radj_forecastermod`` and those rows that originally contained fit radii are overwritten with the modified fit.


10. The second radius fit is based on the planet density models from [Fortney2007]_. The relevant masses are converted to radii as follows:

    * For masses  :math:`\le 17_\oplus`, Eq. 8 (the rock/iron mixture density, as corrected in paper erratum) is used with a rock fraction of 0.67 (such that 1 Earth Mass gives 1 Earth radius). 
    * For masses :math:`> 17 M_\oplus`, Table 4 (mass-radius relationship for 4.5 Gyr planets) are interpolated, assuming a constant core mass of :math:`< 10 M_\oplus`. The mean age of stars in the subset of rows being considered here (for which age is available) is 4.64 Gyr (median of 4.27).  To avoid extrapolating beyond the available grid, semi-major axis values in the inputs are strictly truncated to the range (0.02, 9.5) AU (**this truncation only affects a small subset of known planets, but if any of these end up on a target list, this needs to be revisited**).

   A duplicate column is created from ``pl_radj`` called ``pl_radj_fortney`` and those rows that originally contained fit radii are overwritten with the modified fit.

   This approach consistently produces smaller radii for 'rocky' planets, as compared with the modified Forecaster fit, and consistently produces larger radii for giants, as can seen in the figure below:  

    .. image:: modified_forecaster_v_Fortney.png
        :scale: 50 %
        :alt: modified Forecaster fit vs. Fortney fit

11. For rows with eccentricity information available, a maximum possible angular separation is calculated as:

    .. math::
        
        tan^{-1}\left(\frac{a(1 + e)}{d}\right)
   
   where :math:`a` is the semi-major axis, :math:`d` is the target distance and  :math:`e` is the eccentricity. These values are assigned to a new column ``pl_maxangsep``, which equals ``pl_angsep`` for rows with no eccentricity available. 

12. A minimum possible angular separation is calculated as:

    .. math::
        
        tan^{-1}\left(\frac{s}{d}\right)
   
   where :math:`d` is the target distance and  :math:`s` is the minimum possible projected separation.  :math:`s` is calculated by taking the semi-major axis and scaling it by :math:`s(1 - e)` for those rows with eccentricity available.  This product is then further scaled by :math:`\cos(I)` for those rows with inclination available.  For those rows without inclination values, :math:`I` is assumed to be 90 degrees and  :math:`s`  is set to zero.

.. _photcalcref:

13. We now calculate fiducial values of angular separation and :math:`\Delta\textrm{mag}` at quadrature (planet phase of 90 degrees).  Rather than employing a full orbital solution (which is done for the :ref:`planetorbits_table`), here we simply take the planet's semi-major axis to be the orbital radius and projected separation at the time of quadrature. The angular separation at quadrature is therefore equal to the value already calculated in the ``pl_angsep`` column, and so no additional column needs to be added.

    For the photometry, we calculate the grid data (:math:`p\Phi(\beta)`) and resulting :math:`\Delta\textrm{mag}` for all combinations of cloud level and wavelength of interest (8x5) for 80 total photometry columns.  These columns are named ``quad_pPhi_XXXC_YYYNM`` and ``quad_dMag_XXXC_YYYNM`` where ``XXX`` is the cloud  :math:`f_\textrm{sed}` scaled by 100 (000 representing no cloud) and ``YYY`` is the wavelength in nm.  :math:`\Delta\textrm{mag}` is calculated as:

    .. math::
        
        \Delta\textrm{mag} = -2.5\log_{10}\left(p\Phi(\beta) \left(\frac{R}{r}\right)^2 \right)
    
    For planet radius :math:`R` and orbital radius :math:`r`. **Currently, the planet radius is taken from** ``pl_radj_forecastermod``.  **This can be updated to another fit, or an additional 80 columns can be generated for each new fit.**  For each wavelength, three columns are also add representing the minimum, maximum, and median :math:`\Delta\textrm{mag}` value across all cloud fractions.  These are named: ``quad_dMag_min_YYYNM``, ``quad_dMag_max_YYYNM``, and  ``quad_dMag_med_YYYNM``, respectively. As noted in :ref:`photometry`, it is important to remember that these do not correspond to the same :math:`f_\textrm{sed}` level in all cases.


.. _planetorbits_table:

PlanetOrbits Table
--------------------------------

The PlanetOrbits table contains orbital and photometric data for each known planet over 1 full orbital period.  

For each row in KnownPlanets, orbital data is generated as follows:

1. From KnownPlanets ``pl_orbsmax`` is taken as the semi-major axis :math:`a`.
2. ``pl_orbeccen`` is taken as the eccentricity :math:`e`.  If the eccentricity is undefined, it is set to zero. 
3. ``pl_orbincl`` is taken as the inclination :math:`I`; if it is undefined it is set to 90 degrees. 
4. ``pl_orblper`` is taken as the argument of periapsis :math:`\omega`; if it is undefined it is set to zero. **N.B: this is forcing the assumption of longitude of the ascending node equaling zero.** 
5. ``pl_radj_forecastermod`` is taken to be the planet radius :math:`R`. **We can replace another mass fit here.** 
6. ``st_dist`` is taken to be the target distance :math:`d`.
7. ``st_metfe`` is taken to be the stellar metallicity; if it is undefined it is set to zero.
8. The eccentric anomaly :math:`E` is calculated from the eccentricity for 100 equally spaced mean anomaly points :math:`M` between 0 and 360 degrees, inclusive.  The calculation is done via routine ``eccanom`` from EXOSIMS, which is a machine-precision Newton-Raphson inversion.
9. The Thiele-Innes constants are calculated as:

    .. math::

        A = a\begin{bmatrix} \cos\omega\\ \cos I \sin\omega \\ \sin I \sin \omega \end{bmatrix} \qquad B = a\sqrt{1 - e^2}\begin{bmatrix} -\sin\omega\\ \cos I \cos\omega \\ \sin I \cos \omega \end{bmatrix} 

   and the orbital radius vector is calculated for all 100 eccentric anomalies as:

    .. math::
        
        \mathbf r = A(\cos(E) - e) + B\sin(E)

10. The orbital radius :math:`r` is calculated as the norm of :math:`\mathbf r` and the projected separation :math:`s` is calculated as the norm of the first two rows of :math:`\mathbf r`.
11. The phase angle for all anomaly values is calculated as 
    
    .. math::
        \beta = \cos^{-1}\left(\frac{z}{r}\right)
    
   where :math:`z` is the third component of :math:`\mathbf r` and the angular separation is calculated as 
     
    .. math::
        \alpha = \tan^{-1}\left( \frac{s}{d} \right)

12. The PlanetOrbits table is defined with the following columns:
    
    * M: :math:`M` (rad)
    * r: :math:`r` (AU)
    * s: :math:`s` (AU)
    * WA: :math:`\alpha` (mas)
    * beta: :math:`\beta` (deg)

13. For each row, for each cloud level in the :ref:`photometry` grids, and for each wavelength of interest, the :math:`p\Phi(\beta)` value is interpolated and a :math:`\Delta\textrm{mag}` value is calculated as described in :ref:`Step 13<photcalcref>` of the KnownPlanets generation procedure. These values are stored in new columns ``pPhi_XXXC_YYYNM`` and ``dMag_XXXC_YYYNM`` where ``XXX`` is the cloud  :math:`f_\textrm{sed}` scaled by 100 (000 representing no cloud) and ``YYY`` is the wavelength in nm. 


Completeness Table
--------------------------

The Completeness table contains 2D frequency maps of the joint distribution of planet projected separation and :math:`\Delta\textrm{mag}`.  Currently, values are only generated for planets where ``pl_maxangsep`` is greater or equal to 100 mas and ``pl_minangsep`` is less than or equal to 500 mas, **however this calculation can be extended to all planets**.

For each planet meeting this condition, the following samples are drawn, :math:`n=1\times10^{6}` at a time:

1. Semi-major axis is sampled from a normal distribution, with :math:`\mu =` ``pl_orbsmax`` and :math:`\sigma =` (``pl_orbsmaxerr1`` :math:`-` ``pl_orbsmaxerr2``)/2.  In the case where the error is undefined, :math:`\sigma = \mu/100`.
2. Eccentricity is sampled from a normal distribution, with :math:`\mu =` ``pl_orbeccen`` and :math:`\sigma =` (``pl_orbeccenerr1`` :math:`-` ``pl_oreccenerr2``)/2.  In the case where the error is undefined, :math:`\sigma = \mu/100`. In the case where the eccentricity is undefined, the distribution is taken to be a Rayleigh distribution with :math:`\sigma = 0.0125`.
3. The inclination is sampled from a normal distribution, with :math:`\mu =` ``pl_orbincl`` and :math:`\sigma =` (``pl_orbinclerr1`` :math:`-` ``pl_orbinclerr2``)/2.  In the case where the error is undefined, :math:`\sigma = \mu/100`. In the case where the inclination is undefined, the inclination is sampled from a sinusoidal distribution. For rows where the planet mass (``pl_bmassj``) represents an estimate of the true mass, this sinusoidal inclination is sampled over the full range from 0 to 180 degrees.  However, in cases where the planet mass represents :math:`M\sin I`, a critical inclination value is calculated at which the true planet mass would equal 0.08 Solar masses, and the distribution is sampled only between this critical inclination and 180 degrees minus this critical inclination. 
4. The longitude of periapsis is sampled from a normal distribution, with :math:`\mu =` ``pl_orblper`` and :math:`\sigma =` (``pl_orblpererr1`` :math:`-` ``pl_orblpererr2``)/2.  In the case where the error is undefined, :math:`\sigma = \mu/100`. In the case where the argument of periapsis is undefined, it is sampled from a uniform distribution between 0 and 360 degrees.
5. The longitude of the ascending node is sampled from a uniform distribution between 0 and 360 degrees.
6. The stellar metallicity is not sampled, but taken as a constant value equal to ``st_metfe``, or 0.0, if undefined. 
7. If the planet radius in KnownPlanets was calculated from the mass, and the mass (``pl_bmassj``) represents :math:`M\sin I`, the mass sample is defined as ``pl_bmassj``:math:`/sin(I)`.



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
