.. Known Planets Database documentation master file, created by
   sphinx-quickstart on Tue Aug  7 10:56:42 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Imaging Mission Database
==================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

The Imaging Mission Database is a merge of the information in the NASA Exoplanet Science Institute Exoplanet Archive (hosted at IPAC) with calculated photometric, orbital, and direct imaging completeness information for planets of interest to the direct imaging community. It also includes potential blind search targets from the ExoCat-1 Catalog along with their depth of search values.   This documentation is intended to unambiguously define exactly how the database is generated and list all assumptions made.  It will also (eventually) contain use cases and sample queries. 

.. _photometry:

Planet Photometry
=====================

For photometry, we will using model grids by Natasha Batalha and Nikole Lewis et al. ([Batalha2018]_). These grids represent values of geometric albedo scaled by phase function value (:math:`p\Phi(\beta)`), parametrized by:

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

Integration over filter bandpasses is currently implemented as a left-Riemann sum, so that:

    .. math::
        
        p\Phi(\beta)(\lambda, \Delta\lambda) = \frac{1}{\Delta\lambda} \frac{\Delta\lambda}{n-1}\sum_{i = 0}^{n-1} f_{m,c,d,\beta}\left(\lambda - \frac{\Delta\lambda}{2} + i\frac{\Delta\lambda}{n-1}  \right)

where :math:`f_{m,c,d,\beta}` is the relevant interpolant over the model grid on the metallicity, cloud, orbital distance and phase angle axes, and :math:`n` is taken to be 100.

    .. note::
        
        This is a pretty crude integral approximation, but the associated error is significantly lower than all other errors involved here, so there's not much point in changing it to something more complex.  Straight quadrature over the interpolant is very fragile given the grids, and takes excessively long to compute.

.. _sampleclouds:

In the case where we wish to sample over a distribution of  :math:`f_\textrm{sed}` values, we adopt the following discrete density function:

=========      ==============
f_sed          Frequency
=========      ==============
0.000000       0.099
0.010000       0.001
0.030000       0.005
0.100000       0.010
0.300000       0.025
1.000000       0.280
3.000000       0.300
6.000000       0.280
=========      ==============

This distribution is provided courtesy of Mark Marley (private communication) with the following explanation:

The reflectivity of all types of planets, but especially gas giant planets, is extremely sensitive to cloud properties. The group of modeling codes for extrasolar giant planets developed by Marley and collaborators (e.g., [Marley1999a]_; [Marley1999b]_; [Ackerman2001]_; [Cahoy2010]_; [MacDonald2018]_) describes clouds height and optical depth with a parameter  :math:`f_\textrm{sed}`  which describes the relative importance of vertical transport and sedimentation of cloud particles. When  :math:`f_\textrm{sed}`  is small uplift dominates over sedimentation and clouds are vertically thick with small particles. When f_sed is large sedimentation dominates over upward mixing and particles are large and clouds are thinner. Studies of brown dwarfs and comparisons to solar system gas giants typically find  :math:`f_\textrm{sed}`  values of 1 to 5 best match observations. Efforts to match the vertically extended clouds apparently seen in transiting planets have relied on very small values of  :math:`f_\textrm{sed}`  (much less than 1). 

To date the value of  :math:`f_\textrm{sed}`  has been set by comparison to models to data and there is little a priori basis to choose a particular value. However in the solar system gas giants Jupiter and Saturn, clouds are found in relatively distinct layers compatible with moderate values of  :math:`f_\textrm{sed}` . Given this context our prior expectation for the cool giants to be imaged by WFIRST is for distinct cloud decks compatible with moderate  :math:`f_\textrm{sed}`  (1 to 6). Very thick, extensive clouds such as found in young, warm Jupiters or highly irradiated hot Jupiters are not expected, but are possible. Likewise very thin or absent clouds, perhaps arising from unexpected atmospheric composition, thermal profile, or dynamics are likewise not expected but possible. 

To capture this expectation we assign a frequency of about 30% each to moderate values of  :math:`f_\textrm{sed}`  (1 to 6), a frequency of just a few percent to small values (<1) seen in transiting planets, and the remainder (about 10%) to cloudless case. 

We note that photochemistry and haze production may very well play a determinative role in forming atmospheric aerosols and controlling atmospheric reflectivity, but there are as yet no adequate models to handle this. The range of  :math:`f_\textrm{sed}` values considered here should encompass all types of reflectivity, including that produced by photochemical hazes.

Known Planet Targets
=============================


KnownPlanets Table
-------------------------------

The KnownPlanets table contains all of the target star and known planet properties that are needed to calculate direct imaging observables.  This describes the procedure for creating the table.

#. All data is grabbed from the Exoplanet Archive from the exoplanets, composite, and extended tables via the following API queries: ::

    https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=compositepars&select=*&format=csv"
    https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=exoplanets&select=*&format=csv"
    https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=exomultpars&select=*&format=csv"

#. The leading 'f' is stripped off all data column names for the composite table data and the leading 'm' is stripped off all data column names for the extended table to match the names in the exoplanets table data.
   
#. Composite table columns with different names from the exoplanets table are renamed to match the exoplanets table naming as follows: ::
    
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

#. The composite and exoplanet data sets are merged with the composite table values used preferentially when both are populated.


#. For planets that were previously calculated to have a completeness larger than 0, data is substituted in from the extended table. For each planet of interest, we categorize each source according to the priority listing below. We then substitute data into the joined data from the source with the highest priority.  If two sources are tied for the highest priority, then the most recent one is chosen.


   #. Satisfies the criteria described in step 6 (barring the existence of stellar distance) and has non-null eccentricity, time of periastron passage, argument of periastron, and inclination. 
   
   #. Satisfies the criteria described in step 6 (barring the existence of stellar distance) and has non-null eccentricity, time of periastron passage, and argument of periastron.
   
   #. Satisfies the criteria described in step 6, (barring the existence of stellar distance) has non-null eccentricity, and has either non-null time of periastron passage or argument of periastron.
	
   #. Satisfies the criteria described in step 6 (barring the existence of stellar distance) and has non-null eccentricity.
	
   #. Satisfies the criteria described in step 6 (barring the existence of stellar distance).

      .. note::

         The ``*_orblper`` columns in the various tables in the Exoplanet Archive database are all labeled as longitude of periastron, but the documentation makes it clear that these columns represent the argument of periastron, and so these values will always be referred to as arguments of periastron throughout this documentation. 
   

   The following columns along with their respective errors and limits (where applicable) are substituted from the extended table: ::
		
   'pl_orbper'
   'pl_orbsmax'
   'pl_orbeccen'
   'pl_orbtper'
   'pl_orblper'
   'pl_bmassj'
   'pl_bmassprov'
   'pl_radj'
   'pl_orbincl'
   'pl_orbperreflink'
   'pl_orbsmaxreflink'
   'pl_orbeccenreflink'
   'pl_bmassreflink'
   'pl_radreflink'

   For the last 5 columns denoting reference link columns, data is copied from the 'mpl_reflink' column to populate these 5 columns, since these columns do not exist in the extended table.  Additionally, if 'st_mass' has a non-null value in the row that is being substituted in, then 'st_mass' and its associated error, limit, and reference link columns are also substituted.


#. The data is filtered such that the remaining rows have:
    
    |   Star Distance AND
    |   (Planet Semi-Major Axis OR (Planet Period AND Star Mass)) AND 
    |   (Planet Radius OR Planet Mass OR Planet :math:`m\sin(I))`.

   .. _massperiodsmacalc:

#. For the remaining rows that do not have a planet semi-major axis, it is filled in as follows: The ``st_mass`` value is taken to be the stellar mass (:math:`M_S`) in units of :math:`M_\odot` and the gravitational parameter is calculated as :math:`\mu = G M_S`. The ``pl_orbper`` value is taken to be the orbital period (:math:`T`) in units of days. The semi-major axis is then calculated as: 

   .. math::

      \left( \frac{ \mu T^2 }{ 4\pi^2 }\right)^\frac{1}{3}
    
   The values are converted to AU and copied to the ``pl_orbsmax`` column.  The ``pl_orbsmaxreflink`` for these rows is updated to: "Calculated from stellar mass and orbital period."  
   
   The semi-major axis errors for these calculations are taken to be the propagation of the errors in stellar mass and orbital period through a first-order linearization of the equation above, under the assumption of independence of these errors, such that:

   .. math::

      \sigma_a^2 = \frac{\partial}{\partial T}\left( \frac{ \mu T^2 }{ 4\pi^2 }\right)^\frac{1}{3} \sigma_T^2 + \frac{\partial}{\partial \mu}\left( \frac{ \mu T^2 }{ 4\pi^2 }\right)^\frac{1}{3} \sigma_\mu^2

      = \frac{2^{\frac{2}{3}} \left(T^{2} \mu\right)^{\frac{2}{3}}}{9 \pi^{\frac{4}{3}} T^{2}}\sigma_T^2 + \frac{2^{\frac{2}{3}} \left(T^{2} \mu\right)^{\frac{2}{3}}}{36 \pi^{\frac{4}{3}} \mu^{2}}\sigma_\mu^2

   where :math:`\sigma_T` and :math:`\sigma_\mu` are taken as the mean values from ``st_masserr1`` and ``st_masser2`` and ``pl_orbpererr1`` and ``pl_orbpererr2``, respectively.  The :math:`\sigma_a` values are then copied to the ``pl_orbsmaxerr1`` and (negative) ``pl_orbsmaxerr2`` columns for the missing semi-major axis rows. 

#. The angular separation (``pl_angsep``) is re-calculated for all rows based on the current entries of ``pl_orbsmax`` (:math:`a`) and ``st_dist`` (:math:`d`) as:
    
   .. math::
    
      \tan^{-1}\left( \frac{a}{d} \right)

   with the result converted to miliarcseconds. As above, the errors are propagated via a first-order linearization of the equation, so that:

   .. math::

      \sigma_\alpha = \sqrt{\frac{a^{2} \sigma_{d}^{2} + d^{2} \sigma_{a}^{2}}{\left(a^{2} + d^{2}\right)^{2}}}
   
   where :math:`\sigma_a` and :math:`\sigma_d` are again taken from the ``pl_orbsmaxerr*`` and ``st_disterr*`` columns and the the resulting :math:`\sigma_\alpha` values are written to the ``pl_angseperr*`` columns. 


#. The composite data table has radius values computed from available masses using the Forecaster best-fit ([Chen2016]_).  This tends to overestimate the radius for many Jovian-size planets, so we provide two additional mass estimates (but leave the original column).  We will only carry along masses in units of :math:`M_J`, and so drop columns ``pl_rade,  pl_radelim, pl_radserr2, pl_radeerr1, pl_rads, pl_radslim, pl_radeerr2, pl_radserr1``.

   .. _forecastermodref:

#. The original Forecaster ([Chen2016]_) best fit is composed of linear (in log-log space) segments of the form  :math:`R = 10^{\mathcal{C} + \log_{10}(M)\mathcal{S}}` where :math:`R` and :math:`M` are the radius and mass, respectively, and  :math:`\mathcal{C}` and :math:`\mathcal{S}` are fit coefficients defined in four intervals of mass (Terran, Neptunian, Jovian and Stellar Worlds).  We modify the Forecaster fit as follows:
   
   * The 'Terran Worlds' fit is unchanged.
   * The 'Neptunian Worlds' fit is modified to end at the Saturn mass/radius point.  
   * A new fit is added as a straight line (in log-log space) from Saturn mass/radius to Jupiter mass/radius.
   * The 'Jovian Worlds' fit is modified to be a constant Jupiter radius value from Jupiter Mass through 0.08 solar masses.
   * The 'Stellar Worlds' fit is unchanged. 

   
   The :math:`\mathcal{C}` and :math:`\mathcal{S}` parameters of this new fit are therefore:

   .. math::
    
    
    \mathcal{C} &= [ 0.00346053, -0.06613329,  0.48091861,  1.04956612, -2.84926757] \\
    \mathcal{S} &= [0.279, 0.50376436, 0.22725968, 0., 0.881]
    
   for Earth mass/radius units with mass intervals:

   .. math::
    
      \mathcal{T} = [0, 2.04, 95.16, 317.828407, 26635.6863, \infty] M_\oplus
 
   The radius is calculated as :math:`10^{\mathcal{C} + \log_{10}(M)\mathcal{S}}` and the modified fit looks like this:

   .. image:: modified_forecaster_fit.png
      :scale: 50 %
      :alt: modified Forecaster fit

   A duplicate column is created from ``pl_radj`` called ``pl_radj_forecastermod`` and those rows that originally contained fit radii are overwritten with the modified fit.

   .. _fortneyref:

#. The second radius fit is based on the planet density models from [Fortney2007]_. The relevant masses are converted to radii as follows:

    * For masses  :math:`\le 17_\oplus`, Eq. 8 (the rock/iron mixture density, as corrected in paper erratum) is used with a rock fraction of 0.67 (such that 1 Earth Mass gives 1 Earth radius). 
    * For masses :math:`> 17 M_\oplus`, Table 4 (mass-radius relationship for 4.5 Gyr planets) are interpolated, assuming a constant core mass of :math:`< 10 M_\oplus`. The mean age of stars in the subset of rows being considered here (for which age is available) is 4.64 Gyr (median of 4.27).  To avoid extrapolating beyond the available grid, semi-major axis values in the inputs are strictly truncated to the range (0.02, 9.5) AU.
    
    .. note::

        This truncation only affects a small subset of known planets, but if any of these end up on a target list, this needs to be revisited.

   A duplicate column is created from ``pl_radj`` called ``pl_radj_fortney`` and those rows that originally contained fit radii are overwritten with the modified fit. This fit effectively recreates the 'Terran' world leg of the modified forecaster, but extends it well into 'Neptunian' worlds.  Similarly, while the fits agree for Jovian and Super-Jovian objects, they have different behaviours around the location of Saturn:
    .. image:: massfits.png
        :scale: 100 %
        :alt: modified Forecaster fit vs. Fortney fit

   This approach consistently produces smaller radii for Super-Earth - Neptune planets, as compared with the modified Forecaster fit, and consistently produces larger radii for giants, as can seen in the figure below:  

    .. image:: modified_forecaster_v_Fortney.png
        :scale: 50 %
        :alt: modified Forecaster fit vs. Fortney fit

#. For rows with eccentricity information available, a maximum possible angular separation is calculated as:

    .. math::
        
        tan^{-1}\left(\frac{a(1 + e)}{d}\right)
   
   where :math:`a` is the semi-major axis, :math:`d` is the target distance and  :math:`e` is the eccentricity. These values are assigned to a new column ``pl_maxangsep``, which equals ``pl_angsep`` for rows with no eccentricity available. 

#. A minimum possible angular separation is calculated as:

   .. math::
        
      tan^{-1}\left(\frac{s}{d}\right)
   
   where :math:`d` is the target distance and  :math:`s` is the minimum possible projected separation.  :math:`s` is calculated by taking the semi-major axis and scaling it by :math:`s(1 - e)` for those rows with eccentricity available.  This product is then further scaled by :math:`\cos(I)` for those rows with inclination available.  For those rows without inclination values, :math:`I` is assumed to be 90 degrees and  :math:`s`  is set to zero.

   .. _photcalcref:

#. We now calculate fiducial values of angular separation and :math:`\Delta\textrm{mag}` at quadrature (planet phase of 90 degrees).  We have two approaches for calculating the quadrature radius depending on what data exists for the planet.  If the planet has data for eccentricity and argument of periastron and the inclination data is either null or non-zero, then we calculate the true anomaly using the fact that :math:`\theta = \omega+\nu` and:

	.. math::
			\theta = \sin^{-1}\left(\frac{\cos\beta}{\sin I}\right)

   This gives us two solutions for the true anomaly, :math:`\nu = -\omega` and :math:`\nu = \pi - \omega`.  Two possible values for the radius at quadrature are calculated as: 

	.. math::

			r = \frac{a(1 - e^2)}{1 + e\cos\nu}
			
	
   From these two radius values, we pick the radius that results in the smallest :math:`\Delta\textrm{mag}` (as calculated below) so that we determine the highest possible brightness for the planet.  
   
   In the case where eccentricity or argument of periastron are null, or if the orbit is face-on, then we take the semi-major axis to be the orbial radius at quadrature.

   For the photometry, we calculate the grid data (:math:`p\Phi(\beta)`) and resulting :math:`\Delta\textrm{mag}` for all combinations of cloud level and wavelength of interest (8x5) for 80 total photometry columns.  These columns are named ``quad_pPhi_XXXC_YYYNM`` and ``quad_dMag_XXXC_YYYNM`` where ``XXX`` is the cloud  :math:`f_\textrm{sed}` scaled by 100 (000 representing no cloud) and ``YYY`` is the wavelength in nm.  :math:`\Delta\textrm{mag}` is calculated as:

    .. math::
        
        \Delta\textrm{mag} = -2.5\log_{10}\left(p\Phi(\beta) \left(\frac{R}{r}\right)^2 \right)
    
   for planet radius :math:`R` and orbital radius :math:`r`.  We store the radius at quadrature that is used in the final calculation in the ``quad_dmag`` column.
    
    .. note::

        Currently, the planet radius is taken from ``pl_radj_forecastermod``.  This can be updated to another fit, or an additional 80 columns can be generated for each new fit.
        
   For each wavelength, three columns are also add representing the minimum, maximum, and median :math:`\Delta\textrm{mag}` value across all cloud fractions.  These are named: ``quad_dMag_min_YYYNM``, ``quad_dMag_max_YYYNM``, and  ``quad_dMag_med_YYYNM``, respectively. As noted in :ref:`photometry`, it is important to remember that these do not correspond to the same :math:`f_\textrm{sed}` level in all cases.


.. _planetorbits_table:

PlanetOrbits Table
--------------------------------

The PlanetOrbits table contains orbital and photometric data for each known planet over 1 full orbital period.  

For each row in KnownPlanets, orbital data is generated as follows:

#. From KnownPlanets ``pl_orbsmax`` is taken as the semi-major axis :math:`a`.
#. ``pl_orbeccen`` is taken as the eccentricity :math:`e`.  If the eccentricity is undefined, it is set to zero. 
#. ``pl_orbincl`` is taken as the inclination :math:`I`; if it is undefined it is set to 90 degrees. 
#. ``pl_orblper`` is taken as the argument of periastron :math:`\omega`; if it is undefined it is set to zero. 
#. ``pl_radj_forecastermod`` is taken to be the planet radius :math:`R`. 
   
   .. note::

    We can replace another mass fit here.
    
#. ``st_dist`` is taken to be the target distance :math:`d`.
#. ``st_metfe`` is taken to be the stellar metallicity; if it is undefined it is set to zero.
#. If ``pl_orbper`` (:math:`T`)is defined, or the orbital period can be calculated from the star mass and semi-major axis  (:ref:`see above<massperiodsmacalc>`), and ``pl_opbtper`` (the time of periastron passage: :math:`\tau`) is defined, then a time array is defined with 100 points between :math:`t0` and :math:`t0+T`, where :math:`t0` is taken to be January 1, 2026 00:00:00 UTC. The mean anomaly is then calculated as:

   .. math::

      M = \frac{2\pi}{T}(t - \tau)

   If the period cannot be established, or the time of periapsis passage is undefined, then the mean anomaly is defined as 100 equally spaced points between 0 and 360 degrees. 

#. The eccentric anomaly :math:`E` is calculated from the eccentricity for all values of the mean anomaly array defined above.  The calculation is done via routine ``eccanom`` from EXOSIMS, which is a machine-precision Newton-Raphson inversion. True anomaly is calculated from the eccentric anomaly as:

   .. math::

        \nu = 2\tan^{-1}\left(\sqrt{\frac{1+e}{1 - e}}\tan\left(\frac{E}{2}\right)\right)
  

   .. _orbcalcref:

#.  The orbital radius :math:`r` is calculated as:

    .. math::

        r = \frac{a(1 - e^2)}{1 + e\cos\nu}

#. The projected separation :math:`s` is calculated as:

    .. math::
        
        s = \frac{r \sqrt{4 \cos{\left (2 I \right )} + 4 \cos{\left (2 \nu + 2 \omega \right )} - 2 \cos{\left (- 2 I + 2 \nu + 2 \omega \right )} - 2 \cos{\left (2 I + 2 \nu + 2 \omega \right )} + 12}}{4}

#. The phase angle for all anomaly values is calculated as 
    
    .. math::
        \beta = \cos^{-1}\left(\sin I\sin\theta\right)
    
   where :math:`\theta = \omega+\nu` and the angular separation is calculated as 
     
    .. math::
        \alpha = \tan^{-1}\left( \frac{s}{d} \right)

#. The PlanetOrbits table is defined with the following columns:
    
    * M: :math:`M` (rad)
    * t: :math:`t` (days after 1/1/2026; NaN when period of time of periastron passage are undefined)
    * r: :math:`r` (AU)
    * s: :math:`s` (AU)
    * WA: :math:`\alpha` (mas)
    * beta: :math:`\beta` (deg)

#. For each row, for each cloud level in the :ref:`photometry` grids, and for each wavelength of interest, the :math:`p\Phi(\beta)` value is interpolated and a :math:`\Delta\textrm{mag}` value is calculated as described in :ref:`Step 13<photcalcref>` of the KnownPlanets generation procedure. These values are stored in new columns ``pPhi_XXXC_YYYNM`` and ``dMag_XXXC_YYYNM`` where ``XXX`` is the cloud  :math:`f_\textrm{sed}` scaled by 100 (000 representing no cloud) and ``YYY`` is the wavelength in nm. 


.. _altplanetorbits_table:

AltPlanetOrbits Table
--------------------------------

The AltPlanetOrbits table contains the same basic information as the PlanetOrbits table, but only for planets whose eccentricity is unknown, and for multiple different values of inclination. The exact same set of calculations are performed as for PlanetOrbits, except the inclination is taken at four different values: 30, 60, 90 degrees, and a critical inclination value.  In cases where the only known planet mass is :math:`M\sin(I)`, the critical value is the inclination at which the planet mass would equal 0.08 solar masses.  Otherwise, the critical inclination is taken to be 10 degrees.

.. _comptable:

Completeness Table
--------------------------

The Completeness table contains 2D frequency maps of the joint distribution of planet projected separation and :math:`\Delta\textrm{mag}` (for excruciating detail see [Brown2005]_ and [Garrett2016]_).  Currently, values are only generated for planets where ``pl_maxangsep`` is greater or equal to 100 mas and ``pl_minangsep`` is less than or equal to 500 mas, **however this calculation can be extended to all planets**.

For each planet meeting this condition, the following samples are drawn, :math:`n=1\times10^{6}` at a time:

#. Semi-major axis is sampled from a normal distribution, with :math:`\mu =` ``pl_orbsmax`` and :math:`\sigma =` (``pl_orbsmaxerr1`` :math:`-` ``pl_orbsmaxerr2``)/2.  In the case where the error is undefined, :math:`\sigma = \mu/100`.
#. Eccentricity is sampled from a normal distribution, with :math:`\mu =` ``pl_orbeccen`` and :math:`\sigma =` (``pl_orbeccenerr1`` :math:`-` ``pl_oreccenerr2``)/2.  In the case where the error is undefined, :math:`\sigma = \mu/100`. In the case where the eccentricity is undefined, the distribution is taken to be a Rayleigh distribution with :math:`\sigma = 0.0125`.
#. The inclination is sampled from a normal distribution, with :math:`\mu =` ``pl_orbincl`` and :math:`\sigma =` (``pl_orbinclerr1`` :math:`-` ``pl_orbinclerr2``)/2.  In the case where the error is undefined, :math:`\sigma = \mu/100`. In the case where the inclination is undefined, the inclination is sampled from a sinusoidal distribution. For rows where the planet mass (``pl_bmassj``) represents an estimate of the true mass, this sinusoidal inclination is sampled over the full range from 0 to 180 degrees.  However, in cases where the planet mass represents :math:`M\sin I`, a critical inclination value is calculated at which the true planet mass would equal 0.08 Solar masses, and the distribution is sampled only between this critical inclination and 180 degrees minus this critical inclination. 
#. The argument of periastron is sampled from a normal distribution, with :math:`\mu =` ``pl_orblper`` and :math:`\sigma =` (``pl_orblpererr1`` :math:`-` ``pl_orblpererr2``)/2.  In the case where the error is undefined, :math:`\sigma = \mu/100`. In the case where the argument of periastron is undefined, it is sampled from a uniform distribution between 0 and 360 degrees.
#. The longitude of the ascending node is sampled from a uniform distribution between 0 and 360 degrees.
#. The stellar metallicity is not sampled, but taken as a constant value equal to ``st_metfe``, or 0.0, if undefined. 
#. If the planet radius in KnownPlanets was calculated from the mass, and the mass (``pl_bmassj``) represents :math:`M\sin I`, the mass sample is defined as ``pl_bmassj``:math:`/sin(I)`.
#. The mean anomaly is sampled from a uniform distribution between 0 and 360 degrees.
#. The orbital radius, projected separation, phase angle, and :math:`\Delta\textrm{mag}` are now calculated for each sample exactly as they were for the PlanetOrbits table (see :ref:`Steps 9 - 13<orbcalcref>`). 
#. The cloud :math:`f_\textrm{sed}` is sampled as described in the Planet Photometry discussion (:ref:`see above<sampleclouds>`).
#. A 2D histogram is constructed from the phase angle and :math:`\Delta\textrm{mag}` sampled on a predefined grid with angular separation sampled from 150 to 450 mas in increments of 1 mas and  :math:`\Delta\textrm{mag}` sampled from 0 to 26 in increments of 0.1.
#. A single completeness value is calculated by finding the fraction of sampled points such that the angular separation falls between 150 and 450 mas and the  :math:`\Delta\textrm{mag}` is less than or equal to a curve defined from the 575 nm WFIRST predicted performance.
#. The sampling procedure is repeated (drawing :math:`n` samples at a time) and the histogram and bulk completeness value are updated until the completeness converges to within 0.01%.
#. The Completeness table is defined with the following columns:

    * Name: Planet Name, references KnownPlanets ``pl_name``
    * alpha: Angular Separations at grid centers (mas)
    * dMag: :math:`\Delta\textrm{mag}` values at grid centers
    * H: :math:`\log_{10}(\mathrm{Frequency})`
    * iind: Horizontal index of entry in 2D histogram grid
    * jind: Vertical index of entry in 2D histogram grid

   The iind and jind values are stored only to make it simpler to reconstruct the 2D grid for plotting.

#. For each planet whose completeness is calculated, we also calculate the minimum and maximum angular separations and :math:`\Delta\textrm{mag}` values for which the frequency grid is non-zero.  These values are then stored in new columns appended to the KnownPlanets table, named: ``compMinWA``, ``compMaxWA``, ``compMindMag``, and ``compMaxdMag``, respectively. The bulk completeness is stored in the KnownPlanets table in a new column named ``completeness``.

.. _blindsearch:

Blind Search Targets
=============================


BlindTargs Table
-----------------------

To identify top priority targets for WFIRST CGI imaging where no planets are yet known to exist (known as 'Blind Search Targets'), we start by analyzing the results of all currently available EXOSIMS blind search simulations using a consistent description of the optical system (based on Cycle 6 values).  This data set is made up of approximately 149,000 individual end-to-end mission simulations with identical observatory and optical system descriptions and input target catalog (ExoCat-1 [Turnbull2015]_) but varying planet population models, planet physical models, mission durations, mission operating rules, and target scheduling/prioritization algorithms.  

Of this corpus of data, we identify the distinct number of times a sub-Neptune class planet (:math:`R < 3.883 R_\oplus`) is detected about each target star.  This total number of detections is normalized by the total number of simulations used, and the resulting value is designated the target priority.  The target priority thus encodes not only the likelihood of observing a sub-Neptune planet about a given star, but also the schedulability and the ability to achieve high contrast in the observing time available for that target.  The priority does *not* map directly to the probability of WFIRST CGI detecting a sub-Neptune planet about a given target, given that one exists, but is a closely related metric that can be used for blind search target prioritization. 

This analysis yields a total of 120 targets without known planets for which the priority is non-zero. In addition to the priority and target data available from ExoCat-1, we add two additional calculated columns:

#. Using the tabulated data from Erik Mamajek [Mamajek2018]_ (based in part on [Pecaut2013]_), we create interpolants for stellar radius as a function of spectral type (the interpolant is on the numerical subtype, with a separate interpolant for each spectral class).  The stellar radius for each of the targets is then calculated via the relevant interpolant based on the spectral type listed for the target in ExoCat-1.
#. From the target radius (:math:`R`) and distance (:math:`d`), the target angular radius is then calculated as:

   .. math::
        \tan^{-1}\left( \frac{R}{d} \right)

We define the BlindTargs Table as follows:

   * Name: Target name (from ExoCat-1)
   * Priority: Target Priority
   * dist: Target Distance (pc; from ExoCat-1)
   * spec: Target spectral type (from ExoCat-1)
   * Vmag: Target V band apparent magnitude (from ExoCat-1)
   * L: Target luminosity in solar luminosities (from ExoCat-1)
   * RA: Target RA (deg; from ExoCat-1)
   * DEC: Target DEC (deg; from ExoCat-1)
   * radius: Target radius (solar radii)
   * angrad: Target angular radius (mas)


DoS Table
----------------

For each target in BlindTargs, we also wish to calculate the depth of search for the WFIRST CGI, as defined in [Garrett2017]_. We establish a grid uniform in :math:`\log(a),\log(R)` where :math:`a` is the semi-major axis and :math:`R` is the planet radius, such that :math:`a \in [0.1, 100]` AU, with 100 bins, and :math:`R \in [1,22.6] \, R_\oplus` with 30 bins. For each bin, we define an average contrast limit by taking the WFIRST CGI predicted performance curve at the 575 nm band and finding the expectation value of the value of this curve for the distribution of angular separations for that bin.

For each bin, the depth of search (DoS) is calculated by averaging the completeness over the bin area, as defined in equation (64) of [Garrett2017]_, using the individual average contrast limits in each bin, and marginalizing over the photometric model grids (see  :ref:`photometry`).

   .. note::

    The DoS calculation assumes only circular orbits, which is why the DoS does not extend past the IWA in semi-major axis.  Semi-major axis values larger than the OWA are still possible with circular orbits due to projection effects.  The inclusion of non-zero eccentricities does not significantly affect the results of this calculation, save that a very small number of detections become possible for smas within the IWA.

The DoS values are stored in the database in the same way as the :ref:`comptable`:

    * Name: Target Name, references BlindTargs ``Name``
    * sma: Semi-major axes at grid geometric centers (AU)
    * Rp: Planet radii at grid geometric centers (:math:`R_\oplus`)
    * vals: :math:`\log_{10}(DoS)`
    * iind: Horizontal index of entry in 2D grid
    * jind: Vertical index of entry in 2D grid

The iind and jind values are stored only to make it simpler to reconstruct the 2D grid for plotting.

Finally, an additional column is added to BlindTargs:  ``sumDoS``, equal to the sum over all DoS bins for that target where :math:`R < 3.883 R_\oplus` (a total of 130 bins).  This value represents the expected number of planet observable about a given target upon the first observation *if* we were able to achieve the assumed average bin contrast for all bins *and* the target had one planet in each of those bins.  

   .. note::
   
    It is important to note that while the Priority and sumDoS columns generally track one another, they are not monotonically related.  There are several instances (especially for dimmer targets) where the sumDoS would prioritize a target much higher than its Priority value.  This is due to the fact that, in practice, the contrast floors are probably unachievable for those star with the available integration and target visibility times.





Utility Tables
=================

Alias Table
--------------------------

The Alias table contains all available alternate target names to simplify identification.  The aliases are found by scraping the alias table in the Exoplanet Archive (for known planet targets), and via the SIMBAD name resolver.

For the subset of unique ``pl_hostname`` entries in KnownPlanets:

#. SIMBAD is queried with the star name in ``pl_hostname``.
#. The Exoplanet Archive is queried via ::

    https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=aliastable&objname=starname

#. The two queries are concatenated and a unique list of aliases generated from the unified list. 
#. A unique id number is assigned to the target and the alias table is defined as:

   * SID: Target ID
   * Alias: Target Name
   * NEAName: Binary flag, true for entry matching NASA Exoplanet Archive default name (or EXOCAT default name)

The alias list for every target is guaranteed to include the name in ``pl_hostname`` and so every target's aliases can be resolved by querying on its own name.




Database Use
====================

A detailed schema of the full database is available at https://plandb.sioslab.com/docs/plandbschema/index.html

You can also use the general query page to get information about the database by querying ``show tables`` for a list of all available tables and ``show full columns in TableName`` for a description of all columns in a table (e.g., ``show full columns in KnownPlanets``).

A few sample queries:

    * Select all known planets with non-zero WFIRST completeness and star magnitudes of less than 7:  ::

        select pl_name,  completeness, st_optmag from KnownPlanets where completeness is not NULL and st_optmag <= 7 order by completeness DESC

    * Select everything from the ``BlindTargs`` table: ::

        select * from BlindTargs

    * Get HD catalog numbers of all blind search targets: ::

        select alias as HD from Aliases where SID  in (select SID from Aliases right join BlindTargs ON BlindTargs.Name = Aliases.Alias) and Alias like '%HD%' 
        



   
References
=============
.. [Batalha2018] Batalha, N. E., Smith, A. J. R. W., Lewis, N. K., Marley, M. S., Fortney, J. J. and Macintosh, B. (2018) Color Classification of Extrasolar Giant Planets: Prospects and Cautions, AJ 156(4)
.. [Marley1999a] Marley, M. S., Gelino, C., Stephen, D., Lunine, J. I., and Freedman, R. (1999) Reflected spectra and albedos of extrasolar giant planets. I. Clear and cloudy atmospheres, ApJ 513(2)
.. [Marley1999b] Marley, M. S. and McKay, C. P. (1999) Thermale Structure of Uranus' Atmosphere, Icarus 138
.. [Ackerman2001] Ackerman, A. S. and Marley, M. S. (2001) Precipitating Condensation Clouds in Substellar Atmospheres, ApJ 556
.. [Cahoy2010] Cahoy, K. L., Marley, M. S. and Fortney, J. J. (2010) Exoplanet albedo spectra and colors as a function of planet phase, separation, and metallicity, ApJ 724
.. [MacDonald2018] MacDonald, R. J., Marley M. S., Fortney, J. J. and Lewis, N. K. (2018) Exploring H20 Prominence in Reflection Spectra of Cool Giant Planets, ApJ 858
.. [Chen2016] Chen, J. and Kipping, D. M. (2016) Probabilistic Forecasting of the Masses and Radii of Other Worlds, ApJ 834(1)
.. [Fortney2007] Fortney, J. J., Marley, M. S. and Barnes, J. W. (2007) Planetary Radii across Five Orders of Magnitude in Mass and Stellar Insolation: Application to Transits, ApJ 659, 2
.. [Brown2005] Brown, R. A. (2005) Single-visit photometric and obscurational completeness, ApJ 624
.. [Garrett2016] Garett, D. and Savransky, D. (2016) Analytical Formulation of the Single-visit Completeness Joint Probability Density Function, ApJ 828(1); https://arxiv.org/abs/1607.01682
.. [Turnbull2015] Turnbull, M. (2015) ExoCat-1: The Nearby Stellar Systems Catalog for Exoplanet Imaging Missions, https://arxiv.org/pdf/1510.01731.pdf
.. [Mamajek2018] Mamajek, E. (2018) A Modern Mean Dwarf Stellar Color and Effective Temperature Sequence, http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
.. [Pecaut2013] Pecaut, M. J. and  Mamajek, E. (2013) Intrinsic Colors, Temperatures, and Bolometric Corrections of Pre-main-sequence Stars, ApJS, 208, 9
.. [Garrett2017] Garett, D., Savransky, D., and Macintosh, B. (2017) A Simple Depth-of-Search Metric for Exoplanet Imaging Surveys, AJ 154(2); https://arxiv.org/abs/1706.06132

    


