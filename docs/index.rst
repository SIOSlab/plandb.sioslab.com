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

Integration over filter bandpasses is currently implemented as a left-Riemann sum, so that:

    .. math::
        
        p\Phi(\beta)(\lambda, \Delta\lambda) = \frac{1}{\Delta\lambda} \frac{\Delta\lambda}{n-1}\sum_{i = 0}^{n-1} f_{m,c,d,\beta}\left(\lambda - \frac{\Delta\lambda}{2} + i\frac{\Delta\lambda}{n-1}  \right)

where :math:`f_{m,c,d,\beta}` is the relevant interpolant over the model grid on the metallicity, cloud, orbital distance and phase angle axes, and :math:`n` is taken to be 100.

    .. note::
        
        This is a pretty crude integral approximation, but the associated error is significantly lower than all other errors involved here, so there's not much point in changing it to something more complex.  Straight quadrature over the interpolant is very fragile given the grids, and takes excessively long to compute.

Known Planet Tables
=============================


KnownPlanets Table
-------------------------------

The KnownPlanets table contains all of the target star and known planet properties that are needed to calculate direct imaging observables.  This describes the procedure for creating the table.

1. All data is grabbed from the Exoplanet Archive from the exoplanets, composite, and extended tables via the following API queries: ::

    https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=compositepars&select=*&format=csv"
    https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=exoplanets&select=*&format=csv"
    https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=exomultpars&select=*&format=csv"

2. The leading 'f' is stripped off all data column names for the composite table data and the leading 'm' is stripped off all data column names for the extended table to match the names in the exoplanets table data.
   
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

4. The composite and exoplanet data sets are merged with the composite table values used preferentially when both are populated.


5. For planets that were previously calculated to have a completeness larger than 0, data is substituted in from the extended table. For each planet of interest, we categorize each source according to the priority listing below. We then substitute data into the joined data from the source with the highest priority.  If two sources are tied for the highest priority, then the most recent one is chosen.


   1. Satisfies the criteria described in step 6 (barring the existance of stellar distance) and has non-null eccentricity, time of periaps passage, argument of periapsis, and inclination. 
   
   2. Satisfies the criteria described in step 6 (barring the existance of stellar distance) and has non-null eccentricity, time of periaps passage, and argument of periapsis.
   
   3. Satisfies the criteria described in step 6, (barring the existance of stellar distance) has non-null eccentricity, and has either non-null time of periaps passage or longitude of periapsis.
	
   4. Satisfies the criteria described in step 6 (barring the existance of stellar distance) and has non-null eccentricity.
	
   5. Satisfies the criteria described in step 6 (barring the existance of stellar distance).


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


6. The data is filtered such that the remaining rows have:
    
    |   Star Distance AND
    |   (Planet Semi-Major Axis OR (Planet Period AND Star Mass)) AND 
    |   (Planet Radius OR Planet Mass OR Planet :math:`m\sin(I))`.


7. For the remaining rows that do not have a planet semi-major axis, it is filled in as follows: The `st_mass` value is taken to be the stellar mass (:math:`M_S`) in units of :math:`M_\odot` and the gravitational parameter is calculated as :math:`\mu = G M_S`. The ``pl_orbper`` value is taken to be the orbital period (:math:`T`) in units of days. The semi-major axis is then calculated as: 
   
   .. math::

    \left( \frac{ \mu T^2 }{ 4\pi^2 }\right)^\frac{1}{3}
    
  The values are converted to AU and copied to the `pl_orbsmax`` column.  The ``pl_orbsmaxreflink`` for these rows is updated to: "Calculated from stellar mass and orbital period."


8. The angular separation (``pl_angsep``) is re-calculated for all rows based on the current entries of ``pl_orbsmax`` (:math:`a`) and ``st_dist`` (:math:`d`) as:
    
   .. math::
    
    \tan^{-1}\left( \frac{a}{d} \right)

  with the result converted to miliarcseconds. 


9. The composite data table has radius values computed from available masses using the Forecaster best-fit ([Chen2016]_).  This tends to overestimate the radius for many Jovian-size planets, so we provide two additional mass estimates (but leave the original column).  We will only carry along masses in units of :math:`M_J`, and so drop columns ``pl_rade,  pl_radelim, pl_radserr2, pl_radeerr1, pl_rads, pl_radslim, pl_radeerr2, pl_radserr1``.

.. _forecastermodref:

10. The original Forecaster ([Chen2016]_) best fit is composed of linear (in log-log space) segments of the form  :math:`R = 10^{\mathcal{C} + \log_{10}(M)\mathcal{S}}` where :math:`R` and :math:`M` are the radius and mass, respectively, and  :math:`\mathcal{C}` and :math:`\mathcal{S}` are fit coefficients defined in four intervals of mass (Terran, Neptunian, Jovian and Stellar Worlds).  We modify the Forecaster fit as follows:
   
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

11. The second radius fit is based on the planet density models from [Fortney2007]_. The relevant masses are converted to radii as follows:

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

12. For rows with eccentricity information available, a maximum possible angular separation is calculated as:

    .. math::
        
        tan^{-1}\left(\frac{a(1 + e)}{d}\right)
   
   where :math:`a` is the semi-major axis, :math:`d` is the target distance and  :math:`e` is the eccentricity. These values are assigned to a new column ``pl_maxangsep``, which equals ``pl_angsep`` for rows with no eccentricity available. 

13. A minimum possible angular separation is calculated as:

    .. math::
        
        tan^{-1}\left(\frac{s}{d}\right)
   
   where :math:`d` is the target distance and  :math:`s` is the minimum possible projected separation.  :math:`s` is calculated by taking the semi-major axis and scaling it by :math:`s(1 - e)` for those rows with eccentricity available.  This product is then further scaled by :math:`\cos(I)` for those rows with inclination available.  For those rows without inclination values, :math:`I` is assumed to be 90 degrees and  :math:`s`  is set to zero.

.. _photcalcref:

14. We now calculate fiducial values of angular separation and :math:`\Delta\textrm{mag}` at quadrature (planet phase of 90 degrees).  We have two approaches for calculating the quadrature radius depending on what data exists for the planet.  If the planet has data for eccentricity and argument of periapsis and the inclination data is either null or non-zero, then we calculate the true anomaly using the fact that :math:`\theta = \omega+\nu` and:

	.. math::
			\theta = \sin^{-1}\left(\frac{\cos\beta}{\sin I}\right)

   This gives us two solutions for the true anomaly, :math:`\nu = -\omega` and :math:`\nu = \pi - \omega`.  Two possible values for the radius at quadrature are calculated as: 

	.. math::

			r = \frac{a(1 - e^2)}{1 + e\cos\nu}
			
	
   From these two radius values, we pick the radius that results in the smallest :math:`\Delta\textrm{mag}` (as calculated below) so that we determine the highest possible brightness for the planet.  
   
   In the case where eccentricity or argument of periapsis are null, or if the orbit is face-on, then we take the semi-major axis to be the orbial radius at quadrature.

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

1. From KnownPlanets ``pl_orbsmax`` is taken as the semi-major axis :math:`a`.
2. ``pl_orbeccen`` is taken as the eccentricity :math:`e`.  If the eccentricity is undefined, it is set to zero. 
3. ``pl_orbincl`` is taken as the inclination :math:`I`; if it is undefined it is set to 90 degrees. 
4. ``pl_orblper`` is taken as the argument of periapsis :math:`\omega`; if it is undefined it is set to zero. 

   .. warning::
 
    **N.B: this is forcing the assumption of longitude of the ascending node equaling zero.** 

5. ``pl_radj_forecastermod`` is taken to be the planet radius :math:`R`. 
   
   .. note::

    We can replace another mass fit here.
    
6. ``st_dist`` is taken to be the target distance :math:`d`.
7. ``st_metfe`` is taken to be the stellar metallicity; if it is undefined it is set to zero.
8. The eccentric anomaly :math:`E` is calculated from the eccentricity for 100 equally spaced mean anomaly points :math:`M` between 0 and 360 degrees, inclusive.  The calculation is done via routine ``eccanom`` from EXOSIMS, which is a machine-precision Newton-Raphson inversion. True anomaly is calculated from the eccentric anomaly as:

   .. math::

        \nu = 2\tan^{-1}\left(\sqrt{\frac{1+e}{1 - e}}\tan\left(\frac{E}{2}\right)\right)
  

.. _orbcalcref:

9.  The orbital radius :math:`r` is calculated as:

    .. math::

        r = \frac{a(1 - e^2)}{1 + e\cos\nu}

10. The projected separation :math:`s` is calculated as:

    .. math::
        
        s = \frac{r \sqrt{4 \cos{\left (2 I \right )} + 4 \cos{\left (2 \nu + 2 \omega \right )} - 2 \cos{\left (- 2 I + 2 \nu + 2 \omega \right )} - 2 \cos{\left (2 I + 2 \nu + 2 \omega \right )} + 12}}{4}

11. The phase angle for all anomaly values is calculated as 
    
    .. math::
        \beta = \cos^{-1}\left(\sin I\sin\theta\right)
    
   where :math:`\theta = \omega+\nu` and the angular separation is calculated as 
     
    .. math::
        \alpha = \tan^{-1}\left( \frac{s}{d} \right)

12. The PlanetOrbits table is defined with the following columns:
    
    * M: :math:`M` (rad)
    * r: :math:`r` (AU)
    * s: :math:`s` (AU)
    * WA: :math:`\alpha` (mas)
    * beta: :math:`\beta` (deg)

13. For each row, for each cloud level in the :ref:`photometry` grids, and for each wavelength of interest, the :math:`p\Phi(\beta)` value is interpolated and a :math:`\Delta\textrm{mag}` value is calculated as described in :ref:`Step 13<photcalcref>` of the KnownPlanets generation procedure. These values are stored in new columns ``pPhi_XXXC_YYYNM`` and ``dMag_XXXC_YYYNM`` where ``XXX`` is the cloud  :math:`f_\textrm{sed}` scaled by 100 (000 representing no cloud) and ``YYY`` is the wavelength in nm. 

.. _comptable:

Completeness Table
--------------------------

The Completeness table contains 2D frequency maps of the joint distribution of planet projected separation and :math:`\Delta\textrm{mag}` (for excruciating detail see [Brown2005]_ and [Garrett2016]_).  Currently, values are only generated for planets where ``pl_maxangsep`` is greater or equal to 100 mas and ``pl_minangsep`` is less than or equal to 500 mas, **however this calculation can be extended to all planets**.

For each planet meeting this condition, the following samples are drawn, :math:`n=1\times10^{6}` at a time:

1. Semi-major axis is sampled from a normal distribution, with :math:`\mu =` ``pl_orbsmax`` and :math:`\sigma =` (``pl_orbsmaxerr1`` :math:`-` ``pl_orbsmaxerr2``)/2.  In the case where the error is undefined, :math:`\sigma = \mu/100`.
2. Eccentricity is sampled from a normal distribution, with :math:`\mu =` ``pl_orbeccen`` and :math:`\sigma =` (``pl_orbeccenerr1`` :math:`-` ``pl_oreccenerr2``)/2.  In the case where the error is undefined, :math:`\sigma = \mu/100`. In the case where the eccentricity is undefined, the distribution is taken to be a Rayleigh distribution with :math:`\sigma = 0.0125`.
3. The inclination is sampled from a normal distribution, with :math:`\mu =` ``pl_orbincl`` and :math:`\sigma =` (``pl_orbinclerr1`` :math:`-` ``pl_orbinclerr2``)/2.  In the case where the error is undefined, :math:`\sigma = \mu/100`. In the case where the inclination is undefined, the inclination is sampled from a sinusoidal distribution. For rows where the planet mass (``pl_bmassj``) represents an estimate of the true mass, this sinusoidal inclination is sampled over the full range from 0 to 180 degrees.  However, in cases where the planet mass represents :math:`M\sin I`, a critical inclination value is calculated at which the true planet mass would equal 0.08 Solar masses, and the distribution is sampled only between this critical inclination and 180 degrees minus this critical inclination. 
4. The longitude of periapsis is sampled from a normal distribution, with :math:`\mu =` ``pl_orblper`` and :math:`\sigma =` (``pl_orblpererr1`` :math:`-` ``pl_orblpererr2``)/2.  In the case where the error is undefined, :math:`\sigma = \mu/100`. In the case where the argument of periapsis is undefined, it is sampled from a uniform distribution between 0 and 360 degrees.
5. The longitude of the ascending node is sampled from a uniform distribution between 0 and 360 degrees.
6. The stellar metallicity is not sampled, but taken as a constant value equal to ``st_metfe``, or 0.0, if undefined. 
7. If the planet radius in KnownPlanets was calculated from the mass, and the mass (``pl_bmassj``) represents :math:`M\sin I`, the mass sample is defined as ``pl_bmassj``:math:`/sin(I)`.
8. The mean anomaly is sampled from a uniform distribution between 0 and 360 degrees.
9. The orbital radius, projected separation, phase angle, and :math:`\Delta\textrm{mag}` are now calculated for each sample exactly as they were for the PlanetOrbits table (see :ref:`Steps 9 - 13<orbcalcref>`). 
10. The cloud :math:`f_\textrm{sed}` is sampled from the following distribution:

* 0/9.9%
* 0.01/0.1%
* 0.03/0.5%
* 0.1/1.0%
* 0.3/2.5%
* 1.0/28%
* 3.0/30%
* 6.0/28%

11. A 2D histogram is constructed from the phase angle and :math:`\Delta\textrm{mag}` sampled on a predefined grid with angular separation sampled from 150 to 450 mas in increments of 1 mas and  :math:`\Delta\textrm{mag}` sampled from 0 to 26 in increments of 0.1.
12. A single completeness value is calculated by finding the fraction of sampled points such that the angular separation falls between 150 and 450 mas and the  :math:`\Delta\textrm{mag}` is less than or equal to a curve defined from the 575 nm WFIRST predicted performance.
13. The sampling procedure is repeated (drawing :math:`n` samples at a time) and the histogram and bulk completeness value are updated until the completeness converges to within 0.01%.
14. The Completeness table is defined with the following columns:

    * Name: Planet Name, references KnownPlanets ``pl_name``
    * alpha: Angular Separations at grid centers (mas)
    * dMag: :math:`\Delta\textrm{mag}` values at grid centers
    * H: :math:`\log_{10}(\mathrm{Frequency})`
    * iind: Horizontal index of entry in 2D histogram grid
    * jind: Vertical index of entry in 2D histogram grid

   The iind and jind values are stored only to make it simpler to reconstruct the 2D grid for plotting.

15. For each planet whose completeness is calculated, we also calculate the minimum and maximum angular separations and :math:`\Delta\textrm{mag}` values for which the frequency grid is non-zero.  These values are then stored in new columns appended to the KnownPlanets table, named: ``compMinWA``, ``compMaxWA``, ``compMindMag``, and ``compMaxdMag``, respectively. The bulk completeness is stored in the KnownPlanets table in a new column named ``completeness``.

.. _blindsearch:

Blind Search Tables
=============================


BlindTargs Table
-----------------------

To identify top priority targets for WFIRST CGI imaging where no planets are yet known to exist (known as 'Blind Search Targets'), we start by analyzing the results of all currently available EXOSIMS blind search simulations using a consistent description of the optical system (based on Cycle 6 values).  This data set is made up of approximately 149,000 individual end-to-end mission simulations with identical observatory and optical system descriptions and input target catalog (ExoCat-1 [Turnbull2015]_) but varying planet population models, planet physical models, mission durations, mission operating rules, and target scheduling/prioritization algorithms.  

Of this corpus of data, we identify the distinct number of times a sub-Neptune class planet (:math:`R < 3.883 R_\oplus`) is detected about each target star.  This total number of detections is normalized by the total number of simulations used, and the resulting value is designated the target priority.  The target priority thus encodes not only the likelihood of observing a sub-Neptune planet about a given star, but also the schedulability and the ability to achieve high contrast in the observing time available for that target.  The priority does *not* map directly to the probability of WFIRST CGI detecting a sub-Neptune planet about a given target, given that one exists, but is a closely related metric that can be used for blind search target prioritization. 

This analysis yields a total of 120 targets without known planets for which the priority is non-zero. In addition to the priority and target data available from ExoCat-1, we add two additional calculated columns:

1. Using the tabulated data from Erik Mamajek [Mamajek2018]_ (based in part on [Pecaut2013]_), we create interpolants for stellar radius as a function of spectral type (the interpolant is on the numerical subtype, with a separate interpolant for each spectral class).  The stellar radius for each of the targets is then calculated via the relevant interpolant based on the spectral type listed for the target in ExoCat-1.
2. From the target radius (:math:`R`) and distance (:math:`d`), the target angular radius is then calculated as:

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

1. SIMBAD is queried with the star name in ``pl_hostname``.
2. The Exoplanet Archive is queried via ::

    https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=aliastable&objname=starname

3. The two queries are concatenated and a unique list of aliases generated from the unified list. 
4. A unique id number is assigned to the target and the alias table is defined as:

   * SID: Target ID
   * Alias: Target Name
   * NEAName: Binary flag, true for entry matching NASA Exoplanet Archive default name (or EXOCAT default name)

The alias list for every target is guaranteed to include the name in ``pl_hostname`` and so every target's aliases can be resolved by querying on its own name.




Database Structure
====================

All tables use the InnoDB engine. ``pl_name`` and ``pl_hostname`` are set as indexes for KnownPlanets.  ``Name`` is set as an index for both PlanetOrbits and Completeness and as a foreign key referencing ``pl_name`` in KnownPlanets.  ``SID`` and ``Alias`` in Alias are set as indexes.

   
References
=============
.. [Chen2016] Chen, J. and Kipping, D. M. (2016) Probabilistic Forecasting of the Masses and Radii of Other Worlds, ApJ 834(1)
.. [Fortney2007] Fortney, J. J., Marley, M. S. and Barnes, J. W. (2007) Planetary Radii across Five Orders of Magnitude in Mass and Stellar Insolation: Application to Transits, ApJ 659, 2
.. [Brown2005] Brown, R. A. (2005) Single-visit photometric and obscurational completeness, ApJ 624
.. [Garrett2016] Garett, D. and Savransky, D. (2016) Analytical Formulation of the Single-visit Completeness Joint Probability Density Function, ApJ 828(1); https://arxiv.org/abs/1607.01682
.. [Turnbull2015] Turnbull, M. (2015) ExoCat-1: The Nearby Stellar Systems Catalog for Exoplanet Imaging Missions, https://arxiv.org/pdf/1510.01731.pdf
.. [Mamajek2018] Mamajek, E. (2018) A Modern Mean Dwarf Stellar Color and Effective Temperature Sequence, http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
.. [Pecaut2013] Pecaut, M. J. and  Mamajek, E. (2013) Intrinsic Colors, Temperatures, and Bolometric Corrections of Pre-main-sequence Stars, ApJS, 208, 9
.. [Garrett2017] Garett, D., Savransky, D., and Macintosh, B. (2017) A Simple Depth-of-Search Metric for Exoplanet Imaging Surveys, AJ 154(2); https://arxiv.org/abs/1706.06132

    


