<?php 
$sql = "SELECT 
OF.hostname AS pl_hostname,
-- pl_reflink,
OF.pl_orbper AS pl_orbper,
OF.discoverymethod AS pl_discmethod,
OF.pl_orbsmax AS pl_orbsmax,
OF.pl_orbeccen AS pl_orbeccen,
OF.pl_orbincl AS pl_orbincl,
OF.pl_bmassj AS pl_bmassj,
OF.pl_bmassprov AS pl_bmassprov,
OF.pl_radj AS pl_radj,
-- pl_radreflink,
OF.pl_radj_fortney AS pl_radj_fortney,
OF.pl_radj_forecastermod AS pl_radj_forecastermod,
OF.pl_orbtper AS pl_orbtper,
OF.pl_orblper AS pl_orblper,
OF.pl_eqt AS pl_eqt,
OF.pl_insol AS pl_insol,
OF.pl_angsep AS pl_angsep,
S.minangsep AS pl_minangsep,
S.maxangsep AS pl_maxangsep, #Should this be OF.pl_maxangsep instead
ST.rastr AS ra_str, #These are copied over to orbitfits should they be dropped
ST.decstr AS dec_str,
OF.sy_dist AS st_dist,
OF.sy_plx AS st_plx,
-- gaia_plx,
-- gaia_dist,
OF.sy_vmag AS st_optmag,
-- st_optband,
OF.sy_gaiamag AS gaia_gmag,
ST.teff AS st_teff,
ST.mass AS st_mass,
OF.sy_pmra AS st_pmra,
OF.sy_pmdec AS st_pmdec,
-- gaia_pmra,
-- gaia_pmdec,
ST.radv AS st_radv,
ST.spectype AS st_spstr,
ST.lum AS st_lum,
ST.met AS st_metfe,
ST.age AS st_age,
-- st_bmvj
C.completeness AS completeness,
C.compMinWA AS compMinWA,
C.compMaxWA AS compMaxWA,
C.compMindMag AS compMindMag,
C.compMaxdMag AS compMaxdMag,
OF.elat AS st_elat,
OF.elon AS st_elon 
FROM Stars ST, Planets PL, OrbitFits OF, Completeness C, Scenarios S
WHERE ST.st_id = PL.pl_id
AND PL.pl_id= OF.pl_id
AND PL.pl_id= C.pl_id
AND C.scenario_name= S.scenario_name 
AND PL.pl_name='".$name."'
AND S.scenario_name='".$scenario."'";

?>