<?php include "templates/header.php"; ?>

<?php 
if (empty($_GET["name"])){
    echo "No planet name provided.";
    include "templates/footer.php"; 
    exit;
} else{
    $name = $_GET["name"];
}
?>

<h2> Planet Detail for 
<?php echo $name; ?>
</h2>


<?php
$sql = "SELECT pl_orbper,pl_discmethod,pl_orbsmax,smax_from_orbper,pl_orbeccen,pl_orbincl,pl_bmassj,pl_bmassprov,pl_radj,rad_from_mass,pl_orbtper,pl_orblper,pl_eqt,pl_insol,ra_str,dec_str,st_dist,st_plx,gaia_plx,gaia_dist,st_optmag,st_optband,gaia_gmag,st_teff,st_mass,st_pmra,st_pmdec,gaia_pmra,gaia_pmdec,st_radv,st_spstr,st_lum,st_metfe,st_age,st_bmvj FROM KnownPlanets WHERE CONCAT(pl_hostname,' ',pl_letter)='".$name."'";

include("config.php"); 
$conn = new mysqli($servername, $username, $password, $dbname);
// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
} 
$result = $conn->query($sqlsel.$sql);
if (!$result){
    echo "Query Error:\n".$conn->error;
    $conn->close();
    include "templates/footer.php"; 
    exit;
}
if ($result->num_rows == 0) {
    echo "Planet Not Found.";
    $result->close();
    $conn->close();
    include "templates/footer.php"; 
    exit;
}
if ($result->num_rows > 1) {
    echo "Multiple matches found.";
    $result->close();
    $conn->close();
    include "templates/footer.php"; 
    exit;
}

$row = $result->fetch_assoc();
$wd = '50';
echo " <div style='float: left; width: 90%;'> ";
echo "<TABLE class='results'>";
echo "<TR><TH colspan='2'> Planet Properties</TH></TR>";
echo "<TR><TH style='width:".$wd."%'>Discovered via</TH><TD>".$row[pl_discmethod]."</TD></TR>";
echo "<TR><TH style='width:".$wd."%'>Period (days)</TH><TD>".$row[pl_orbper]."</TD></TR>";
echo "<TR><TH style='width:".$wd."%'>Semi-major Axis (AU)</TH><TD>".$row[pl_orbsmax];
if ($row[smax_from_orbper])
    echo " (calculated from period)";
echo"</TD></TR>";
echo "<TR><TH style='width:".$wd."%'>Eccentricity</TH><TD>".$row[pl_orbeccen]."</TD></TR>";
echo "<TR><TH style='width:".$wd."%'>Inclination (deg)</TH><TD>".$row[pl_orbincl]."</TD></TR>";
echo "<TR><TH style='width:".$wd."%'>".$row[pl_bmassprov]." (Jupiter Masses)</TH><TD>".$row[pl_bmassj]."</TD></TR>";
echo "<TR><TH style='width:".$wd."%'>Radius (Jupiter Radii)</TH><TD>".$row[pl_radj];
if ($row[rad_from_mass])
    echo " (estimated from mass)";
echo"</TD></TR>";
echo "<TR><TH style='width:".$wd."%'>Periapsis Passage Time (JD)</TH><TD>".$row[pl_orbtper]."</TD></TR>";
echo "<TR><TH style='width:".$wd."%'>Longitude of Periapsis (deg)</TH><TD>".$row[pl_orblper]."</TD></TR>";
echo "<TR><TH style='width:".$wd."%'>Equilibrium Temperature (K)</TH><TD>".$row[pl_eqt]."</TD></TR>";
echo "<TR><TH style='width:".$wd."%'>Insolation Flux (Earth fluxes)</TH><TD>".$row[pl_insor]."</TD></TR>";
echo "</TABLE>";


echo "<TABLE class='results'>";
echo "<TR><TH colspan='2'> Star Properties</TH></TR>";
echo "<TR><TH style='width:".$wd."%'>RA, DEC</TH><TD>".$row[ra_str].", ".$row[dec_str]."</TD></TR>";
echo "<TR><TH style='width:".$wd."%'>Distance (GAIA Distance) (pc)</TH><TD>".$row[st_dist]." (".$row[gaia_dist].")</TD></TR>";
echo "<TR><TH style='width:".$wd."%'>Parallax (GAIA Parallax) (mas)</TH><TD>".$row[st_plx]." (".$row[gaia_plx].")</TD></TR>";
echo "<TR><TH style='width:".$wd."%'>Proper Motion RA/DEC (GAIA PM) (mas/yr)</TH><TD>".$row[st_pmra].", ".$row[st_pmdec]." (".$row[gaia_pmra].", ".$row[gaia_pmdec].")</TD></TR>";
echo "<TR><TH style='width:".$wd."%'>Radial Velocity (km/s)</TH><TD>".$row[st_radv]."</TD></TR>";
echo "<TR><TH style='width:".$wd."%'>".$row[st_optband]. " band Magnitude</TH><TD>".$row[st_optmag]."</TD></TR>";
echo "<TR><TH style='width:".$wd."%'>GAIA G band Magnitude</TH><TD>".$row[gaia_gmag]."</TD></TR>";
echo "<TR><TH style='width:".$wd."%'>Effective Temperature (K)</TH><TD>".$row[st_teff]."</TD></TR>";
echo "<TR><TH style='width:".$wd."%'>Mass (Solar Masses)</TH><TD>".$row[st_mass]."</TD></TR>";
echo "<TR><TH style='width:".$wd."%'>Spectral Type</TH><TD>".$row[st_spstr]."</TD></TR>";
echo "<TR><TH style='width:".$wd."%'>Luminosity  (Solar Luminosities)</TH><TD>".$row[st_lum]."</TD></TR>";
echo "<TR><TH style='width:".$wd."%'>Metallicity (dex)</TH><TD>".$row[st_metfe]."</TD></TR>";
echo "<TR><TH style='width:".$wd."%'>Age (Gyr)</TH><TD>".$row[st_age]."</TD></TR>";
echo "<TR><TH style='width:".$wd."%'>B-V (Johnson) (mag)</TH><TD>".$row[st_bmvj]."</TD></TR>";

echo "</TABLE>";

    
echo "</DIV>";

?>



<?php include "templates/footer.php"; ?>

