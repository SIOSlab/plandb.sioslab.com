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
$sql = "SELECT pl_orbper,pl_discmethod,pl_orbsmax,smax_from_orbper,pl_orbeccen,pl_orbincl,pl_bmassj,pl_bmassprov,pl_radj,rad_from_mass,pl_orbtper,pl_orblper,pl_eqt,pl_insol,ra_str,dec_str,st_dist,st_optmag,st_optband,gaia_gmag,st_teff,st_mass,gaia_plx,gaia_dist,st_pmra,st_pmdec,gaia_pmra,gaia_pmdec,st_radv,st_spstr,st_lum,st_metfe,st_age,st_bmvj FROM KnownPlanets WHERE CONCAT(pl_hostname,' ',pl_letter)='".$name."'";

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



echo "</TABLE></DIV>";
?>



<?php include "templates/footer.php"; ?>

