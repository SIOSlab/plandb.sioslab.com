<?php include "templates/header.php"; ?>

<?php 
if (empty($_GET["name"])){
    echo "No planet name provided.";
    include "templates/footer.php"; 
    exit;
} else{
    $name = $_GET["name"];
}

$sql = "SELECT pl_hostname,pl_orbper,pl_orbperreflink,pl_discmethod,pl_orbsmax,pl_orbsmaxreflink,pl_orbeccenreflink,pl_orbeccen,pl_orbincl,pl_bmassj,pl_bmassprov,pl_bmassreflink,pl_radj,pl_radreflink,pl_orbtper,pl_orblper,pl_eqt,pl_insol,pl_angsep,pl_minangsep,pl_maxangsep,ra_str,dec_str,st_dist,st_plx,gaia_plx,gaia_dist,st_optmag,st_optband,gaia_gmag,st_teff,st_mass,st_pmra,st_pmdec,gaia_pmra,gaia_pmdec,st_radv,st_spstr,st_lum,st_metfe,st_age,st_bmvj,completeness,compMinWA,compMaxWA,compMindMag,compMaxdMag FROM KnownPlanets WHERE pl_name='".$name."'";

include("config.php"); 
$conn = new mysqli($servername, $username, $password, $dbname);
// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
} 
$result = $conn->query($sqlsel.$sql);
if (!$result){
    include "templates/headerclose.php"; 
    echo "Query Error:\n".$conn->error;
    $conn->close();
    include "templates/footer.php"; 
    exit;
}
if ($result->num_rows == 0) {
    include "templates/headerclose.php"; 
    echo "Planet Not Found.";
    $result->close();
    $conn->close();
    include "templates/footer.php"; 
    exit;
}
if ($result->num_rows > 1) {
    include "templates/headerclose.php"; 
    echo "Multiple matches found.";
    $result->close();
    $conn->close();
    include "templates/footer.php"; 
    exit;
}

$sql2 = "select * from PlanetOrbits where Name = '".$name."'";
$resultp = $conn->query($sql2);

$row = $result->fetch_assoc();
if ($row[completeness]){
    $sql3 = "SELECT * FROM Completeness WHERE Name='".$name."'";
    $resultc = $conn->query($sql3);

}

$sqlaliases = "select Alias from Aliases where SID = (select SID from Aliases where Alias = '".$row[pl_hostname]."')";
$resultaliases = $conn->query($sqlaliases);

if (($resultp && ($resultp->num_rows > 0)) || $row[completeness]) {
    echo '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>';
}

$result->close();

include "templates/headerclose.php"; 
?>

<h2> Planet Detail for 
<?php echo $name; ?>
</h2>
<p>
 <form action="planorbitsave.php" method="POST">
    <?php 
    echo '<button type="submit" name="name" value="'.$name.'">Save All Orbit Data</button>';
    ?>
 </form>
</p>

<div class="container">
<?php
$wd = '50';
echo " <div style='float: left; width: 90%; margin-bottom: 2em;'>\n";
echo "<TABLE class='results'>\n";
echo "<TR><TH colspan='2'> Planet Properties</TH></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Discovered via</TH><TD>".$row[pl_discmethod]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Period (days)</TH><TD>".
    number_format((float)$row[pl_orbper], 2, '.', '');
if ($row[pl_orbperreflink])
    echo " (".$row[pl_orbperreflink].")";
echo "</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Semi-major Axis (AU)</TH><TD>".
    number_format((float)$row[pl_orbsmax], 2, '.', '');
if ($row[pl_orbsmaxreflink])
    echo " (".$row[pl_orbsmaxreflink].")";
echo"</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Eccentricity</TH><TD>".$row[pl_orbeccen];
if ($row[pl_orbeccenreflink])
    echo " (".$row[pl_orbeccenreflink].")";
echo "</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Inclination (deg)</TH><TD>".$row[pl_orbincl]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>".$row[pl_bmassprov]." (Jupiter Masses)</TH><TD>".$row[pl_bmassj];
if ($row[pl_orbsmaxreflink])
    echo " (".$row[pl_orbsmaxreflink].")";
echo "</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Radius (Jupiter Radii)</TH><TD>".$row[pl_radj];
if ($row[pl_radreflink])
    echo " (".$row[pl_radreflink].")";
echo "</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Periapsis Passage Time (JD)</TH><TD>".$row[pl_orbtper]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Longitude of Periapsis (deg)</TH><TD>".$row[pl_orblper]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Equilibrium Temperature (K)</TH><TD>".$row[pl_eqt]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Insolation Flux (Earth fluxes)</TH><TD>".$row[pl_insor]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Angular Separation @ sma (mas)</TH><TD>".
    number_format((float)$row[pl_angsep], 2, '.', '')."</TD></TR>\n";
//echo "<TR><TH style='width:".$wd."%'>Minimum Angular Separation (mas)</TH><TD>".$row[pl_minangsep]."</TD></TR>\n";
//echo "<TR><TH style='width:".$wd."%'>Maximum Angular Separation (mas)</TH><TD>".$row[pl_maxangsep]."</TD></TR>\n";
echo "</TABLE>\n";


echo "<TABLE class='results'>\n";
echo "<TR><TH colspan='2'> Star Properties</TH></TR>\n";
echo "<TR><TH style='width:".$wd."%'>RA, DEC</TH><TD>".$row[ra_str].", ".$row[dec_str]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Distance (GAIA Distance) (pc)</TH><TD>".$row[st_dist]." (".$row[gaia_dist].")</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Parallax (GAIA Parallax) (mas)</TH><TD>".$row[st_plx]." (".$row[gaia_plx].")</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Proper Motion RA/DEC (GAIA PM) (mas/yr)</TH><TD>".$row[st_pmra].", ".$row[st_pmdec]." (".$row[gaia_pmra].", ".$row[gaia_pmdec].")</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Radial Velocity (km/s)</TH><TD>".$row[st_radv]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>".$row[st_optband]. " band Magnitude</TH><TD>".$row[st_optmag]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>GAIA G band Magnitude</TH><TD>".$row[gaia_gmag]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Effective Temperature (K)</TH><TD>".$row[st_teff]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Mass (Solar Masses)</TH><TD>".$row[st_mass]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Spectral Type</TH><TD>".$row[st_spstr]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Luminosity  (Solar Luminosities)</TH><TD>".$row[st_lum]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Metallicity (dex)</TH><TD>".$row[st_metfe]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Age (Gyr)</TH><TD>".$row[st_age]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>B-V (Johnson) (mag)</TH><TD>".$row[st_bmvj]."</TD></TR>\n";

if ($resultaliases){
    if ($resultaliases->num_rows > 0) {
        echo "<TR><TH style='width:".$wd."%'>Aliases</TH><TD>";
        while($rowa = $resultaliases->fetch_assoc()) {
            echo $rowa['Alias']."; &nbsp; ";
        }
        echo "</TD></TR>\n";
    }
    $resultaliases->close();
}

echo "</TABLE>\n";
    
echo "</DIV><br><br>\n";

echo '<DIV style="clear: both;"></DIV>';

if ($resultp){
    if ($resultp->num_rows == 0){
        echo "No PlanetOrbit rows returned.";
    } else{
        echo '<div id="plot1Div" style="width:500px; height:500px; float:left;"></div>';
        echo '<div id="plot2Div" style="width:500px; height:500px; float:left;"></div>';
        echo "\n\n";
        echo "<script>\n";
        echo "var xsize = 100, x = new Array(xsize), y1 = new Array(xsize),
              y2 = new Array(xsize), y3 = new Array(xsize), y4 = new Array(xsize), 
              y5 = new Array(xsize), y6 = new Array(xsize); \n";
        $i = 0;
        while($rowp = $resultp->fetch_assoc()) {
            echo "x[".$i."]=".$rowp[M].";";
            echo "y1[".$i."]=".$rowp[dMag_100C_575NM].";";
            echo "y2[".$i."]=".$rowp[dMag_000C_575NM].";\n";
            echo "y3[".$i."]=".$rowp[pPhi_000C_575NM].";";
            echo "y4[".$i."]=".$rowp[pPhi_100C_575NM].";\n";
            echo "y5[".$i."]=".$rowp[r].";";
            echo "y6[".$i."]=".$rowp[WA].";\n";
            $i++;
        }

        echo "var trace1 = {\n
                x: x,
                y: y1,
                type: 'scatter',
                name: '\u0394 mag Full Clouds 575 nm',
                line: { color: 'red' }
              };\n
              var trace2 = {
                x: x,
                y: y2,
                fill: 'tonexty',
                type: 'scatter',
                name: '\u0394 mag No Clouds 575 nm',
                line: { color: 'red', dash: 'dot' }
              };\n
              var trace3 = {\n
                x: x,
                y: y3,
                type: 'scatter',
                name: 'p* \u03A6 (\u03B2) No Clouds 575 nm',
                line: { color: 'blue' },
                yaxis: 'y2'
              };\n
              var trace4 = {
                x: x,
                y: y4,
                fill: 'tonexty',
                type: 'scatter',
                name: 'p* \u0394 mag Full Clouds 575 nm',
                line: { color: 'blue', dash: 'dot' },
                yaxis: 'y2'
              };\n


             var data = [trace1, trace2, trace3, trace4];\n

             var layout = {\n
                xaxis: {title: 'Mean Anomaly (rad)',
                        tickvals:[0,Math.PI/2,Math.PI,3*Math.PI/2,2*Math.PI],
                        ticktext:['0', '\u03C0/2', '\u03C0', '3\u03C0/2', '2\u03C0'] },
                yaxis: {title: '\u0394 mag', titlefont: {color: 'red'}, tickfont: {color: 'red'}},
                yaxis2: {title: 'p * \u03A6 (\u03B2)', titlefont: {color: 'blue'}, tickfont: {color: 'blue'}, overlaying: 'y', side: 'right'},
                legend: {x: 0, y:-0.25, 'orientation': 'h'},
                margin: { t: 30, b:0}
             };\n

             Plotly.newPlot('plot1Div', data, layout);\n";

        echo "var trace5 = {\n
                x: x,
                y: y5,
                type: 'scatter',
                name: 'Orbital Radius (AU)',
                line: { color: 'red' }
              };\n
              var trace6 = {\n
                x: x,
                y: y6,
                type: 'scatter',
                name: 'Angular Separation (mas)',
                line: { color: 'blue' },
                yaxis: 'y2'
              };\n

             var data2 = [trace5, trace6];\n

             var layout2 = {\n
                xaxis: {title: 'Mean Anomaly (rad)',
                        tickvals:[0,Math.PI/2,Math.PI,3*Math.PI/2,2*Math.PI],
                        ticktext:['0', '\u03C0/2', '\u03C0', '3\u03C0/2', '2\u03C0'] },
                yaxis: {title: 'Orbital Radius (AU)', titlefont: {color: 'red'}, tickfont: {color: 'red'}},
                yaxis2: {title: 'Angular Separation (mas)', titlefont: {color: 'blue'}, tickfont: {color: 'blue'}, overlaying: 'y', side: 'right'},
                showlegend: false,
                margin: { t: 30, b:160}
             };\n

             Plotly.newPlot('plot2Div', data2, layout2);\n";


        echo "</script>\n";
        echo '<DIV style="clear: both;"></DIV>';

        echo "<p>Note: if no inclination available, orbit is assumed edge-on. If no eccentricity is available, orbit is assumed circular. For full documentation see <a href=docs/html/index.html#planetorbits-table>here</a>.</p>";
    }
    $resultp->close();
}
?>

</DIV>


<?php  
if ($resultc){

    echo '<div id="compDiv" style="width:800px; height:640px; margin:auto;"></div>';
    echo "\n\n";
    echo "<script>\n";
    echo "var xsize = 400, ysize = 260, x = new Array(xsize), y = new Array(ysize), z = new Array(ysize), i, j;\n";
    echo "x[0] = 100.5;\n";
    echo "for(var i = 1; i < xsize; i++) {x[i] = x[i-1]+1;}\n";
    echo "y[0] = 0.05;\n";
    echo "for(var i = 1; i < ysize; i++) {y[i] = y[i-1]+0.1;}\n";
    echo "for (var i = 0; i < ysize; i++) { z[i] = new Array(xsize).fill(0); }\n";


    while($rowc = $resultc->fetch_assoc()) {
        echo "z[".$rowc[jind]."][".$rowc[iind]."]=".$rowc[H].";";
    }
    echo "\n\n";

    echo "var box1 = {
        x: [100, 100, 500, 500],
        y: [0, 22.5, 22.5, 0],
        type: 'scatter'
    };\n";    

    echo "var data = {
		z: z,
		x: x,
		y: y,
        type: 'contour',
        colorscale: 'Hot',
        colorbar:{
            title: 'log(Frequency)',
        }
    };\n";

    echo "var layout = {
        title: 'Completeness for \u03B1 \u2208 [0.1, 0.5] arcsec, \u0394 mag < 22.5  = ".$row[completeness]."',
        xaxis: {title: 'Separation (mas)',range: [".$row[compMinWA].",".$row[compMaxWA]."]},
        yaxis: {title: 'Delta mag',range: [".$row[compMindMag].",".$row[compMaxdMag]."]},
    };\n";
    $resultc->close();

    
    echo "Plotly.newPlot('compDiv', [data,box1], layout);" ;
    echo "</script>\n";
    echo "<p>Completeness assumes cloud-free atmosphere at 575 nm. For full documentation see <a href=docs/html/index.html#completeness-table>here</a>.</p>";
}
?>

<?php 
$conn->close();
include "templates/footer.php"; 
?>




