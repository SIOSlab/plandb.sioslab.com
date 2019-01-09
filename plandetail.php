<?php include "templates/header.php"; ?>

<?php 
if (empty($_GET["name"])){
    echo "No planet name provided.";
    include "templates/footer.php"; 
    exit;
} else{
    $name = $_GET["name"];
}

$sql = "SELECT pl_hostname,pl_orbper,pl_orbperreflink,pl_discmethod,pl_orbsmax,pl_orbsmaxreflink,pl_orbeccenreflink,pl_orbeccen,pl_orbincl,pl_bmassj,pl_bmassprov,pl_bmassreflink,pl_radj,pl_radreflink,pl_radj_fortney,pl_radj_forecastermod,pl_orbtper,pl_orblper,pl_eqt,pl_insol,pl_angsep,pl_minangsep,pl_maxangsep,ra_str,dec_str,st_dist,st_plx,gaia_plx,gaia_dist,st_optmag,st_optband,gaia_gmag,st_teff,st_mass,st_pmra,st_pmdec,gaia_pmra,gaia_pmdec,st_radv,st_spstr,st_lum,st_metfe,st_age,st_bmvj,completeness,compMinWA,compMaxWA,compMindMag,compMaxdMag,st_elat,st_elon FROM KnownPlanets WHERE pl_name='".$name."'";

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

$sql4 = "select * from AltPlanetOrbits where Name = '".$name."'";
$resultap = $conn->query($sql4);


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
if ($row[pl_radreflink] == '<a refstr="CALCULATED VALUE" href="/docs/composite_calc.html" target=_blank>Calculated Value</a>' or $row[pl_radreflink] == '<a refstr=CALCULATED_VALUE href=/docs/composite_calc.html target=_blank>Calculated Value</a>') {
    echo "<TR><TH style='width:".$wd."%'>Radius Based on Modified Forecaster (Jupiter Radii)</TH><TD>".$row[pl_radj_forecastermod]." (<a href='docs/html/index.html#forecastermodref' target=_blank>See here</a>)</TD></TR>\n";
    echo "<TR><TH style='width:".$wd."%'>Radius Based on Fortney et al., 2007 (Jupiter Radii)</TH><TD>".$row[pl_radj_fortney]." (<a href='docs/html/index.html#fortneyref' target=_blank>See here</a>)</TD></TR>\n";
}
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
echo "<TR><TH style='width:".$wd."%'>Ecliptic Lat, Lon</TH><TD>".$row[st_elat].", ".$row[st_elon]."</TD></TR>\n";
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
        echo "var xsize = 100, x = new Array(xsize), r = new Array(xsize), WA = new Array(xsize); \n";
        
        $clouds = array("000","001","003","010","030","100","300","600");
        $bands = array("575","660","730","760","825");
        foreach ($clouds as &$c) {
            foreach ($bands as &$b){
                echo "var d".$c."C".$b."NM = new Array(xsize), p".$c."C".$b."NM = new Array(xsize);\n";
            }
        }

        $i = 0;
        while($rowp = $resultp->fetch_assoc()) {
            if ($i == 0){ 
                $havet = !(is_null($rowp[t]));
            }
            echo "x[".$i."]="; if ($havet){echo $rowp[t].";";} else{echo $rowp[M].";";}
            echo "r[".$i."]=".$rowp[r].";";
            echo "WA[".$i."]=".$rowp[WA].";\n";

            foreach ($clouds as &$c) {
                foreach ($bands as &$b){
                    echo "d".$c."C".$b."NM[".$i."]=";
                    $tmp = "dMag_".$c."C_".$b."NM"; 
                    if ($rowp[$tmp]){ echo $rowp[$tmp]; } else{ echo "NaN";} echo";";
                    echo "p".$c."C".$b."NM[".$i."]=";
                    $tmp = "pPhi_".$c."C_".$b."NM"; 
                    if ($rowp[$tmp]){ echo $rowp[$tmp]; } else{ echo "NaN";} echo";\n";
                }
            }
            $i++;
        }

        $cloudnames = array("No Cloud", "f1.0", "f0.01", "f0.03", "f0.1", "f0.3", "f3.0", "f6.0");
        $clouds =     array("000",      "100",  "001",   "003",   "010",   "030",  "300",  "600");
        echo "var datan = [";
        for ($i = 0; $i < count($clouds); $i++) {
            echo "
                  { 
                    x: x,
                    y: d".$clouds[$i]."C575NM,
                    type: 'scatter',";
                    if ($i == 1){
                        echo "fill: 'tonexty',";
                    }
            echo "
                    name: '".$cloudnames[$i]."',
                    line: { color: 'red' },
                    visible: true
                   },";
        }
        for ($i = 0; $i < count($clouds); $i++) {
            echo "
                  { 
                    x: x,
                    y: p".$clouds[$i]."C575NM,
                    type: 'scatter',";
                    if ($i == 1){
                        echo "fill: 'tonexty',";
                    }
            echo "
                    name: '".$cloudnames[$i]."',
                    line: { color: 'blue' },
                    yaxis: 'y2',
                    visible: true
                   },";
        }

        for ($i = 0; $i < count($clouds); $i++) {
            echo "
                  { 
                    x: x,
                    y: d".$clouds[$i]."C660NM,
                    type: 'scatter',";
                    if ($i == 1){
                        echo "fill: 'tonexty',";
                    }
            echo "
                    name: '".$cloudnames[$i]."',
                    line: { color: 'red' },
                    visible: false
                   },";
        }
        for ($i = 0; $i < count($clouds); $i++) {
            echo "
                  { 
                    x: x,
                    y: p".$clouds[$i]."C660NM,
                    type: 'scatter',";
                    if ($i == 1){
                        echo "fill: 'tonexty',";
                    }
            echo "
                    name: '".$cloudnames[$i]."',
                    line: { color: 'blue' },
                    yaxis: 'y2',
                    visible: false
                   },";
        }
        
        for ($i = 0; $i < count($clouds); $i++) {
            echo "
                  { 
                    x: x,
                    y: d".$clouds[$i]."C730NM,
                    type: 'scatter',";
                    if ($i == 1){
                        echo "fill: 'tonexty',";
                    }
            echo "
                    name: '".$cloudnames[$i]."',
                    line: { color: 'red' },
                    visible: false
                   },";
        }
        for ($i = 0; $i < count($clouds); $i++) {
            echo "
                  { 
                    x: x,
                    y: p".$clouds[$i]."C730NM,
                    type: 'scatter',";
                    if ($i == 1){
                        echo "fill: 'tonexty',";
                    }
            echo "
                    name: '".$cloudnames[$i]."',
                    line: { color: 'blue' },
                    yaxis: 'y2',
                    visible: false
                   },";
        }

        for ($i = 0; $i < count($clouds); $i++) {
            echo "
                  { 
                    x: x,
                    y: d".$clouds[$i]."C760NM,
                    type: 'scatter',";
                    if ($i == 1){
                        echo "fill: 'tonexty',";
                    }
            echo "
                    name: '".$cloudnames[$i]."',
                    line: { color: 'red' },
                    visible: false
                   },";
        }
        for ($i = 0; $i < count($clouds); $i++) {
            echo "
                  { 
                    x: x,
                    y: p".$clouds[$i]."C760NM,
                    type: 'scatter',";
                    if ($i == 1){
                        echo "fill: 'tonexty',";
                    }
            echo "
                    name: '".$cloudnames[$i]."',
                    line: { color: 'blue' },
                    yaxis: 'y2',
                    visible: false
                   },";
        }

        for ($i = 0; $i < count($clouds); $i++) {
            echo "
                  { 
                    x: x,
                    y: d".$clouds[$i]."C825NM,
                    type: 'scatter',";
                    if ($i == 1){
                        echo "fill: 'tonexty',";
                    }
            echo "
                    name: '".$cloudnames[$i]."',
                    line: { color: 'red' },
                    visible: false
                   },";
        }
        for ($i = 0; $i < count($clouds); $i++) {
            echo "
                  { 
                    x: x,
                    y: p".$clouds[$i]."C825NM,
                    type: 'scatter',";
                    if ($i == 1){
                        echo "fill: 'tonexty',";
                    }
            echo "
                    name: '".$cloudnames[$i]."',
                    line: { color: 'blue' },
                    yaxis: 'y2',
                    visible: false
                   }";
             if ($i < count($clouds)-1){
                 echo ",\n";
             }
        }



        echo "];\n\n\n";

        echo "var td1 = {\n
                x: x,
                y: d000C575NM,
                type: 'scatter',
                name: 'No Cloud',
                line: { color: 'red' }
              };\n
              var td2 = {\n
                x: x,
                y: d100C575NM,
                fill: 'tonexty',
                type: 'scatter',
                name: 'f1.0',
                line: { color: 'red' }
              };\n
              var td3 = {\n
                x: x,
                y: d001C575NM,
                type: 'scatter',
                name: 'f0.01',
                line: { color: 'red' }
              };\n
              var td4 = {\n
                x: x,
                y: d003C575NM,
                type: 'scatter',
                name: 'f0.03',
                line: { color: 'red' }
              };\n
              var td5 = {\n
                x: x,
                y: d010C575NM,
                type: 'scatter',
                name: 'f0.1',
                line: { color: 'red' }
              };\n
              var td6 = {\n
                x: x,
                y: d030C575NM,
                type: 'scatter',
                name: 'f0.3',
                line: { color: 'red' }
              };\n
              var td7 = {\n
                x: x,
                y: d300C575NM,
                type: 'scatter',
                name: 'f3.0',
                line: { color: 'red' }
              };\n
              var td8 = {\n
                x: x,
                y: d600C575NM,
                type: 'scatter',
                name: 'f6.0',
                line: { color: 'red' }
              };\n
              var tp1 = {\n
                x: x,
                y: p000C575NM,
                type: 'scatter',
                name: 'No Cloud',
                line: { color: 'blue' },
                yaxis: 'y2'
              };\n
              var tp2 = {\n
                x: x,
                y: p100C575NM,
                fill: 'tonexty',
                type: 'scatter',
                name: 'f1.0',
                line: { color: 'blue' },
                yaxis: 'y2'
              };\n
              var tp3 = {\n
                x: x,
                y: p001C575NM,
                type: 'scatter',
                name: 'f0.01',
                line: { color: 'blue' },
                yaxis: 'y2'
              };\n
              var tp4 = {\n
                x: x,
                y: p003C575NM,
                type: 'scatter',
                name: 'f0.03',
                line: { color: 'blue' },
                yaxis: 'y2'
              };\n
              var tp5 = {\n
                x: x,
                y: p010C575NM,
                type: 'scatter',
                name: 'f0.1',
                line: { color: 'blue' },
                yaxis: 'y2'
              };\n
              var tp6 = {\n
                x: x,
                y: p030C575NM,
                type: 'scatter',
                name: 'f0.3',
                line: { color: 'blue' },
                yaxis: 'y2'
              };\n
              var tp7 = {\n
                x: x,
                y: p300C575NM,
                type: 'scatter',
                name: 'f3.0',
                line: { color: 'blue' },
                yaxis: 'y2'
              };\n
              var tp8 = {\n
                x: x,
                y: p600C575NM,
                type: 'scatter',
                name: 'f6.0',
                line: { color: 'blue' },
                yaxis: 'y2'
              };\n


             var data = [td1,td2,td3,td4,td5,td6,td7,td8,tp1,tp2,tp3,tp4,tp5,tp6,tp7,tp8];\n
             
             var updatemenus=[
                {
                    buttons: [
                        {
                            args: ['yaxis', {title: '\u0394 mag', titlefont: {color: 'red'}, tickfont: {color: 'red'}}],
                            label: '\u0394 mag Axis Normal',
                            method: 'relayout'
                        },
                        {
                            args: ['yaxis', {title: '\u0394 mag', titlefont: {color: 'red'}, tickfont: {color: 'red'},autorange:'reversed'}],
                            label:'\u0394 mag Axis Reversed',
                            method:'relayout'
                        }
                    ],
                    direction: 'down',
                    pad: {'r': 10, 't': 10},
                    showactive: true,
                    type: 'dropdown',
                    x: 0.1,
                    xanchor: 'left',
                    y: 1.1,
                    yanchor: 'top'
                },
                {
                    buttons: [
                        {
                            args: ['visible', [";
                            for ($i = 0; $i < count($clouds)*2; $i++) { echo "true, ";}
                            echo "false";
                            for ($i = 0; $i < count($clouds)*2*4-1; $i++) { echo ", false";}
                            echo "]],
                            label: '575 nm',
                            method: 'restyle'
                        },
                        {
                            args: ['visible', [";
                            for ($i = 0; $i < count($clouds)*2; $i++) { echo "false, ";}
                            for ($i = 0; $i < count($clouds)*2; $i++) { echo "true, ";}
                            echo "false";
                            for ($i = 0; $i < count($clouds)*2*3-1; $i++) { echo ", false";}
                            echo "]],
                            label: '660 nm',
                            method: 'restyle'
                        },
                        {
                            args: ['visible', [";
                            for ($i = 0; $i < count($clouds)*2*2; $i++) { echo "false, ";}
                            for ($i = 0; $i < count($clouds)*2; $i++) { echo "true, ";}
                            echo "false";
                            for ($i = 0; $i < count($clouds)*2*2-1; $i++) { echo ", false";}
                            echo "]],
                            label: '730 nm',
                            method: 'restyle'
                        },
                        {
                            args: ['visible', [";
                            for ($i = 0; $i < count($clouds)*2*3; $i++) { echo "false, ";}
                            for ($i = 0; $i < count($clouds)*2; $i++) { echo "true, ";}
                            echo "false";
                            for ($i = 0; $i < count($clouds)*2-1; $i++) { echo ", false";}
                            echo "]],
                            label: '760 nm',
                            method: 'restyle'
                        },

                        {
                            args: ['visible', [";
                            for ($i = 0; $i < count($clouds)*2*4; $i++) { echo "false, ";}
                            echo "true";
                            for ($i = 0; $i < count($clouds)*2-1; $i++) { echo ", true";}
                            echo "]],
                            label: '825 nm',
                            method: 'restyle'
                        }

                    ],
                    direction: 'down',
                    pad: {'r': 10, 't': 10},
                    showactive: true,
                    type: 'dropdown',
                    x: 0.9,
                    xanchor: 'right',
                    y: 1.1,
                    yanchor: 'top'
                }
                
             ]


             var layout = {\n
                updatemenus: updatemenus,
                xaxis: {title:";
                if ($havet){echo "'Time After 1/1/2026 (days)'},";}
                else{echo "'Mean Anomaly (rad)',
                        tickvals:[0,Math.PI/2,Math.PI,3*Math.PI/2,2*Math.PI],
                        ticktext:['0', '\u03C0/2', '\u03C0', '3\u03C0/2', '2\u03C0'] },";}
                echo "yaxis2: {title: 'p * \u03A6 (\u03B2)', titlefont: {color: 'blue'}, tickfont: {color: 'blue'}, overlaying: 'y', side: 'right'},
                yaxis:{title: '\u0394 mag', titlefont: {color: 'red'}, tickfont: {color: 'red'}},
                margin: { t: 30, b:50, l:50, r:75},
                showlegend: false,
             };\n

             Plotly.newPlot('plot1Div', datan, layout);\n";

        echo "var trace5 = {\n
                x: x,
                y: r,
                type: 'scatter',
                name: 'Orbital Radius (AU)',
                line: { color: 'red' }
              };\n
              var trace6 = {\n
                x: x,
                y: WA,
                type: 'scatter',
                name: 'Angular Separation (mas)',
                line: { color: 'blue' },
                yaxis: 'y2'
              };\n

             var data2 = [trace5, trace6];\n

             var layout2 = {\n
                xaxis: {title:";
                if ($havet){echo "'Time After 1/1/2026 (days)'},";}
                else{echo "'Mean Anomaly (rad)',
                        tickvals:[0,Math.PI/2,Math.PI,3*Math.PI/2,2*Math.PI],
                        ticktext:['0', '\u03C0/2', '\u03C0', '3\u03C0/2', '2\u03C0'] },";}
                echo "yaxis: {title: 'Orbital Radius (AU)', titlefont: {color: 'red'}, tickfont: {color: 'red'}},
                yaxis2: {title: 'Angular Separation (mas)', titlefont: {color: 'blue'}, tickfont: {color: 'blue'}, overlaying: 'y', side: 'right'},
                showlegend: false,
                margin: { t: 30, b:50, l:75, r:50}
             };\n

             Plotly.newPlot('plot2Div', data2, layout2);\n";


        echo "</script>\n";
        echo '<DIV style="clear: both;"></DIV>';

        echo "<p>If no inclination available, orbit is assumed edge-on. If no eccentricity is available, orbit is assumed circular. For full documentation see <a href=docs/html/index.html#planetorbits-table target=_blank>here</a>.</p>";
    }
    $resultp->close();
}

if ($resultap){
    if ($resultap->num_rows > 0){
        echo '<div id="plot3Div" style="width:800px; height:640px; margin:auto;"></div>';
        echo "\n\n";
        echo "<script>\n";
        echo "var xsize = ".$resultap->num_rows.", WA90 = new Array(xsize), dMag90 = new Array(xsize),
              WA60 = new Array(xsize), dMag60 = new Array(xsize),
              WA30 = new Array(xsize), dMag30 = new Array(xsize),
              WAcrit = new Array(xsize), dMagcrit = new Array(xsize), msizes = new Array(xsize), txtvals = new Array(xsize); \n";

        $maxi = $resultap->num_rows;
        $i = 0;
        while($rowp = $resultap->fetch_assoc()) {
            if ($i == 0){ $Icrit = round($rowp[Icrit] * 180.0/pi(), 2); }
            echo "msizes[".$i."]=".(-19/$maxi*$i + 20).";";
            echo "txtvals[".$i."]='t=".sprintf("%2.3g",$rowp[t])."';";
            echo "WA90[".$i."]=".$rowp[WA_I90].";";
            echo "WA60[".$i."]=".$rowp[WA_I60].";";
            echo "WA30[".$i."]=".$rowp[WA_I30].";";
            echo "WAcrit[".$i."]=".$rowp[WA_Icrit].";";
            echo "dMag90[".$i."]="; if ($rowp[dMag_300C_575NM_I90]){ echo $rowp[dMag_300C_575NM_I90]; } else{ echo "NaN";} echo";";
            echo "dMag60[".$i."]="; if ($rowp[dMag_300C_575NM_I60]){ echo $rowp[dMag_300C_575NM_I60]; } else{ echo "NaN";} echo";";
            echo "dMag30[".$i."]="; if ($rowp[dMag_300C_575NM_I30]){ echo $rowp[dMag_300C_575NM_I30]; } else{ echo "NaN";} echo";";
            echo "dMagcrit[".$i."]="; if ($rowp[dMag_300C_575NM_Icrit]){ echo $rowp[dMag_300C_575NM_Icrit]; } else{ echo "NaN";} echo";\n";
            $i++;
        }

        echo "var d1 = {\n
                x: WA90,
                y: dMag90,
                text: txtvals, 
                type: 'scatter',
                mode: 'lines+markers',
                marker: {size: msizes},
                name: 'I = 90\u00B0',
                line: { color: 'red' }
              };\n
              var d2 = {\n
                x: WA60,
                y: dMag60,
                text: txtvals, 
                type: 'scatter',
                mode: 'lines+markers',
                marker: {size: msizes},
                name: 'I = 60\u00B0',
                line: { color: 'blue' }
              };\n
              var d3 = {\n
                x: WA30,
                y: dMag30,
                text: txtvals, 
                type: 'scatter',
                mode: 'lines+markers',
                marker: {size: msizes},
                name: 'I = 30\u00B0',
                line: { color: 'green' }
              };\n
              var d4 = {\n
                x: WAcrit,
                y: dMagcrit,
                text: txtvals, 
                type: 'scatter',
                mode: 'lines+markers',
                marker: {size: msizes},
                name: 'I = ".$Icrit."\u00B0',
                line: { color: 'orange' }
              };\n

              var dmaglim = {
                x: [150, 155.1337625 , 180.1553371 , 210.18122662, 250.21574597,
               300.25889517, 350.30204436, 395.34087864, 450],
                y: [22.01610885074595, 22.05977185, 22.30204688, 22.57879263, 22.98455007, 23.03667541,
               23.11031286, 23.11031286, 23.110312860818773],
                type: 'scatter',
                name: '\u0394 mag limit',
                line: {color: 'black'}
               };\n    

             var data = [d1,d2,d3,d4];\n

            var updatemenus=[
                {
                    buttons: [
                        {
                            args: ['yaxis', {title: '\u0394 mag', titlefont: {color: 'red'}, tickfont: {color: 'red'}}],
                            label: '\u0394 mag Axis Normal',
                            method: 'relayout'
                        },
                        {
                            args: ['yaxis', {title: '\u0394 mag', titlefont: {color: 'red'}, tickfont: {color: 'red'},autorange:'reversed'}],
                            label:'\u0394 mag Axis Reversed',
                            method:'relayout'
                        }
                    ],
                    direction: 'down',
                    pad: {'r': 10, 't': 10},
                    showactive: true,
                    type: 'dropdown',
                    x: 0.1,
                    xanchor: 'left',
                    y: 1.1,
                    yanchor: 'top'
                }];

             var layout = {\n
                updatemenus: updatemenus,
                xaxis: {title: 'Angular Separation (mas)'},
                yaxis: {title: '\u0394 mag'},
                showlegend: true,
                margin: { t: 30, b:50, l:75, r:50}
             };\n

             Plotly.newPlot('plot3Div', data, layout);\n";

        echo "</script>\n";
        echo "<p>575 nm, f<sub>sed</sub> = 3.0 photometry for different orbit inclinations. If no eccentricity is available, orbit is assumed circular. The curves cover the full planet orbit, or 10 years (whichever is shorter). Time between points is 30 days. Time starts at 1/1/2026.  In cases where the argument of periastron is unknown, the planet is assumed to be at periastron at t = 0. Markers decrease in size with time.</p>";
    }
    $resultap->close();
}
?>


</DIV>


<?php  
if ($resultc){

    echo '<div id="compDiv" style="width:800px; height:640px; margin:auto;"></div>';
    echo "\n\n";
    echo "<script>\n";
    echo "var xsize = 300, ysize = 260, x = new Array(xsize), y = new Array(ysize), z = new Array(ysize), i, j;\n";
    echo "x[0] = 150.5;\n";
    echo "for(var i = 1; i < xsize; i++) {x[i] = x[i-1]+1;}\n";
    echo "y[0] = 0.05;\n";
    echo "for(var i = 1; i < ysize; i++) {y[i] = y[i-1]+0.1;}\n";
    echo "for (var i = 0; i < ysize; i++) { z[i] = new Array(xsize).fill(0); }\n";


    while($rowc = $resultc->fetch_assoc()) {
        echo "z[".$rowc[jind]."][".$rowc[iind]."]=".$rowc[H].";";
    }
    echo "\n\n";

    echo "var box1 = {
        x: [150, 155.1337625 , 180.1553371 , 210.18122662, 250.21574597,
       300.25889517, 350.30204436, 395.34087864, 450],
        y: [22.01610885074595, 22.05977185, 22.30204688, 22.57879263, 22.98455007, 23.03667541,
       23.11031286, 23.11031286, 23.110312860818773],
        type: 'scatter',
        name: '\u0394 mag limit',
        line: { color: 'blue' }
    };\n";    

    echo "var data = {
		z: z,
		x: x,
		y: y,
        type: 'contour',
        colorscale: 'Hot',
        name: 'Normalized Frequency',
        colorbar:{
            title: 'log(Frequency)',
        }
    };\n";




    echo "
            var updatemenus=[
                {
                    buttons: [
                        {
                            args: ['yaxis', {title: '\u0394 mag',range: [".$row[compMindMag].",".$row[compMaxdMag]."]}],
                            label: '\u0394 mag Axis Normal',
                            method: 'relayout'
                        },
                        {
                            args: ['yaxis', {title: '\u0394 mag',range: [".$row[compMaxdMag].",".$row[compMindMag]."]}],
                            label:'\u0394 mag Axis Reversed',
                            method:'relayout'
                        }
                    ],
                    direction: 'down',
                    pad: {'r': 10, 't': 10},
                    showactive: true,
                    type: 'dropdown',
                    x: 0.1,
                    xanchor: 'left',
                    y: 1.1,
                    yanchor: 'top'
                }];

    var layout = {
        updatemenus: updatemenus,
        title: 'Completeness at 575 nm for \u03B1 \u2208 [0.15, 0.45] arcsec = ".$row[completeness]."',
        xaxis: {title: 'Separation (mas)',range: [".$row[compMinWA].",".$row[compMaxWA]."]},
        yaxis: {title: '\u0394 mag',range: [".$row[compMindMag].",".$row[compMaxdMag]."]},
    };\n";
    $resultc->close();

    
    echo "Plotly.newPlot('compDiv', [data,box1], layout);" ;
    echo "</script>\n";
    echo "<p>For full documentation see <a href=docs/html/index.html#completeness-table target=_blank>here</a>.</p>";
}
?>

<?php 
$conn->close();
include "templates/footer.php"; 
?>




