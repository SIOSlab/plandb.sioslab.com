<?php include "templates/header.php"; ?>

<?php
if (empty($_GET["name"])){
    echo "No planet name provided.";
    include "templates/footer.php";
    exit;
} else{
    $name = $_GET["name"];
    $sql_id = "SELECT distinct pl_id, pl_name FROM Planets where pl_name ='".$name."'";
}



include("config.php");
$conn = new mysqli($servername, $username, $password, $dbname);
// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}
$result_id = $conn->query($sql_id);

if (!$result_id){
    include "templates/headerclose.php";
    echo "Query Error:\n".$conn->error;
    $conn->close();
    include "templates/footer.php";
    exit;
}
if ($result_id->num_rows == 0) {
    include "templates/headerclose.php";
    echo "Planet Not Found.";
    $result->close();
    $conn->close();
    include "templates/footer.php";
    exit;
}
if ($result_id->num_rows > 1) {
    include "templates/headerclose.php";
    echo "Multiple matches found.";
    $result->close();
    $conn->close();
    include "templates/footer.php";
    exit;
}
$row = $result_id->fetch_assoc();
$pl_id = $row[pl_id];

$sql = "SELECT Planets.pl_id, Planets.st_name,orbper,reflink,discmethod,orbsmax,orbeccen,orbincl,orbinclerr1,orbinclerr2,bmassj,bmassprov,radj,radj_fortney,radj_forecastermod,orbtper,orblper,eqt,insol,angsep,minangsep,maxangsep,
ra_str,dec_str,dist,plx,gaia_plx,gaia_dist,optmag,optband,gaia_gmag,teff,mass,pmra,pmdec,gaia_pmra,gaia_pmdec,radv,spstr,Stars.lum,metfe,age,bmvj,completeness,compMinWA,compMaxWA,compMindMag,
compMaxdMag,elat,elon,orbtper_next,orbtper_2026,calc_sma,Stars.st_id
FROM Planets LEFT JOIN Stars ON Planets.st_id = Stars.st_id LEFT JOIN OrbitFits ON OrbitFits.pl_id = Planets.pl_id WHERE Planets.pl_id='".$pl_id."' AND default_fit = 1";

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
$row = $result->fetch_assoc();

if($row[orbtper]) //Need if statement so graph displays correctly if using either time or mean anomoly
  $sql2 = "select * from Orbits where pl_id = '".$pl_id."' AND default_orb = 1";
else
  $sql2 = "select * from Orbits where pl_id = '".$pl_id."' AND default_orb = 1 ORDER BY M";

$resultp = $conn->query($sql2);

// $row = $result->fetch_assoc();
if ($row[completeness]){
    $sql3 = "SELECT * FROM Completeness WHERE pl_id='".$pl_id."'";
    $resultc = $conn->query($sql3);

}

// $sql4 = "select * from AltPlanetOrbits where Name = '".$name."'";
// $resultap = $conn->query($sql4);

$sql4 = "select Orbits.*, orbincl, OrbitFits.is_Icrit from Orbits LEFT JOIN OrbitFits
  ON Orbits.orbitfit_id = OrbitFits.orbitfit_id where Orbits.pl_id = '".$pl_id."'";
$resultap = $conn->query($sql4);


$sqlaliases = "select Alias from Aliases where st_id = '".$row[st_id]."'";
//(select st_id from Aliases where Alias = '".$row[pl_hostname]."')";
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
echo "<TR><TH style='width:".$wd."%'>Reference </TH><TD>".$row[reflink]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Discovered via</TH><TD>".$row[discmethod]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Period (days)</TH><TD>";
if($row[orbper])
    echo number_format((float)$row[orbper], 2, '.', '');
if ($row[orbperreflink])
    echo " (".$row[orbperreflink].")";
echo "</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Semi-major Axis (AU)</TH><TD>".
    number_format((float)$row[orbsmax], 2, '.', '');
if ($row[calc_sma] == 1)
    echo " (".'Calculated from stellar mass and orbital period.'.")";
echo"</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Eccentricity</TH><TD>".$row[orbeccen];
if ($row[orbeccenreflink])
    echo " (".$row[orbeccenreflink].")";
echo "</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Inclination (deg)</TH><TD>";
if ($row[orbincl] && !($row[orbincl] == 90 && $row[orbinclerr1] == 0 && $row[orbinclerr2]) == 0)
  echo $row[orbincl];
echo "</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>".$row[bmassprov]." (Jupiter Masses)</TH><TD>".$row[bmassj];
if ($row[orbsmaxreflink])
    echo " (".$row[orbsmaxreflink].")";
echo "</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Radius (Jupiter Radii)</TH><TD>".$row[radj];
if ($row[radreflink])
    echo " (".$row[radreflink].")";
echo "</TD></TR>\n";
if ($row[radreflink] == '<a refstr="CALCULATED VALUE" href="/docs/composite_calc.html" target=_blank>Calculated Value</a>' or $row[radreflink] == '<a refstr=CALCULATED_VALUE href=/docs/composite_calc.html target=_blank>Calculated Value</a>') {
    echo "<TR><TH style='width:".$wd."%'>Radius Based on Modified Forecaster (Jupiter Radii)</TH><TD>".$row[radj_forecastermod]." (<a href='docs/html/index.html#forecastermodref' target=_blank>See here</a>)</TD></TR>\n";
    echo "<TR><TH style='width:".$wd."%'>Radius Based on Fortney et al., 2007 (Jupiter Radii)</TH><TD>".$row[radj_fortney]." (<a href='docs/html/index.html#fortneyref' target=_blank>See here</a>)</TD></TR>\n";
}
echo "<TR><TH style='width:".$wd."%'>Periapsis Passage Time (JD)</TH><TD>".$row[orbtper]."</TD></TR>\n";
if($row[orbtper]){
  echo "<TR><TH style='width:".$wd."%'>Next Periapsis Passage Time (JD)</TH><TD>".$row[orbtper_next]."</TD></TR>\n";
  echo "<TR><TH style='width:".$wd."%'>First Periapsis Passage Time After 1/1/2026 (JD)</TH><TD>".$row[orbtper_2026]."</TD></TR>\n";
}
echo "<TR><TH style='width:".$wd."%'>Longitude of Periapsis (deg)</TH><TD>".$row[orblper]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Equilibrium Temperature (K)</TH><TD>".$row[eqt]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Insolation Flux (Earth fluxes)</TH><TD>".$row[insor]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Angular Separation @ sma (mas)</TH><TD>".
    number_format((float)$row[angsep], 2, '.', '')."</TD></TR>\n";
//echo "<TR><TH style='width:".$wd."%'>Minimum Angular Separation (mas)</TH><TD>".$row[pl_minangsep]."</TD></TR>\n";
//echo "<TR><TH style='width:".$wd."%'>Maximum Angular Separation (mas)</TH><TD>".$row[pl_maxangsep]."</TD></TR>\n";
echo "</TABLE>\n";


echo "<TABLE class='results'>\n";
echo "<TR><TH colspan='2'> Star Properties</TH></TR>\n";
echo "<TR><TH style='width:".$wd."%'>RA, DEC</TH><TD>".$row[ra_str].", ".$row[dec_str]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Ecliptic Lat, Lon</TH><TD>".$row[elat].", ".$row[elon]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Distance (GAIA Distance) (pc)</TH><TD>".$row[dist]." (".$row[gaia_dist].")</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Parallax (GAIA Parallax) (mas)</TH><TD>".$row[plx]." (".$row[gaia_plx].")</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Proper Motion RA/DEC (GAIA PM) (mas/yr)</TH><TD>".$row[pmra].", ".$row[pmdec]." (".$row[gaia_pmra].", ".$row[gaia_pmdec].")</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Radial Velocity (km/s)</TH><TD>".$row[radv]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>".$row[optband]. " band Magnitude</TH><TD>".$row[optmag]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>GAIA G band Magnitude</TH><TD>".$row[gaia_gmag]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Effective Temperature (K)</TH><TD>".$row[teff]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Mass (Solar Masses)</TH><TD>".$row[mass]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Spectral Type</TH><TD>".$row[spstr]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Luminosity  log(Solar Luminosities)</TH><TD>".$row[lum]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Metallicity (dex)</TH><TD>".$row[metfe]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Age (Gyr)</TH><TD>".$row[age]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>B-V (Johnson) (mag)</TH><TD>".$row[bmvj]."</TD></TR>\n";

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

        $clouds = array("000C","001C","003C","010C","030C","100C","300C","600C","min","max","med");
        $bands = array("575","660","730","760","825");
        foreach ($clouds as &$c) {
            foreach ($bands as &$b){
                echo "var d".$c.$b."NM = new Array(xsize), p".$c.$b."NM = new Array(xsize);\n";
            }
        }

        $i = 0;
        while($rowp = $resultp->fetch_assoc()) {
            if ($i == 0){
                // $havet = !(is_null($rowp[t]));
                $havet = !(is_null($rowp[t])) && !(is_null($row[orbtper]));
            }
            echo "x[".$i."]="; if ($havet){echo $rowp[t].";";} else{echo $rowp[M].";";}
            echo "r[".$i."]=".$rowp[r].";";
            echo "WA[".$i."]=".$rowp[WA].";\n";

            foreach ($clouds as &$c) {
                foreach ($bands as &$b){
                    echo "d".$c.$b."NM[".$i."]=";
                    $tmp = "dMag_".$c."_".$b."NM";
                    if ($rowp[$tmp]){ echo $rowp[$tmp]; } else{ echo "NaN";} echo";";
                    echo "p".$c.$b."NM[".$i."]=";
                    $tmp = "pPhi_".$c."_".$b."NM";
                    if ($rowp[$tmp]){ echo $rowp[$tmp]; } else{ echo "NaN";} echo";\n";
                }
            }
            $i++;
        }

        $cloudnames = array("MIN", "MAX", "No Cloud", "f1.0", "f0.01", "f0.03", "f0.1", "f0.3", "f3.0", "f6.0");
        $clouds =     array("min", "max", "000C",     "100C",  "001C",  "003C", "010C", "030C",  "300C", "600C");
        echo "var datan = [";
        for ($i = 0; $i < count($clouds); $i++) {
            echo "
                  {
                    x: x,
                    y: d".$clouds[$i]."575NM,
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
                    y: p".$clouds[$i]."575NM,
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
                    y: d".$clouds[$i]."660NM,
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
                    y: p".$clouds[$i]."660NM,
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
                    y: d".$clouds[$i]."730NM,
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
                    y: p".$clouds[$i]."730NM,
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
                    y: d".$clouds[$i]."760NM,
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
                    y: p".$clouds[$i]."760NM,
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
                    y: d".$clouds[$i]."825NM,
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
                    y: p".$clouds[$i]."825NM,
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

        echo "var updatemenus=[
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
    if ($resultap->num_rows > 0 and $resultap->num_rows % 4 == 0){
        echo '<div id="plot3Div" style="width:800px; height:640px; margin:auto;"></div>';
        echo "\n\n";
        echo "<script>\n";
        //TODO: Fix array sizes
        $rownum = ($resultap->num_rows) / 4;
        echo "var xsize = ".$rownum.", WA90 = new Array(xsize), dMag90 = new Array(xsize),
              WA60 = new Array(xsize), dMag60 = new Array(xsize),
              WA30 = new Array(xsize), dMag30 = new Array(xsize),
              WAcrit = new Array(xsize), dMagcrit = new Array(xsize), msizes = new Array(xsize), txtvals = new Array(xsize); \n";

        $maxi = $resultap->num_rows;
        $i = 0;
        $i90_ctr = 0;
        $i60_ctr = 0;
        $i30_ctr = 0;
        $icrit_ctr = 0;
        while($rowp = $resultap->fetch_assoc()) {
          // var_dump($rowp);
            // if ($i == 0){ $Icrit = round($rowp[Icrit] * 180.0/pi(), 2); }
            // echo "msizes[".$i."]=".(-19/$maxi*$i + 20).";";
            // echo "txtvals[".$i."]='t=".sprintf("%2.3g",$rowp[t])."';";
            // echo "WA90[".$i."]=".$rowp[WA_I90].";";
            // echo "WA60[".$i."]=".$rowp[WA_I60].";";
            // echo "WA30[".$i."]=".$rowp[WA_I30].";";
            // echo "WAcrit[".$i."]=".$rowp[WA_Icrit].";";
            // echo "dMag90[".$i."]="; if ($rowp[dMag_300C_575NM_I90]){ echo $rowp[dMag_300C_575NM_I90]; } else{ echo "NaN";} echo";";
            // echo "dMag60[".$i."]="; if ($rowp[dMag_300C_575NM_I60]){ echo $rowp[dMag_300C_575NM_I60]; } else{ echo "NaN";} echo";";
            // echo "dMag30[".$i."]="; if ($rowp[dMag_300C_575NM_I30]){ echo $rowp[dMag_300C_575NM_I30]; } else{ echo "NaN";} echo";";
            // echo "dMagcrit[".$i."]="; if ($rowp[dMag_300C_575NM_Icrit]){ echo $rowp[dMag_300C_575NM_Icrit]; } else{ echo "NaN";} echo";\n";
            // $i++;

            // foreach ($rowp as $row){
          if($rowp[is_Icrit] == 1){
            $Icrit = round($rowp[orbincl], 2);
            echo "msizes[".$icrit_ctr."]=".(-19/$maxi*$i + 20).";";
            echo "txtvals[".$icrit_ctr."]='t=".sprintf("%2.3g",$rowp[t])."';";
            echo "WAcrit[".$icrit_ctr."]=".$rowp[WA].";";
            echo "dMagcrit[".$icrit_ctr."]="; if ($rowp[dMag_300C_575NM]){ echo $rowp[dMag_300C_575NM]; } else{ echo "NaN";} echo";\n";
            $icrit_ctr++;
          }
          else{
            $i_str = $rowp[orbincl];
            $ctr = 0;
            if($i_str == 90){
              $ctr = $i90_ctr;
            }else if($i_str == 60){
              $ctr = $i60_ctr;
            }else if($i_str == 30){
              $ctr = $i30_ctr;
            }else{
              break;
            }

            echo "msizes[".$ctr."]=".(-19/$maxi*$i + 20).";";
            echo "txtvals[".$ctr."]='t=".sprintf("%2.3g",$rowp[t])."';";
            echo "WA".$i_str."[".$ctr."]=".$rowp[WA].";";
            echo "dMag".$i_str."[".$ctr."]="; if ($rowp[dMag_600C_575NM]){ echo $rowp[dMag_600C_575NM]; } else{ echo "NaN";} echo";";

            if($i_str == 90){
              $i90_ctr++;
            }else if($i_str == 60){
              $i60_ctr++;
            }else{
              $i30_ctr++;
            }
          }
        }
        // }

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
