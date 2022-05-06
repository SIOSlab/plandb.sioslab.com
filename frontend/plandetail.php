<?php include "templates/header.php"; ?>

<?php 

function debug_to_console($data) {
    $output = $data;
    if (is_array($output))
        $output = implode(',', $output);

    echo "<script>console.log('Debug Objects: " . $output . "' );</script>";
}

if (empty($_GET["name"])){
    echo "No planet name provided.";
    include "templates/footer.php"; 
    exit;
} else{
    $name = $_GET["name"];
}
//if (empty($_GET["scenario"])){
    //echo "No scenario name provided.";
    //include "templates/footer.php"; 
    //exit;
//} else{
    //$scenario = $_GET["scenario"];
//}

//if (empty($_GET["of_id"])){
    //echo "No orbitfit_id name provided.";
    //include "templates/footer.php"; 
    //exit;
//} else{
    //$orbitfit_id = $_GET["of_id"];
//}

$sql = "SELECT 
ST.st_name AS pl_hostname,
-- pl_reflink,
PL.pl_orbper AS pl_orbper,
PL.discoverymethod AS pl_discmethod,
PL.pl_orbsmax AS pl_orbsmax,
PL.pl_orbeccen AS pl_orbeccen,
PL.pl_orbincl AS pl_orbincl,
PL.pl_bmassj AS pl_bmassj,
PL.pl_bmassprov AS pl_bmassprov,
PL.pl_radj AS pl_radj,
-- pl_radreflink,
PL.pl_radj_fortney AS pl_radj_fortney,
PL.pl_radj_forecastermod AS pl_radj_forecastermod,
PL.pl_eqt AS pl_eqt,
PL.pl_insol AS pl_insol,
PL.pl_angsep AS pl_angsep,
-- S.minangsep AS pl_minangsep,
-- S.maxangsep AS pl_maxangsep, #Should this be OFT.pl_maxangsep instead
ST.rastr AS ra_str, #These are copied over to orbitfits should they be dropped
ST.decstr AS dec_str,
ST.sy_dist AS st_dist,
ST.sy_plx AS st_plx,
-- gaia_plx,
-- gaia_dist,
ST.sy_vmag AS st_optmag,
-- st_optband,
ST.sy_gaiamag AS gaia_gmag,
ST.teff AS st_teff,
ST.mass AS st_mass,
ST.sy_pmra AS st_pmra,
ST.sy_pmdec AS st_pmdec,
-- gaia_pmra,
-- gaia_pmdec,
PL.pl_rvamp AS pl_rvamp,
ST.st_radv AS st_radv,
ST.spectype AS st_spstr,
ST.lum AS st_lum,
ST.met AS st_metfe,
ST.age AS st_age,
-- st_bmvj
-- C.completeness AS completeness,
-- C.compMinWA AS compMinWA,
-- C.compMaxWA AS compMaxWA,
-- C.compMindMag AS compMindMag,
-- C.compMaxdMag AS compMaxdMag,
ST.elat AS st_elat,
ST.elon AS st_elon 
FROM Stars ST, Planets PL, OrbitFits OFT
-- Completeness C, Scenarios S
WHERE ST.st_id = PL.st_id
AND PL.pl_id= OFT.pl_id
AND OFT.default_fit = 1
-- AND PL.pl_id= C.pl_id
-- AND C.scenario_name= S.scenario_name 
AND PL.pl_name='".$name."'";
//AND S.scenario_name='".$scenario."'
//AND OFT.orbitfit_id='".$orbitfit_id."'";


include("config.php"); 
$conn = new mysqli($servername, $username, $password, $dbname);
// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
} 
$result = $conn->query($sql);

//$scenario_count_sql = "SELECT * FROM Scenarios";
//$scenario_table = mysqli_query($conn, $scenario_count_sql);
//$num_scenarios = mysqli_num_rows($scenario_table);
//$count2 = mysqli_num_rows($scenarios);
//$num_result = mysqli_num_rows($result);

//echo "Scenarios: $num_scenarios<br>Result: $num_result";
//while($row = mysqli_fetch_array($scenario_count))
    //{ print_r($row); }
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

 //$sql2 = "select * from PlanetOrbits where Name = '".$name."'";
$sql2 =  "SELECT O.*
FROM OrbitFits OFT, Orbits O
WHERE O.orbitfit_id = OFT.orbitfit_id
AND OFT.pl_name = '".$name."'
AND OFT.default_fit = TRUE";
//AND OFT.orbitfit_id='".$orbitfit_id."'";
$resultp = $conn->query($sql2);

$row = $result->fetch_assoc();
//print_r($row);
//if ($row['completeness']){
    // $sql3 = "SELECT * FROM Completeness WHERE Name='".$name."'";
    //$sql3 = "SELECT P.*
    //FROM Planets PL, PDFs P
    //WHERE P.pl_id = PL.pl_id
    //AND PL.pl_name='".$name."'
    //AND PL.scenario_name='Optimistic_NF_Imager_10000hr'";
//}
    
$sql3 = "SELECT C.*, PL.st_id
    FROM Completeness C, Planets PL
    WHERE C.pl_id = PL.pl_id
    AND PL.pl_name = '".$name."'";
$resultc = $conn->query($sql3);

//$sqlpdfs = "SELECT PDFs.jind, PDFs.iind, PDFs.H FROM PDFs WHERE PDFs.name = '".$name."'";
$sqlpdfs = "SELECT * FROM PDFs WHERE PDFs.name = '".$name."'";
$resultpdf = $conn->query($sqlpdfs);
//print_r($resultpdf);
$sqlcontr = "SELECT CC.* FROM ContrastCurves CC, Planets PL WHERE CC.st_id=PL.st_id AND PL.pl_name='".$name."'";
$resultcontr = $conn->query($sqlcontr);

$sql4 =  "SELECT O.*, PL.pl_orbincl
FROM Stars ST, Planets PL, OrbitFits OFT, Completeness C, Scenarios S, Orbits O
WHERE ST.st_id = PL.st_id
AND PL.pl_id= OFT.pl_id
AND O.orbitfit_id = OFT.orbitfit_id
AND PL.pl_id= C.pl_id
AND C.scenario_name= S.scenario_name 
AND PL.pl_name='".$name."'
AND S.scenario_name='Optimistic_NF_Imager_10000hr'";
//AND OFT.pl_orbincl = 90";
// Note that the scenario statement is not necessary, but reduces the number of points which makes the plot look better
$resultap = $conn->query($sql4);


//$sqlaliases = "select Alias from Aliases where SID = (select SID from Aliases where Alias = '".$row['pl_hostname']."')";
//$resultaliases = $conn->query($sqlaliases);

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
echo "<TR><TH style='width:".$wd."%'>Discovered via</TH><TD>".$row['pl_discmethod']."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Period (days)</TH><TD>".
    number_format((float)$row['pl_orbper'], 2, '.', '');
// if ($row[pl_reflink])
//     echo " (".$row[pl_reflink].")";
echo "</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Semi-major Axis (AU)</TH><TD>".
    number_format((float)$row['pl_orbsmax'], 2, '.', '');
// if ($row[pl_reflink])
//     echo " (".$row[pl_reflink].")";
echo"</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Eccentricity</TH><TD>".$row['pl_orbeccen'];
// if ($row[pl_reflink])
//     echo " (".$row[pl_reflink].")";
echo "</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Inclination (deg)</TH><TD>".$row['pl_orbincl']."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>".$row['pl_bmassprov']." (Jupiter Masses)</TH><TD>".$row['pl_bmassj'];
// if ($row[pl_reflink])
//     echo " (".$row[pl_reflink].")";
echo "</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Confirmed Radius (Jupiter Radii)</TH><TD>".$row['pl_radj'];
echo "<TR><TH style='width:".$wd."%'>Estimated Radius (Modified Forecaster)  (Jupiter Radii)</TH><TD>".number_format($row['pl_radj_forecastermod'], 4);
// if ($row[pl_radreflink])
//     echo " (".$row[pl_radreflink].")";
// echo "</TD></TR>\n";
// if ($row[pl_radreflink] == '<a refstr="CALCULATED VALUE" href="/docs/composite_calc.html" target=_blank>Calculated Value</a>' or $row[pl_radreflink] == '<a refstr=CALCULATED_VALUE href=/docs/composite_calc.html target=_blank>Calculated Value</a>') {
//     echo "<TR><TH style='width:".$wd."%'>Radius Based on Modified Forecaster (Jupiter Radii)</TH><TD>".$row[pl_radj_forecastermod]." (<a href='docs/html/index.html#forecastermodref' target=_blank>See here</a>)</TD></TR>\n";
//     echo "<TR><TH style='width:".$wd."%'>Radius Based on Fortney et al., 2007 (Jupiter Radii)</TH><TD>".$row[pl_radj_fortney]." (<a href='docs/html/index.html#fortneyref' target=_blank>See here</a>)</TD></TR>\n";
// }
echo "<TR><TH style='width:".$wd."%'>Periapsis Passage Time (JD)</TH><TD>".$row['pl_orbtper']."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Longitude of Periapsis (deg)</TH><TD>".$row['pl_orblper']."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Equilibrium Temperature (K)</TH><TD>".$row['pl_eqt']."</TD></TR>\n";
// echo "<TR><TH style='width:".$wd."%'>Insolation Flux (Earth fluxes)</TH><TD>".$row['pl_insor']."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Angular Separation @ sma (mas)</TH><TD>".
    number_format((float)$row['pl_angsep'], 2, '.', '')."</TD></TR>\n";
//echo "<TR><TH style='width:".$wd."%'>Minimum Angular Separation (mas)</TH><TD>".$row[pl_minangsep]."</TD></TR>\n";
//echo "<TR><TH style='width:".$wd."%'>Maximum Angular Separation (mas)</TH><TD>".$row[pl_maxangsep]."</TD></TR>\n";
echo "</TABLE>\n";


echo "<TABLE class='results'>\n";
echo "<TR><TH colspan='2'> Star Properties</TH></TR>\n";
echo "<TR><TH style='width:".$wd."%'>RA, DEC</TH><TD>".$row['ra_str'].", ".$row['dec_str']."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Ecliptic Lat, Lon</TH><TD>".$row['st_elat'].", ".$row['st_elon']."</TD></TR>\n";
// echo "<TR><TH style='width:".$wd."%'>Distance (GAIA Distance) (pc)</TH><TD>".$row[st_dist]." (".$row[gaia_dist].")</TD></TR>\n";
// echo "<TR><TH style='width:".$wd."%'>Parallax (GAIA Parallax) (mas)</TH><TD>".$row[st_plx]." (".$row[gaia_plx].")</TD></TR>\n";
// echo "<TR><TH style='width:".$wd."%'>Proper Motion RA/DEC (GAIA PM) (mas/yr)</TH><TD>".$row[st_pmra].", ".$row[st_pmdec]." (".$row[gaia_pmra].", ".$row[gaia_pmdec].")</TD></TR>\n"; Replaced by line below
echo "<TR><TH style='width:".$wd."%'>Proper Motion RA/DEC (mas/yr)</TH><TD>".$row['st_pmra'].", ".$row['st_pmdec']."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Radial Velocity (km/s)</TH><TD>".$row['st_radv']."</TD></TR>\n";
// echo "<TR><TH style='width:".$wd."%'>".$row[st_optband]. " band Magnitude</TH><TD>".$row[st_optmag]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>GAIA G band Magnitude</TH><TD>".$row['gaia_gmag']."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Effective Temperature (K)</TH><TD>".$row['st_teff']."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Mass (Solar Masses)</TH><TD>".$row['st_mass']."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Spectral Type</TH><TD>".$row['st_spstr']."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Luminosity  log(Solar Luminosities)</TH><TD>".$row['st_lum']."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Metallicity (dex)</TH><TD>".$row['st_metfe']."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Age (Gyr)</TH><TD>".$row['st_age']."</TD></TR>\n";
// echo "<TR><TH style='width:".$wd."%'>B-V (Johnson) (mag)</TH><TD>".$row[st_bmvj]."</TD></TR>\n";

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
                $havet = !(is_null($rowp['t']));
            }
            echo "x[".$i."]="; if ($havet){echo $rowp['t'].";";} else{echo $rowp['M'].";";}
            echo "r[".$i."]=".$rowp['r'].";";
            echo "WA[".$i."]=".$rowp['WA'].";\n";

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
    $of_ids = array_column($resultap->fetch_all(MYSQLI_ASSOC), 'orbitfit_id'); 
    $num_of_ids = count(array_unique($of_ids));
    $resultap = $conn->query($sql4);
    if ($resultap->num_rows > 0){
        echo '<div id="plot3Div" style="width:800px; height:640px; margin:auto;"></div>';
        echo "\n\n";
        echo "<script>\n";
        echo "var xsize = ".$resultap->num_rows.", WA90 = new Array(xsize), dMag90 = new Array(xsize),
              WA60 = new Array(xsize), dMag60 = new Array(xsize),
              WA30 = new Array(xsize), dMag30 = new Array(xsize),
              WAcrit = new Array(xsize), dMagcrit = new Array(xsize), msizes = new Array(xsize), txtvals = new Array(xsize); \n";
        if($num_of_ids == 8){
            echo "var WA90_alt = new Array(xsize), dMag90_alt = new Array(xsize),
                  WA60_alt = new Array(xsize), dMag60_alt = new Array(xsize),
                  WA30_alt = new Array(xsize), dMag30_alt = new Array(xsize),
                  WAcrit_alt = new Array(xsize), dMagcrit_alt = new Array(xsize); \n";
        };

        $maxi = $resultap->num_rows;
        $i = 0;
        $first_incl = true;
        $done_with_first = false;
        while($rowp = $resultap->fetch_assoc()) {
            // if ($i == 0){ $Icrit = round($rowp['Icrit'] * 180.0/pi(), 2); }
            if ($i == 0){ $Icrit = 0; $first_id = $rowp['orbitfit_id'];}
            if (($rowp['orbitfit_id'] != $first_id) && ($done_with_first==false)){
                $num_per_incl = $i;
                $done_with_first = true;
            };
            if ($done_with_first == false){
                echo "msizes[".$i."]=".($i/10+5).";";
            }
            else{
                echo "msizes[".$i."]=".(($i%$num_per_incl)/10+5).";";
            };
            echo "txtvals[".$i."]='t=".sprintf("%2.3g",$rowp['t'])."';";
            $incl = $rowp['pl_orbincl'];
            if ($incl < 15) {
                if($rowp['from_IPAC'] == 1){
                    echo "WAcrit[".$i."]=".$rowp['WA'].";";
                    echo "dMagcrit[".$i."]="; if ($rowp['dMag_300C_575NM']){ echo $rowp['dMag_300C_575NM']; } else{ echo "NaN";} echo";";
                }
                else{
                    echo "WAcrit_alt[".$i."]=".$rowp['WA'].";";
                    echo "dMagcrit_alt[".$i."]="; if ($rowp['dMag_300C_575NM']){ echo $rowp['dMag_300C_575NM']; } else{ echo "NaN";} echo";";
                };
            } elseif ($incl < 45){
                if($rowp['from_IPAC'] == 1){
                    echo "WA30[".$i."]=".$rowp['WA'].";";
                    echo "dMag30[".$i."]="; if ($rowp['dMag_300C_575NM']){ echo $rowp['dMag_300C_575NM']; } else{ echo "NaN";} echo";";
                }
                else{
                    echo "WA30_alt[".$i."]=".$rowp['WA'].";";
                    echo "dMag30_alt[".$i."]="; if ($rowp['dMag_300C_575NM']){ echo $rowp['dMag_300C_575NM']; } else{ echo "NaN";} echo";";
                };
            } elseif ($incl < 75){
                if($rowp['from_IPAC'] == 1){
                    echo "WA60[".$i."]=".$rowp['WA'].";";
                    echo "dMag60[".$i."]="; if ($rowp['dMag_300C_575NM']){ echo $rowp['dMag_300C_575NM']; } else{ echo "NaN";} echo";";
                }
                else{
                    echo "WA60_alt[".$i."]=".$rowp['WA'].";";
                    echo "dMag60_alt[".$i."]="; if ($rowp['dMag_300C_575NM']){ echo $rowp['dMag_300C_575NM']; } else{ echo "NaN";} echo";";
                };
            } else {
                if($rowp['from_IPAC'] == 1){
                    echo "WA90[".$i."]=".$rowp['WA'].";";
                    echo "dMag90[".$i."]="; if ($rowp['dMag_300C_575NM']){ echo $rowp['dMag_300C_575NM']; } else{ echo "NaN";} echo";";
                }
                else{
                    echo "WA90_alt[".$i."]=".$rowp['WA'].";";
                    echo "dMag90_alt[".$i."]="; if ($rowp['dMag_300C_575NM']){ echo $rowp['dMag_300C_575NM']; } else{ echo "NaN";} echo";";
                };
            };
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
              };\n
              var d2 = {\n
                x: WA60,
                y: dMag60,
                text: txtvals, 
                type: 'scatter',
                mode: 'lines+markers',
                marker: {size: msizes},
                name: 'I = 60\u00B0',
              };\n
              var d3 = {\n
                x: WA30,
                y: dMag30,
                text: txtvals, 
                type: 'scatter',
                mode: 'lines+markers',
                marker: {size: msizes},
                name: 'I = 30\u00B0',
              };\n
              var d4 = {\n
                x: WAcrit,
                y: dMagcrit,
                text: txtvals, 
                type: 'scatter',
                mode: 'lines+markers',
                marker: {size: msizes},
                name: 'I = ".$Icrit."\u00B0',
                };";
                if ($num_of_ids == 8){
                    echo " var d5 = {\n
                                x: WA90_alt,
                                y: dMag90_alt,
                                text: txtvals, 
                                type: 'scatter',
                                mode: 'lines+markers',
                                marker: {size: msizes},
                                name: 'I = 90\u00B0 (Non-Exoplanet Archive Fit)',
                              };\n
                              var d6 = {\n
                                x: WA60_alt,
                                y: dMag60_alt,
                                text: txtvals, 
                                type: 'scatter',
                                mode: 'lines+markers',
                                marker: {size: msizes},
                                name: 'I = 60\u00B0 (Non-Exoplanet Archive Fit)',
                              };\n
                              var d7 = {\n
                                x: WA30_alt,
                                y: dMag30_alt,
                                text: txtvals, 
                                type: 'scatter',
                                mode: 'lines+markers',
                                marker: {size: msizes},
                                name: 'I = 30\u00B0 (Non-Exoplanet Archive Fit)',
                              };\n
                              var d8 = {\n
                                x: WAcrit_alt,
                                y: dMagcrit_alt,
                                text: txtvals, 
                                type: 'scatter',
                                mode: 'lines+markers',
                                marker: {size: msizes},
                                name: 'I = ".$Icrit."\u00B0 (Non-Exoplanet Archive Fit)',
                                };\n
                                var data = [d1, d2, d3, d4, d5, d6, d7, d8];\n";
                }
                else{
                echo ";\n
                 var data = [d1,d2,d3,d4];\n";
                };

              echo "var dmaglim = {
                x: [150, 155.1337625 , 180.1553371 , 210.18122662, 250.21574597,
               300.25889517, 350.30204436, 395.34087864, 450],
                y: [22.01610885074595, 22.05977185, 22.30204688, 22.57879263, 22.98455007, 23.03667541,
               23.11031286, 23.11031286, 23.110312860818773],
                type: 'scatter',
                name: '\u0394 mag limit',
                line: {color: 'black'}
               };\n    


            var updatemenus=[
                {
                    buttons: [
                        {
                            args: ['yaxis', {title: '\u0394 mag', titlefont: {color: 'black'}, tickfont: {color: 'black'}}],
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
                margin: { t: 30, b:50, l:75, r:50},";
                if ($num_of_ids == 4) {
                    echo "colorway: ['#440154', '#31688e', '#35b779', '#fde725']";
                }
                elseif($num_of_ids == 8){
                    echo "colorway: ['#440154', '#46327e', '#365c8d', '#277f8e', '#1fa187', '#4ac16d', '#a0da39', '#fde725']";
                };
                echo "};\n

             Plotly.newPlot('plot3Div', data, layout);\n";

        echo "</script>\n";
        echo "<p>575 nm, f<sub>sed</sub> = 3.0 photometry for different orbit inclinations. If no eccentricity is available, orbit is assumed circular. The curves cover the full planet orbit, or 10 years (whichever is shorter). Time between points is 30 days. Time starts at 1/1/2026.  In cases where the argument of periastron is unknown, the planet is assumed to be at periastron at t = 0. Markers decrease in size with time. To select a single orbit, double click on it in the legend. To deselect an orbit, single click on it in the legend. To reset, double click on an unselected orbit in the legend. </p>";
    }
    $resultap->close();
}
?>


</DIV>
<?php
$scenarios = array("Conservative_NF_Imager_25hr", "Conservative_NF_Imager_100hr", "Conservative_NF_Imager_10000hr", "Optimistic_NF_Imager_25hr", "Optimistic_NF_Imager_100hr",
"Optimistic_NF_Imager_10000hr", "Conservative_Amici_Spec_100hr" , "Conservative_Amici_Spec_400hr", "Conservative_Amici_Spec_10000hr", "Optimistic_Amici_Spec_100hr",
"Optimistic_Amici_Spec_400hr", "Optimistic_Amici_Spec_10000hr", "Conservative_WF_Imager_25hr", "Conservative_WF_Imager_100hr", "Conservative_WF_Imager_10000hr",
"Optimistic_WF_Imager_25hr", "Optimistic_WF_Imager_100hr", "Optimistic_WF_Imager_10000hr");
$all_scenario_rmas = array();
$all_scenario_dMags = array();
foreach($scenarios as $scenario) {
    $sqlcontr = "SELECT CC.r_mas, CC.dMag FROM ContrastCurves CC, Planets PL WHERE CC.st_id=PL.st_id AND PL.pl_name='".$name."' AND CC.scenario_name ='".$scenario."'";
    $resultcontr = $conn->query($sqlcontr);
    $rowcount = mysqli_num_rows($resultcontr);
    //printf("Total rows for scenario '".$scenario."': %d\n\n", $rowcount);
    $scenario_rmas = array_column($resultcontr->fetch_all(MYSQLI_ASSOC), 'r_mas'); 
    $resultcontr = $conn->query($sqlcontr);
    $scenario_dMags = array_column($resultcontr->fetch_all(MYSQLI_ASSOC), 'dMag'); 
    //print_r($scenario_dMags);
    //while($row = $resultcontr->fetch_assoc()){
        //$scenario_rmas[] = $row[0];
        //$scenario_dMags[] = $row[1];
    //}
    //echo $rowcount;
    if (count($scenario_rmas) == 0){
        $scenario_angles_sql = "SELECT minangsep, maxangsep FROM Scenarios WHERE scenario_name='".$scenario."'";
        $scenario_angles = $conn->query($scenario_angles_sql);
        $scenario_minangsep = array_column($scenario_angles->fetch_all(MYSQLI_ASSOC), 'minangsep');
        //echo $scenario;
        //print_r($scenario_minangsep);
        //print_r("\n\n");
        $scenario_angles = $conn->query($scenario_angles_sql);
        $scenario_maxangsep = array_column($scenario_angles->fetch_all(MYSQLI_ASSOC), 'maxangsep');
        //print_r($scenario_maxangsep);
        $minangsep_f = (float) str_replace(" arcsec", "", $scenario_minangsep[0]);
        $maxangsep_f = (float) str_replace(" arcsec", "", $scenario_maxangsep[0]);
        //print_r($minangsep_f);
        $scenario_rmas = array(1000*$minangsep_f, 1000*$maxangsep_f);
        $scenario_dMags = array(13, 13);
    }
    $implode_rmas = implode(", ", $scenario_rmas);
    $str_list_rmas = "[".$implode_rmas."]";
    $implode_dMags = implode(", ", $scenario_dMags);
    $str_list_dMags = "[".$implode_dMags."]";
    array_push($all_scenario_rmas, $str_list_rmas);
    array_push($all_scenario_dMags, $str_list_dMags);
}
//print_r($all_scenario_rmas[0]);
//echo $all_scenario_rmas[0];
//echo $all_scenario_dMags[0];
?>

<style>
table.beta {
  border-collapse: collapse;
  border-spacing: 0;
  width: 50%;
  border: 1px solid #ddd;
  margin-left:auto;
  margin-right:auto;
}

.beta th, .beta td {
  text-align: left;
  padding: 4px;
}

.beta tr:nth-child(even) {
  background-color: #f2f2f2;
}
</style>
<table class="beta">
    <tr class="beta">
        <td>Observing scenario</td>
        <td>Completeness</td>
    </tr>
<?php
$i = 0;
while ($row = $resultc->fetch_assoc()){
    $class = ($i) ? "":"alt";
    echo "<tr class=\"".$class."\">";
    echo "<td class=beta>".str_replace("_", " ", $row['scenario_name'])."</td>";
    echo "<td class=beta>".$row['completeness']."</td>";
    echo "</tr>";
    $i = ($i==0) ? 1:0;
}
?>
</table>
<?php  
if ($resultc){
    echo '<div id="compDiv" style="width:1200px; height:640px; margin:auto;"></div>';
    echo "\n\n";
    echo "<script>\n";

    echo "var scenario0 = {
        x: ".$all_scenario_rmas[0].",
        y: ".$all_scenario_dMags[0].",
        type: 'scatter',
        name: '".str_replace("_", " ", $scenarios[0])."',
        hoverlabel: {namelength :-1}
        //line: { color: 'blue' }
    };\n";    
    echo "var scenario1 = {
        x: ".$all_scenario_rmas[1].",
        y: ".$all_scenario_dMags[1].",
        type: 'scatter',
        name: '".str_replace("_", " ", $scenarios[1])."',
        hoverlabel: {namelength :-1}
    };\n";    
    echo "var scenario2 = {
        x: ".$all_scenario_rmas[2].",
        y: ".$all_scenario_dMags[2].",
        type: 'scatter',
        name: '".str_replace("_", " ", $scenarios[2])."',
        hoverlabel: {namelength :-1}
    };\n";    
    echo "var scenario3 = {
        x: ".$all_scenario_rmas[3].",
        y: ".$all_scenario_dMags[3].",
        type: 'scatter',
        name: '".str_replace("_", " ", $scenarios[3])."',
        hoverlabel: {namelength :-1}
    };\n";    
    echo "var scenario4 = {
        x: ".$all_scenario_rmas[4].",
        y: ".$all_scenario_dMags[4].",
        type: 'scatter',
        name: '".str_replace("_", " ", $scenarios[4])."',
        hoverlabel: {namelength :-1}
    };\n";    
    echo "var scenario5 = {
        x: ".$all_scenario_rmas[5].",
        y: ".$all_scenario_dMags[5].",
        type: 'scatter',
        name: '".str_replace("_", " ", $scenarios[5])."',
        hoverlabel: {namelength :-1}
    };\n";    
    echo "var scenario6 = {
        x: ".$all_scenario_rmas[6].",
        y: ".$all_scenario_dMags[6].",
        type: 'scatter',
        name: '".str_replace("_", " ", $scenarios[6])."',
        hoverlabel: {namelength :-1}
    };\n";    
    echo "var scenario7 = {
        x: ".$all_scenario_rmas[7].",
        y: ".$all_scenario_dMags[7].",
        type: 'scatter',
        name: '".str_replace("_", " ", $scenarios[7])."',
        hoverlabel: {namelength :-1}
    };\n";    
    echo "var scenario8 = {
        x: ".$all_scenario_rmas[8].",
        y: ".$all_scenario_dMags[8].",
        type: 'scatter',
        name: '".str_replace("_", " ", $scenarios[8])."',
        hoverlabel: {namelength :-1}
    };\n";    
    echo "var scenario9 = {
        x: ".$all_scenario_rmas[9].",
        y: ".$all_scenario_dMags[9].",
        type: 'scatter',
        name: '".str_replace("_", " ", $scenarios[9])."',
        hoverlabel: {namelength :-1}
    };\n";    
    echo "var scenario10 = {
        x: ".$all_scenario_rmas[10].",
        y: ".$all_scenario_dMags[10].",
        type: 'scatter',
        name: '".str_replace("_", " ", $scenarios[10])."',
        hoverlabel: {namelength :-1}
    };\n";    
    echo "var scenario11 = {
        x: ".$all_scenario_rmas[11].",
        y: ".$all_scenario_dMags[11].",
        type: 'scatter',
        name: '".str_replace("_", " ", $scenarios[11])."',
        hoverlabel: {namelength :-1}
    };\n";    
    echo "var scenario12 = {
        x: ".$all_scenario_rmas[12].",
        y: ".$all_scenario_dMags[12].",
        type: 'scatter',
        name: '".str_replace("_", " ", $scenarios[12])."',
        hoverlabel: {namelength :-1}
    };\n";    
    echo "var scenario13 = {
        x: ".$all_scenario_rmas[13].",
        y: ".$all_scenario_dMags[13].",
        type: 'scatter',
        name: '".str_replace("_", " ", $scenarios[13])."',
        hoverlabel: {namelength :-1}
    };\n";    
    echo "var scenario14 = {
        x: ".$all_scenario_rmas[14].",
        y: ".$all_scenario_dMags[14].",
        type: 'scatter',
        name: '".str_replace("_", " ", $scenarios[14])."',
        hoverlabel: {namelength :-1}
    };\n";    
    echo "var scenario15 = {
        x: ".$all_scenario_rmas[15].",
        y: ".$all_scenario_dMags[15].",
        type: 'scatter',
        name: '".str_replace("_", " ", $scenarios[15])."',
        hoverlabel: {namelength :-1}
    };\n";    
    echo "var scenario16 = {
        x: ".$all_scenario_rmas[16].",
        y: ".$all_scenario_dMags[16].",
        type: 'scatter',
        name: '".str_replace("_", " ", $scenarios[16])."',
        hoverlabel: {namelength :-1}
    };\n";    
    echo "var scenario17 = {
        x: ".$all_scenario_rmas[17].",
        y: ".$all_scenario_dMags[17].",
        type: 'scatter',
        name: '".str_replace("_", " ", $scenarios[17])."',
        hoverlabel: {namelength :-1}
    };\n";    

    $alphasql = "SELECT DISTINCT alpha FROM PDFs WHERE name='".$name."'";
    $alphas = $conn->query($alphasql);
    $x_alpha = array_column($alphas->fetch_all(MYSQLI_ASSOC), 'alpha'); 
    $implode_x = implode(", ", $x_alpha);
    $str_list_x = "[".$implode_x."]";
    $x_0 = $x_alpha[0];
    $x_step = $x_alpha[1]-$x_alpha[0];

    $dMagsql = "SELECT DISTINCT dMag FROM PDFs WHERE name='".$name."'";
    $dMag = $conn->query($dMagsql);
    $y_dMag = array_column($dMag->fetch_all(MYSQLI_ASSOC), 'dMag'); 
    $implode_y = implode(", ", $y_dMag);
    $str_list_y = "[".$implode_y."]";
    $y_0 = $y_dMag[0];
    $y_step = $y_dMag[1]-$y_dMag[0];

    echo "var xsize = 200, ysize = 120, x = new Array(xsize), y = new Array(ysize), z = new Array(ysize), i, j;\n";
    echo "x[0] = 0;\n";
    echo "for(var i = 1; i < xsize; i++) {x[i] = x[i-1]+".$x_step.";}\n";
    echo "y[0] = 10;\n";
    echo "for(var i = 1; i < ysize; i++) {y[i] = y[i-1]+".$y_step.";}\n";
    echo "for (var i = 0; i < ysize; i++) { z[i] = new Array(xsize).fill(0); }\n";
    //echo $str_list_x;
    //echo $str_list_y;
    while($rowpdf = $resultpdf->fetch_assoc()) {
        //echo "z[".$rowpdf['jind']."][".$rowpdf['iind']."]=".$rowpdf['H'].";";
        echo "z[".$rowpdf['jind']."][".$rowpdf['iind']."]=".$rowpdf['H'].";";
    }
    
    echo "\n\n";
    echo "var data = {
		z: z,
		x: x,
		y: y,
        type: 'contour',
        colorscale: 'Greys',
        name: 'Normalized Frequency',
        hoverlabel: {namelength :-1},
        colorbar:{
            title: 'log(Frequency)',
            x: 0.9,
            y:0.5,
        }
    };\n";

    echo "var updatemenus=[
        {
            buttons: [
                {
                    args: ['yaxis', {title: '\u0394 mag', titlefont: {color: 'black'}, tickfont: {color: 'black'}}],
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
        }];";
    echo "
    var layout = {
    updatemenus: updatemenus,
    showlegend: true, 
    legend:{x: 1, y: 0.5}, 
    //legend:{'orientation': 'h'}, 
    xaxis: {title: 'Separation (mas)'},
    yaxis: {title: '\u0394 mag'},
    colorway: ['#440154', '#481668', '#482878', '#443983', '#3e4989', '#375a8c', '#31688e', '#2b758e', '#26828e', '#21918c', '#1f9e89', '#25ab82', '#35b779', '#4ec36b', '#6ece58', '#90d743', '#b5de2b', '#dae319', '#fde725']};";
    $resultc->close();

    
    echo "Plotly.newPlot('compDiv', [data,scenario0,scenario1,scenario2, scenario3, scenario4, scenario5, scenario6, scenario7, scenario8, scenario9, scenario10, scenario11, scenario12, scenario13, scenario14, scenario15, scenario16, scenario17], layout);" ;
    //echo "Plotly.newPlot('compDiv', [data,scenario0,scenario1], layout);" ;
    echo "</script>\n";
    echo "<p>To select a single contrast curve, double click on it in the legend. To deselect a contrast curve, single click on it in the legend. To reset, double click on an unselected contrast curve in the legend. Contrast curves are not calculated for scenarios and appear as flat lines in scenarios where none of the star's planets can reach the IWA.</p>";
    echo "<p>For full documentation see <a href=docs/html/index.html#completeness-table target=_blank>here</a>.</p>";
    //echo ($x_alpha[1] - $x_alpha[0]);
    //echo "\nUnique alphas: ".count($x_alpha);
    //echo "\nUnique dMags: ".count($y_dMag);
    ////echo "\nUnique dMags: ".count(y_dMag);
    //echo "\nPDF rows: ".mysqli_num_rows($resultpdf);
    //echo "\nx step: ".$x_step;
    //echo "\ny step: ".$y_step;
}
?>
    </div>
</div>
<?php 
$conn->close();
include "templates/footer.php"; 
?>




