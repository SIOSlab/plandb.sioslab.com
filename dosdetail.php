<?php include "templates/header.php"; ?>

<?php 
if (empty($_GET["name"])){
    echo "No target name provided.";
    include "templates/footer.php"; 
    exit;
} else{
    $name = $_GET["name"];
}


include("config.php"); 
$conn = new mysqli($servername, $username, $password, $dbname);
// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
} 

$sql0 = "select * from BlindTargs WHERE Name='".$name."'";
$result = $conn->query($sql0);
if (!$result){
    include "templates/headerclose.php"; 
    echo "Query Error:\n".$conn->error;
    $conn->close();
    include "templates/footer.php"; 
    exit;
}
if ($result->num_rows == 0) {
    include "templates/headerclose.php"; 
    echo "Target Not Found.";
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

$sql = "SELECT * FROM DoS WHERE Name='".$name."'";
$resultc = $conn->query($sql);

$row = $result->fetch_assoc();
$result->close();

$sqlaliases = "select Alias from Aliases where SID = (select SID from Aliases where Alias = '".$row[Name]."')";
$resultaliases = $conn->query($sqlaliases);

echo '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>';
include "templates/headerclose.php"; 
?>

<h2> Depth of Search Detail for 
<?php echo $name; ?>
</h2>

<div class="container">
<?php
$wd = '50';
echo " <div style='float: left; width: 90%; margin-bottom: 2em;'>\n";
echo "<TABLE class='results'>\n";
echo "<TR><TH colspan='2'> Target Properties</TH></TR>\n";
echo "<TR><TH style='width:".$wd."%'>RA, DEC (deg)</TH><TD>".$row[RA].", ".$row[DEC]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Distance (pc)</TH><TD>".$row[dist]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>V band Magnitude</TH><TD>".$row[Vmag]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Spectral Type</TH><TD>".$row[spec]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Luminosity  (Solar Luminosities)</TH><TD>".$row[L]."</TD></TR>\n";
echo "<TR><TH style='width:".$wd."%'>Calculated Stellar Radius (Solar radii) (mas) </TH><TD>".$row[radius]." (".number_format((float)$row[angrad], 4, '.', '').")</TD></TR>\n";

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


if ($resultc){

    echo '<div id="DoSDiv" style="width:800px; height:640px; margin:auto;"></div>';
    echo "\n\n";
    echo "<script>\n";
    echo "var ac = [0.10357597,  0.11098365,  0.11892112,  0.12742628,  0.13653971,
        0.14630494,  0.15676857,  0.16798055,  0.1799944 ,  0.19286747,
        0.20666122,  0.22144149,  0.23727883,  0.25424885,  0.27243255,
        0.29191673,  0.31279441,  0.33516525,  0.35913604,  0.3848212 ,
        0.41234334,  0.44183385,  0.4734335 ,  0.50729314,  0.54357439,
        0.58245046,  0.62410691,  0.6687426 ,  0.71657061,  0.76781924,
        0.82273314,  0.88157444,  0.94462403,  1.01218288,  1.0845735 ,
        1.16214144,  1.24525699,  1.33431691,  1.42974633,  1.53200079,
        1.64156842,  1.75897226,  1.88477273,  2.01957037,  2.16400863,
        2.31877703,  2.48461435,  2.66231224,  2.85271897,  3.05674344,
        3.27535961,  3.50961105,  3.760616  ,  4.02957264,  4.31776488,
        4.62656842,  4.95745738,  5.31201129,  5.69192265,  6.099005  ,
        6.5352016 ,  7.00259468,  7.50341538,  8.04005444,  8.61507355,
        9.23121762,  9.89142789, 10.59885594, 11.35687876, 12.16911483,
       13.03944147, 13.97201327, 14.97128195, 16.04201763, 17.18933159,
       18.41870064, 19.73599331, 21.14749784, 22.6599522 , 24.28057623,
       26.01710618, 27.87783154, 29.87163468, 32.00803324, 34.29722554,
       36.75013928, 39.37848371, 42.1948055 , 45.21254868, 48.44611875,
       51.9109515 , 55.62358669, 59.60174696, 63.86442249, 68.43196162,
       73.32616797, 78.57040456, 84.18970531, 90.21089454, 96.66271504];\n";
    echo "var Rc = [ 1.05476232,  1.17028477,  1.29845978,  1.44067311,  1.5984623 ,
        1.77353329,  1.96777887,  2.18329913,  2.42242416,  2.68773927,
        2.98211292,  3.30872774,  3.67111493,  4.07319244,  4.51930734,
        5.01428281,  5.56347028,  6.17280731,  6.84888176,  7.59900301,
        8.43128101,  9.35471396, 10.37928555, 11.51607298, 12.77736663,
       14.17680299, 15.72951212, 17.45228113, 19.36373579, 21.48454181];\n";
    echo "var xsize = 100, ysize = 30, z = new Array(ysize), i, j;\n";
    echo "for (i = 0; i < ysize; i++) { z[i] = new Array(xsize).fill(NaN); }\n";

    while($rowc = $resultc->fetch_assoc()) {
        echo "z[".$rowc[jind]."][".$rowc[iind]."]=".$rowc[vals].";";
    }
    echo "\n\n";

    echo "var IWA = {
        x: [".$row[dist]*0.150.",".$row[dist]*0.150."],
        y: [1.05476232,21.48454181],
        type: 'scatter',
        name: 'Projected IWA'
    };\n";    

    echo "var OWA = {
        x: [".$row[dist]*0.450.",".$row[dist]*0.450."],
        y: [1.05476232,21.48454181],
        type: 'scatter',
        name: 'Projected OWA'
    };\n";    



    echo "var data = {
		z: z,
		x: ac,
		y: Rc,
        type: 'contour',
        colorscale: 'Blackbody',
        name: 'Normalized DoS',
        colorbar:{
            title: 'log(DoS)',
        }
    };\n";

    echo "var layout = {
        xaxis: {title: 'Semi-major Axis (AU)', type:'log'},
        yaxis: {title: 'Planet Radius (R<sub>\u2295</sub>)',
                type:'log',
                range:[0.023154606797951807,1.3321260961390506]},
        showlegend: false
    };\n";
    $resultc->close();

    
    echo "Plotly.newPlot('DoSDiv', [IWA,OWA,data], layout);" ;
    echo "</script>\n";
}
?>


<?php 
$conn->close();
include "templates/footer.php"; 
?>



