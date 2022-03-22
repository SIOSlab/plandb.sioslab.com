<?php 
include "templates/header.php"; 
include "templates/headerclose.php"; 
?>

<h2> General Query </h2>

<p>
This interface provides direct querying to the full database. Queries including selection of pl_name will automatically create links to known planet detail pages. See the <a href="docs/plandbschema/index.html" target=_blank>Schema</a> for all available tables and columns.

<form action="index.php" method="POST">
        <textarea name="querytext" rows="4" cols="100">
<?php 
if (!empty($_GET["querytext"])){
    $sql =$_GET["querytext"];}
elseif (!empty($_POST["querytext"])){
    $sql = $_POST["querytext"]; }
else {
    $sql = "SELECT PL.pl_name AS pl_name, 
    OFT.pl_angsep AS pl_angsep,
    C.completeness AS completeness,
    S.minangsep AS scenario_IWA,
    S.maxangsep AS scenario_OWA,
    PL.pl_radj_forecastermod AS pl_radj_forecastermod,
    OFT.pl_bmassj AS pl_bmassj,
    OFT.pl_orbsmax AS pl_orbsmax,
    S.scenario_name AS scenario_name,
    OFT.orbitfit_id AS orbitfit_id
    FROM Planets PL, OrbitFits OFT, Completeness C, Scenarios S
    WHERE C.completeness > 0 
    AND PL.pl_id= OFT.pl_id
    AND PL.pl_id= C.pl_id
    AND C.scenario_name= S.scenario_name 
    AND OFT.default_fit = 1
    AND C.scenario_name = 'Optimistic_NF_Imager_10000hr'
    ORDER BY C.completeness DESC";}
    
echo "$sql";
?>
</textarea><br>
<!--<input type="submit">--!>

<button type="submit" name="submit" formaction="index.php">Submit</button>
<button type="submit" name="save" formaction="plansave.php">Save to CSV</button>

</form>
</p>



<?php include("config.php"); ?>

<?php
$conn = new mysqli($servername, $username, $password, $dbname);
// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
} 

echo "<h4>Query</h4><p>".$sql.";</p>\n\n";
$result = $conn->query($sql);
if ($result){
    if ($result->num_rows > 0) {
        echo "<h4>Result</h4><p>".$result->num_rows." rows returned.</p>\n\n";
        while($row = $result->fetch_assoc()) {
           if (empty($counter)){
              $counter = 0;
              echo "<div class='results-outer'>\n";
              echo "<table class='results' id='gentable'><thead><tr>\n";
              foreach(array_keys($row) as $paramName)
                echo "<th>".$paramName."</th>";
              echo "</tr></thead>\n";
           } 
           echo "<tr>";
           foreach(array_keys($row) as $paramName) {
               echo "<td>";
               if ($paramName == 'pl_name'){
                    echo "<a href='plandetail.php?name=".urlencode($row[$paramName]).
                    //"&scenario=".urlencode($row['scenario_name']).
                    //"&of_id=".urlencode($row['orbitfit_id']).
                    "'>".$row[$paramName]."</a>";                
               } else{
                   if (is_numeric($row[$paramName])){
                       echo number_format((float)$row[$paramName], 2, '.', '');

                   } else{
                       echo $row[$paramName];
                   }
               }
               echo "</td>";    
           }
           echo "</tr>";
         $counter++;
        }
        echo "</table></div>\n";
    }
    $result->close();
} else{
    echo "Query Error:\n".$conn->error;
}
$conn->close();
?>


<?php include "templates/footer.php"; ?>

