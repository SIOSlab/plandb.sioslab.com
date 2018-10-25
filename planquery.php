<?php 
include "templates/header.php"; 
include "templates/headerclose.php"; 
?>

<h2> Known Planet Detail Query </h2>

<?php 
$sqlsel = "SELECT pl_name as Name, pl_angsep, st_optmag,completeness FROM KnownPlanets WHERE "; 
$sqlord = "ORDER by completeness DESC";
?>

<p>See <a href="index.php?querytext=show columns in KnownPlanets">"show columns in KnownPlanets"</a> for all available columsn to query on.
This interface filters planets of interest and links to detail pages for the results.  Only enter the conditions clause of the query (the part after WHERE):</br></br>

<?php echo "$sqlsel";?>
 <form action="planquery.php" method="POST">
    <textarea name="querytext" rows="4" cols="100">
<?php 
if (empty($_POST["querytext"]))
    $sql = "completeness > 0";
else
    $sql = $_POST["querytext"]; 
echo $sql;
?>
</textarea><br>
<input type="submit">
</form>
</p>
<?php echo "$sqlord";?>


<?php include("config.php"); ?>

<?php
$conn = new mysqli($servername, $username, $password, $dbname);
// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
} 

echo "<h4>Query</h4><p>".$sqlsel.$sql." ".$sqlord.";</p>\n\n";
$result = $conn->query($sqlsel.$sql." ".$sqlord);
if ($result){
    if ($result->num_rows > 0) {
        echo "<h4>Result</h4><p>".$result->num_rows." rows returned.</p>\n\n";

        echo "<div class='results-outer'>\n";
        echo "<table class='results' id='gentable'><thead><tr>\n";
        echo "<th>Planet Name</th><th>Separation (mas)</th><th>Optical Magnitude</th><th>Completeness</th>";
        echo "</tr></thead>\n";
        while($row = $result->fetch_assoc()) {
            echo "<tr><td>";
            echo "<a href='plandetail.php?name=".urlencode($row["Name"])."'>".$row["Name"]."</a>";
            echo "</td><td>";
            echo $row["pl_angsep"];
            echo "</td><td>";
            echo $row["st_optmag"];
            echo "</td><td>";
            echo $row["completeness"];
            echo "</td></tr>";
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


