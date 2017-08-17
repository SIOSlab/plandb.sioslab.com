<?php include "templates/header.php"; ?>

<h2> Planet Detail Query </h2>

<?php $sqlsel = "SELECT CONCAT(pl_hostname,' ',pl_letter) as Name FROM KnownPlanets WHERE "; ?>
<p>
This interface filters planets of interest and links to details pages for the results.  Only enter the conditions clause of the query (the part after WHERE).  The select portion is always:</br>
<?php echo "$sqlsel";?>

 <form action="planquery.php" method="POST">
    <textarea name="querytext" rows="4" cols="100">
<?php 
if (empty($_POST["querytext"]))
    $sql = "pl_maxangsep > 150 AND pl_minangsep < 450";
else
    $sql = $_POST["querytext"]; 
echo $sql;
?>
</textarea><br>
<input type="submit">
</form>
</p>


<?php include("config.php"); ?>

<?php
$conn = new mysqli($servername, $username, $password, $dbname);
// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
} 

echo "<p>Query:</br>".$sqlsel.$sql.";</p>\n\n";
$result = $conn->query($sqlsel.$sql);
if ($result){
    if ($result->num_rows > 0) {
        echo "<div class='results-outer'>\n";
        echo "<table class='results' id='gentable'><thead><tr>\n";
        echo "<th>Planet Name</th>";
        echo "</tr></thead>\n";
        while($row = $result->fetch_assoc()) {
            echo "<tr><td>";
            echo "<a href='plandetail.php?name=".urlencode($row["Name"])."'>".$row["Name"]."</a>";
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


