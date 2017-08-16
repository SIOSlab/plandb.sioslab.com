<?php include "templates/header.php"; ?>


<p>
 <form action="index.php" method="POST">
        <textarea name="querytext" rows="4" cols="150">
<?php 
if (empty($_POST["querytext"]))
    $sql = "select pl_hostname, pl_letter,pl_angsep,pl_minangsep,pl_maxangsep,pl_radj,pl_bmassj,pl_orbsmax from KnownPlanets where pl_maxangsep > 150 AND pl_minangsep < 450";
else
    $sql = $_POST["querytext"]; 
echo "$sql";
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

echo "<p>Query:</br>".$sql.";</p>\n\n";
$result = $conn->query($sql);
if ($result){
    if ($result->num_rows > 0) {
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
               if (is_numeric($row[$paramName])){
                   echo number_format((float)$row[$paramName], 2, '.', '');

               } else{
                   echo $row[$paramName];
               }
                echo "</td>";    
           }
           echo "</tr>";
         $counter++;
        }
        echo "</table</div>\n";
    }
    $result->close();
} else{
    echo "Query Error:\n".$conn->error;
}
$conn->close();
?>


<?php include "templates/footer.php"; ?>

