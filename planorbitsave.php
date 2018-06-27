<?php 

$name = $_POST["name"]; 

$namestring = $name;
//Make alphanumeric (removes all other characters)
$namestring = preg_replace("/[^a-zA-Z0-9_\s+-.]/", "", $namestring);
//Convert whitespaces to underscore
$namestring = preg_replace("/[\s]/", "_", $namestring);

include("config.php");

$conn = new mysqli($servername, $username, $password, $dbname);
// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
} 

$sql = "select * from PlanetOrbits where Name = '".$name."'";
$result = $conn->query($sql);
if ($result){
    if ($result->num_rows > 0) {
        header('Content-Type: application/csv');
        header('Content-Disposition: attachment; filename="'.$namestring.'_orbit_data.csv";');
        $f = fopen('php://output', 'w');
        
        while($row = $result->fetch_assoc()) {
            if (empty($counter)){
                $counter = 1;
                fputcsv($f, array_keys($row), ",");
            } 

           fputcsv($f, $row, ",");
        }
    }
    $result->close();
} else{
    echo "Query Error:\n".$conn->error;
}
$conn->close();

?>

