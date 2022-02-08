<?php 

$sql = $_POST["querytext"]; 

include("config.php");

$conn = new mysqli($servername, $username, $password, $dbname);
// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
} 

$result = $conn->query($sql);
if ($result){
    if ($result->num_rows > 0) {
        header('Content-Type: application/csv');
        header('Content-Disposition: attachment; filename="output.csv";');
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

