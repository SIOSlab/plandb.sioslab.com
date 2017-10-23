<html>
<head>
  <!-- Plotly.js -->
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  
</head>



<?php
$name = "GJ 849 b";
$sql = "SELECT * FROM Completeness WHERE Name='".$name."'";
include("config.php"); 
$conn = new mysqli($servername, $username, $password, $dbname);
// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
} 
$result = $conn->query($sqlsel.$sql);
?>


<body>

<?php echo $result->num_rows; ?>

<!-- Plotly chart will be drawn inside this DIV -->
<div id="myDiv" style="width:480px; height:400px;"></div>
  <script>
var x = [];
var y = [];
for (var i = 0; i < 500; i ++) {
    x[i] = Math.random();
    y[i] = Math.random() + 1;
}

var data = [
  {
    x: x,
    y: y,
    type: 'histogram2d'
  }
];
Plotly.newPlot('myDiv', data);
  </script>
</body>
</html>
