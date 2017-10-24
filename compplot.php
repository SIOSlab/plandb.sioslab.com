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

var xsize = 400, ysize = 260, x = new Array(xsize), y = new Array(ysize), z = new Array(ysize), i, j;

x[0] = 100.5;
for(var i = 1; i < xsize; i++) {
    x[i] = x[i-1]+1;
}

y[0] = 0.05;
for(var i = 1; i < ysize; i++) {
    y[i] = y[i-1]+0.1;
}

for (var i = 0; i < ysize; i++) {
    z[i] = new Array(xsize).fill(0);
}

<?php 
    while($row = $result->fetch_assoc()) {
        echo "z[".$row[jind]."][".$row[iind]."]=".$row[H].";";
    }
    echo "\n\n";
?>

var data = [ {
		z: z,
		x: x,
		y: y,
		type: 'contour'
	}
];


Plotly.newPlot('myDiv', data);
  </script>
</body>
</html>
