<!doctype html>
<html lang="en">

<head>
	<meta charset="utf-8">
	<meta http-equiv="x-ua-compatible" content="ie=edge">
	<meta name="viewport" content="width=device-width, initial-scale=1">

	<title>Planet Database</title>

    <link rel="stylesheet" href="css/main.css">

    <script src="//code.jquery.com/jquery-1.11.1.js"></script>
    <script src="js/jquery.tablesorter.js"></script>
    <script type="text/javascript">
    $(function() {		
        $("#gentable").tablesorter();
    });	
    </script>

</head>

<body>
	<h1>Known Planet Database</h1>

<p>See the IPAC <a href='https://exoplanetarchive.ipac.caltech.edu/docs/API_exoplanet_columns.html'>schema here</a> for available column names.  Additional columns are: smax_from_orbper (bool: semi-major axis calculated from orbital period), pl_maxangsep, pl_minangsep, rad_from_mass (planet radius (pl_radj only) calculated from pl_bmassj using Forecaster). You can also query "show columns in KnownPlanets"</p> 

