import numpy as np
import warnings
import os

def delete():
	try:
		# The scr and dst values
		os.remove("trone.src.xml")
		os.remove("trone.dst.xml")
	except:
		warnings.warn("TRIPS.PY: Src/Dst files not found.")
	try:
		# The generated routes
		os.remove("trips.trips.xml")
		os.remove("trone.rou.alt.xml")
		os.remove("trone.rou.xml")
	except:
		warnings.warn("TRIPS.PY: Route files not found.")

def generate():
	# SRC & DST GENERATION
	a = np.random.randint(10, size=5)
	a = a/a.sum()
	file = open("trone.src.xml", "w")
	file.write(str('''
	<edgedata>
		<interval begin="0" end="600"/>
			<edge id="231940830#1" value="'''+str(a[0])+'''"/> <!--trone:from centre-->
			<edge id="-16233798#1" value="'''+str(a[1])+'''"/> <!--sceptre:from jourdan-->
			<edge id="-8382447#1" value="'''+str(a[2])+'''"/> <!--trone:from etterbeek-->
			<edge id="450925440#0" value="'''+str(a[3])+'''"/> <!--henriette-->
			<edge id="17670100" value="'''+str(a[4])+'''"/> <!--malibran:from flagey-->
		</interval>
	</edgedata>'''))
	file.close()

	a = np.random.randint(10, size=9)
	a = a/a.sum()
	file = open("trone.dst.xml", "w")
	file.write('''
	<edgedata>
		<interval begin="0" end="600"/>
			<edge id="-231940830#1" value="'''+str(a[0])+'''"/> <!--trone:to centre-->
			<edge id="14362915" value="'''+str(a[1])+'''"/> <!--street to EC-->
			<edge id="16233798#1" value="'''+str(a[2])+'''"/> <!--sceptre:to jourdan-->
			<edge id="8382447#1" value="'''+str(a[3])+'''"/> <!--trone:to etterbeek-->
			<edge id="524151750" value="'''+str(a[4])+'''"/> <!--wery-->
			<edge id="-450925440#0" value="'''+str(a[5])+'''"/> <!--henriette-->
			<edge id="-17670100" value="'''+str(a[6])+'''"/> <!--malibran:to flagey-->
			<edge id="4730637#0" value="'''+str(a[7])+'''"/> <!--soucis-->
			<edge id="510492456" value="'''+str(a[8])+'''"/> <!--goffart-->
		</interval>
	</edgedata>''')
	file.close()
	warnings.warn("TRIPS.PY: Src/Dst files created.")

	# ROUTE GENERATION
	os.system("python randomTrips.py -n trone.net.xml -r trone.rou.xml -b 0 -e 600 -p 1 -l --weights-prefix trone  --trip-attributes=departSpeed=\'max\'")
	warnings.warn("TRIPS.PY: Route files created.")