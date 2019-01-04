'''
Source, destination, trip and route file remover & generator.
'''

import numpy as np
import warnings
import os
import inspect
from subprocess import DEVNULL, STDOUT, call



def delete(prefix):
	'''
	Args:
		prefix: Prefix of the (to-be-deleted) files' names.
	Returns:
		Always None; Tries to delete the existing source/destination files as well as the trips files.
	'''
	# Fetch working directory.
	working_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
	try:
		# The scr and dst values
		call("del "+prefix+".src.xml", shell=True, cwd=working_dir, stdout=DEVNULL, stderr=STDOUT)
		call("del "+prefix+".dst.xml", shell=True, cwd=working_dir, stdout=DEVNULL, stderr=STDOUT)
	except:
		warnings.warn("SRC/DST Files not found.")
	try:
		# The generated routes
		call("del trips.trips.xml", shell=True, cwd=working_dir, stdout=DEVNULL, stderr=STDOUT)
		call("del "+prefix+".rou.alt.xml", shell=True, cwd=working_dir, stdout=DEVNULL, stderr=STDOUT)
		call("del "+prefix+".rou.xml", shell=True, cwd=working_dir, stdout=DEVNULL, stderr=STDOUT)
	except:
		warnings.warn("TRIP/ROUTE Files not found.")

def generate(prefix, src, dst, rng, scale=(0,0)):
	'''
	Args:
		prefix: Prefix of the (to-be-generated) files' names.
		src: List of the names of the source edges to be used.
		dst: List of the names of the destination edges to be used.
		scale: Tuple of 2 std.dev.
	Returns:
		Always None; Generates new source/destination files as well as trips files.
	'''
	# Fetch working directory.
	working_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
	# SRC & DST GENERATION
	a = rng.normal(loc=100, scale=scale[0], size=len(src))
	a = a/a.sum()
	file = open(prefix+".src.xml", "w")
	startstring = str('''<edgedata>''')+str('''\n''')+str(''' <interval begin="0" end="3600"/>''')
	midstring = ""
	for src_id, weight in zip(src,a):
		midstring += str('''\n''')+str('''  <edge id="'''+str(src_id)+'''" value="'''+str(weight)+'''"/>''')
	endstring = str('''\n''')+str(''' </interval>''')+str('''\n''')+str('''</edgedata>''')
	file.write(startstring+midstring+endstring)
	file.close()

	a = rng.normal(loc=100, scale=scale[1], size=len(dst))
	a = a/a.sum()
	file = open(prefix+".dst.xml", "w")
	startstring = str('''<edgedata>''')+str('''\n''')+str(''' <interval begin="0" end="3600"/>''')
	midstring = ""
	for dst_id, weight in zip(dst,a):
		midstring += str('''\n''')+str('''  <edge id="'''+str(dst_id)+'''" value="'''+str(weight)+'''"/>''')
	endstring = str('''\n''')+str(''' </interval>''')+str('''\n''')+str('''</edgedata>''')
	file.write(startstring+midstring+endstring)
	file.close()

	# ROUTE GENERATION
	# -b: beginning period of route creation in seconds.
	# -e: ending period of route creation in seconds.
	# -p: creation of a route from -b to -e every -p periods.
	command = "python randomTrips.py -n "+prefix+".net.xml -a vtype.cfg -r "+prefix+".rou.xml -b 0 -e 3600 -p 1.5 -s "+str(rng.randint(0,1e6))+" -l --weights-prefix "+prefix+"  --trip-attributes departSpeed=\'max\' --trip-attributes type=\'base\'"
	call(command, shell=True, cwd=working_dir, stdout=DEVNULL, stderr=STDOUT)