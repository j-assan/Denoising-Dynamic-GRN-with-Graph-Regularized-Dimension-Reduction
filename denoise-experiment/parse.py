import numpy as np
import json

def argparser(filepath = None):
	sweep = {}
	args = {}
	to_args = False
	if filepath == None:
		file = open("input.in")
	else:
		file = open(filepath)
	data = file.readlines()
	for line in data:
		if line.strip().startswith("#") or line.strip() == '':
			continue
		if line.strip() == 'args':
			to_args = True
			continue
		# parse input, assign values to variables
		key, value = line.split("=")
		value = value.strip()
		if '[(' in value:
			value = str2list(value)
		elif '[' in value:
			value = json.loads(value)
		elif '(' in value:
			value = tuple(map(int, value.strip('()').split(',')))
		elif '.' in value:
			value = float(value)
		elif value == "True":
			value = True
		elif value == "False":
			value = False
		elif value == "None":
			value = None
		else:
			try:
				value = int(value)
			except:
				pass
		if to_args: args[key.strip()] = value
		else: sweep[key.strip()] = value
	file.close()
	return sweep, args


def str2list(strArray):

	lItems = []
	trans = str.maketrans({' ': '', '[': '', ']': ''})
	strArray = strArray.translate(trans)
	for tup in strArray.split("),"):
		trans = str.maketrans({'(': '', ')': ''})
		tup = tup.translate(trans)
		lParts = tup.split(',')
		n = len(lParts) 
		assert n == 2, "not a tuple"
		lItems.append((int(lParts[0]), int(lParts[1])))
	return lItems
