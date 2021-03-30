from skimage import io
import	random
import	sys
import	argparse
import	seaborn as sns
import	matplotlib.pyplot as plt
import	numpy as np
import	pandas as pd
from scipy import stats, integrate
import statsmodels.api as sm
from scipy import signal



def	possible_values(Cl, X, Y):
	L = []
	for i in range(X):
		for j in range(Y):
			v = Cl[i][j]
			if v not in L:
				L.append(v)
	return L

# Extract the k closest points to (x,y) from newLung
# Short cut: extract the poitns in the square delimited by (x-k), (y-k) - (x+k),y+k)
def	extract_neighs(newLung, x, y, k, Xo, Yo):
	NeighList = []
	NeighList.append(newLung[x][y])
	no_elem_window = (2 * k) + 1
	window_value = signal.gaussian(no_elem_window, std=2)
	values_neigh = window_value[k+1:]
	for level in range(k, 0, -1):
		list_level = []
		for i in range(x-level, x+level+1):
			if i>=0 and i<Xo:
				if (i == (x-level)) or (i == (x+level)):
					for j in range(y-level, y+level+1):
						if j>=0 and j<Yo:
							list_level.append(newLung[i][j] * values_neigh[level-1])
						else:
							list_level.append(0.0)
				else:
					if (y-level)>=0:
						list_level.append(newLung[i][j] * values_neigh[level-1])
					else:
						list_level.append(0.0)
					if (y+level)<Yo:
						list_level.append(newLung[i][j] * values_neigh[level-1])
					else:
						list_level.append(0.0)
			else:
				list_level.append(0.0)
		NeighList += list_level


	# for i in range(x - k, x + k + 1):
	# 	for j in range(y - k, y + k + 1):
	# 		if i > 0 and i < Xo and j > 0 and j < Yo:
	# 			NeighList.append(newLung[i][j])
	# 		else:
	# 			NeighList.append(0.0)
	avg = np.mean(NeighList)

	return [NeighList, avg]

def	normalize(atts):
	nAtts = []
	mn = min(atts)
	mx = max(atts)
	r = mx - mn
	for v in atts:
		nv = (v - mn) / r
		nAtts.append(nv)

	return nAtts
	

def	save_atts(FF, Pts, attsLung):

	f = open(FF, "w")
	for p in Pts:
		for j in attsLung[p]:
			f.write( str(j) + "\t")
		f.write( str(p[0]) + "_" + str(p[1]) + "\n")
	f.close()

def	save_csv(FF, X, Y, Lung):
	f = open(FF, "w")
	for i in range(X):
		for j in range(Y):
			f.write(str(i) + "\t" + str(j) + "\t" + str(Lung[i][j]) + "\n")
	f.close()
	

"""
This program reads a lung triplet consisting of:
1. The original image
2. The mask for the lung
3. The <<class>> of the pixel

python extract_atts_lung.py  -i Train/tr_im8.png  -m  LungMask/tr_lungmask8.png  -c Mask/tr_mask8.png  -o results/im_8.csv  -res_images/im_8.png
"""
parser = argparse.ArgumentParser()
parser.add_argument('-i', action = "store", dest = "i", help = "The input file containing the original lung image")
parser.add_argument('-m', action = "store", dest = "m", help = "The input file containing the lung mask (what is lung)")
parser.add_argument('-c', action = "store", dest = "c", help = "The input file containing the class of the lung  ground-glass (1), consolidation(2), pleural effusion (3)")
parser.add_argument('-r', action = "store", dest = "r", help = "The output file containing the data matrix,, normalized for each pixel ")
parser.add_argument('-rr', action = "store", dest = "rr", help = "The output file containing the data matrix,, normalized globally")
parser.add_argument('-o', action = "store", dest = "o", help = "The output file containing the image")
parser.add_argument('-s', action = "store", dest = "s", help = "The output file containing the image in csv")
parser.add_argument('-o2', action = "store", dest = "o2", help = "The output file containing the image when averaged by nk neighbors")
parser.add_argument('-nk', action = "store", dest = "nk", help = "The number of neighbos of each pixel to compare with")

args = parser.parse_args()

Orig  = io.imread(args.i)
[Xo, Yo] =  Orig.shape
print "S = ", Xo, Yo

Values_Orig = possible_values(Orig, Xo, Yo)
#print "Orig = ", Orig[49]
#print "PV Orig = ", Values_Orig

Lung  = io.imread(args.m)
# Values: 0, 128
#print "Lung = ", Lung[49], Lung[380]
[Xl, Yl] =  Lung.shape
print "S = ", Xl, Yl
Values_Lung = possible_values(Lung, Xl, Yl)
print "PV Lung = ", Values_Lung
# PV Lung =  [0, 128, 255]

Cl = io.imread(args.c)
# Values: 0, 128, 255
[Xc, Yc] =  Cl.shape
#print "Cl = ", Cl[49], Cl[380]
print "S = ", Xc, Yc
Values_Cl = possible_values(Cl, Xc, Yc)
#print "PV CL = ", Values_Cl
# PV CL =  [0, 170, 85, 255]
# PV CL =  [0, 255, 128]
# PV CL =  [0, 255]
# Inconsistencies...?

newLung = Orig.copy()
for i in range(Xo):
	for j in range(Yo):
		if Lung[i][j] > 0:
			newLung[i][j] = Orig[i][j]
		else:
			newLung[i][j] = 0


#newLung = apply_mask(Orig, Xo, Yo, Lung, Xl, Yl)
#print "nL = ", newLung[5]
io.imsave(args.o, newLung)

# Create the feature vector. For each vector filtered in (newLung), select 
# some features

newLungNeigh = newLung.copy()

nk = int(args.nk)
attsLung = {}
attsLung_nonorm = {}
for i in range(Xo):
	if i % 50 == 0:
		print "row = ", i
	for j in range(Yo):
		if newLung[i][j] > 0:
			#the value of its k nearest neighbors
			[Neighs, avg] = extract_neighs(newLung, i, j, nk, Xo, Yo)
			avg = np.mean(Neighs)
			mx = max(Neighs)
			mn = min(Neighs)
			H = stats.entropy(Neighs)

			atts = Neighs
			atts.append(avg)
			atts.append(mx)
			atts.append(mn)
			atts.append(H)
			attsN = normalize(atts)
			#print "p = ", i, j, atts
			#attsLung[(i,j)] = atts
			attsLung[(i,j)] = attsN
			# The non-normilized attributes
			attsLung_nonorm[(i,j)] = atts

			newLungNeigh[i][j] = int(avg)
		else:
			newLungNeigh[i][j] = 0

# Normalize data considering all vectors for each attribute
kK = attsLung.keys()
L = []
for k in kK:
	L.append(attsLung_nonorm[k])

L = np.array(L)

ln = len(attsLung[k])
mn = [1000000.0] * ln
mx = [-1000000.0] * ln
R = [0.0] * ln
for i in range(ln):
	#print "L0 = ", L[0], len(L[0])
	#print "i = ", i
	P = L[:,i]
	#print "P = ", P
	mn[i] = min(P)
	mx[i] = max(P)
	R[i] = mx[i] - mn[i]

attsLungN = {}
for k in kK:
	L = []
	for i,a in enumerate(attsLung_nonorm[k]):
		v = (a - mn[i]) / R[i]
		L.append(v)
	attsLungN[k] = L
Pts = attsLungN.keys()
save_atts(args.rr, Pts, attsLungN)

Pts = attsLung.keys()
save_atts(args.r, Pts, attsLung)

io.imsave(args.o2, newLungNeigh)

save_csv(args.s, Xo, Yo, newLungNeigh)
print "done"
