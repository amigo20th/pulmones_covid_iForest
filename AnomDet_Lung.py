#
import	pandas as pd
import	numpy as np
import	argparse
from pyod.models.lof import LOF
from pyod.models.iforest import IForest

"""
python AnomDet_Lung.py  -i results/im_2_k_3.csv  -m lof/if/mad  -o  AD/im_2_k_3_AD_lof_if_mad.csv  -th 10
"""

parser = argparse.ArgumentParser()
parser.add_argument('-i', action = "store", dest = "i", help = "The input file containing the description of each pixel")
parser.add_argument('-m', action = "store", dest = "m", help = "The anom det method: lof, if, mad")
parser.add_argument('-th', action = "store", dest = "th", help = "The relevant parameter for the anom det method")
parser.add_argument('-o', action = "store", dest = "o", help = "The output file containing the anomaly code for each pixel (row)")
args = parser.parse_args()

print "loading"
DV = np.loadtxt(args.i, dtype = 'str', delimiter = '\t')
print "loading done"

Vars = []
nv = len(DV[0])
for i in range(nv):
	Vars.append(i)

#df = pd.DataFrame(Vars)

print "v = ", len(Vars)
df = pd.DataFrame(DV, columns = Vars)

# The last column is the vector label, no needed
df = df.iloc[:, :-1]

df.apply(pd.to_numeric)

print "x"
#Number of states
St = 5
Label = []
if args.m == 'lof' or args.m == 'LOF' or args.m == 'Lof':
	print "lof"
	#print "D = ", df[:30]
	numK = int(args.th)
	print "creating LOF"
	clf = LOF(n_neighbors=numK)
	print "fitting LOF"
	#E = np.array(df[:30])
	E = np.array(df)
	E = np.asfarray(E,float)
	print "E = ", E[0:30]
	clf.fit(E)
	#clf.fit(df[:50])
	print "fitting done"
	#print "l = ", clf.decision_scores_
	mn = min(clf.decision_scores_)
	mx = max(clf.decision_scores_)
	R = mx - mn
	print "R = ", R, mn, mx
	for lab in clf.decision_scores_:
		v = int( St * ( (lab - mn)/R) )
		Label.append(v)
else:
	if args.m == 'IF' or args.m == 'if' or args.m == 'If':
		print "lof"
		#print "D = ", df[:30]
		numE = int(args.th)
		print "creating IF"
		clf = IForest(n_estimators=numE)
		print "fitting IF"
		#E = np.array(df[:30])
		E = np.array(df)
		E = np.asfarray(E,float)
		#print "E = ", E[0:30]
		clf.fit(E)
		#clf.fit(df[:50])
		print "fitting done"
		#print "l = ", clf.decision_scores_
		mn = min(clf.decision_scores_)
		mx = max(clf.decision_scores_)
		R = mx - mn
		print "R = ", R, mn, mx
		for lab in clf.decision_scores_:
			v = int( St * ( (lab - mn)/R) )
			Label.append(v)



print "saving"
#print "L = ", Label
np.savetxt(args.o, Label, fmt='%i')
