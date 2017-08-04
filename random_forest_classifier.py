from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from python_utilities.read_input_pandas import read_csv
import sys
import numpy as np

file_name = sys.argv[1]
data_frame = read_csv(file_name,',')

# These columns contains strings, for now dropping it, need to find later how these can be taken into account
narray = data_frame.drop(['PassengerId','Name','Sex','Ticket','Embarked','Cabin'],axis=1).dropna()

narray_x = narray.drop(['Survived'],axis=1).values
narray_x_col_names = narray.drop(['Survived'],axis=1).columns.values

narray_y = narray['Survived'].values

rf = RandomForestClassifier(n_estimators=30,max_depth=4)

scores = []

for i in range(narray_x.shape[1]):
	score = cross_val_score(rf,narray_x[:,i:i+1],narray_y,scoring='r2',cv=ShuffleSplit(len(narray_x),10,.3))
	scores.append((round(np.mean(score),3),narray_x_col_names[i]))
print(sorted(scores,reverse=True))
