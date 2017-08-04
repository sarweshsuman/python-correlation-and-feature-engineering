from scipy.stats import pearsonr
from python_utilities.read_input_pandas import read_csv
import sys

file_name = sys.argv[1]


data_frame = read_csv(file_name,',')

# For Sex vs Survived comparison
#x1=data_frame['Sex'].replace('male',0)
#x1=x1.replace('female',1)
#x1=x1.values

# For Age vs Survived
#data_frame = data_frame.dropna()
#x1 = data_frame['Age'].values
#x2 = data_frame['Survived'].values

# For Fare vs Survived
x1 = data_frame['Fare'].values
x2 = data_frame['Survived'].values

print("Comparing between {} and {}".format(x1,x2))
print(pearsonr(x2,x1))
