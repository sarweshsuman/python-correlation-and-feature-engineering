import pandas as pd

from python_utilities.read_input_pandas import read_csv
import sys

file_name = sys.argv[1]

data_frame = read_csv(file_name,',')

print(data_frame.corr())  # will give the correlation between each data point vs other data point in a tabular format.
