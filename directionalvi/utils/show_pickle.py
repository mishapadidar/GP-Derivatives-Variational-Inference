import pickle
import sys

filename = sys.argv[1]

# load the test point
with open(filename, "rb") as f:
   d= pickle.load(f)
   for item in d.items():
     print(item)
