import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import numpy as np
import glob

# read the data
data_files = glob.glob("./output/data*.pickle")

data = []
for ff in data_files:
  # attributes
  attrib = {}
  # load
  d = pickle.load(open(ff, "rb"))  
  attrib['mode']= d['mode']
  attrib['ni']  = d['num_inducing']
  if d['mode'] == 'SVGP':
    d['num_directions']= 0
  attrib['nd']  = d['num_directions']
  attrib['M']   = d['num_inducing']*(d['num_directions']+1)
  attrib['nll'] = d['test_nll'].item()
  attrib['mse'] = d['test_mse'].item()
  attrib['test_time']  = d['test_time']
  attrib['train_time'] = d['train_time']
  # add an indicator attribute for plotting
  attrib['indic'] = d['mode'] + str(d['num_directions'])
  data.append(attrib)
# make a pandas df
df = pd.DataFrame.from_dict(data,orient='columns')
print(df)

# plot
sns.set()
sns.lineplot(x='M',y='nll',hue='indic',style='indic',palette='colorblind',err_style='band',markers=True,dashes=False,linewidth=3,data=df)
plt.title("NLL vs Inducing Matrix size")
plt.ylabel("NLL")
plt.xlabel("Inducing Matrix Size")
plt.show()

