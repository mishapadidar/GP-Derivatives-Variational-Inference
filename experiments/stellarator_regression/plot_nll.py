import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import numpy as np
import glob

# read the data
data_files = glob.glob("./output/data_stell_regress_*.pickle")

plt.figure(figsize=(10,10))

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
  attrib['run'] = d['mode']
  if d['mode'] == 'SVGP' and d['mll_type'] == 'PLL':
    d['mode'] = "PPGPR"
  elif d['mode'] == 'DSVGP' and d['mll_type'] == 'PLL':
    d['mode'] = "DPPGPR"
  if "D" in d['mode']:
    attrib['run'] = d['mode'] + str(d['num_directions'])
  else:
    attrib['run'] = d['mode']
  data.append(attrib)
# make a pandas df
df = pd.DataFrame.from_dict(data,orient='columns')
#df = df[df['M'] > 400]
print(df)

# plot
rc = {'figure.figsize':(10,5),
      'axes.facecolor':'white',
      'axes.grid' : True,
      'grid.color': '.8',
      'font.family':'Times New Roman',
      'font.size' : 15}
plt.rcParams.update(rc)
#sns.set()
#sns.set_style("whitegrid")
#sns.set_context("paper", font_scale=2.0)
sns.lineplot(x='M',y='nll',hue='run',style='run',palette='colorblind',err_style='band',markers=True,dashes=False,linewidth=5,markersize=12,data=df)
plt.title("NLL vs Inducing Matrix size")
plt.ylabel("NLL")
plt.xlabel("Inducing Matrix Size")
plt.legend(loc=1)
plt.show()

