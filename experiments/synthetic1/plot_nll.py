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
  if d['mode'] == "ExactGradGP":
    attrib['mode'] = d['mode']
    attrib['run']  = d['mode']
    attrib['M']   = d['M']
    attrib['nll'] = d['test_nll'].item()
    attrib['rmse'] = np.sqrt(d['test_mse'].item())
    print(f"ExactGradGP nll: {d['test_nll'].item()}, rmse: {np.sqrt(d['test_mse'].item())}")
    # dont plot ExactGradGP
    # data.append(attrib)
    continue

  attrib['ni']  = d['num_inducing']
  if d['mode'] == 'SVGP':
    d['num_directions']= 0
  attrib['nd']  = d['num_directions']
  attrib['M']   = d['num_inducing']*(d['num_directions']+1)
  attrib['nll'] = d['test_nll'].item()
  attrib['rmse'] = np.sqrt(d['test_mse'].item())
  attrib['test_time']  = d['test_time']
  attrib['train_time'] = d['train_time']
  if d['mode'] == 'SVGP' and d['mll_type'] == 'PLL':
    d['mode'] = "PPGPR"
  elif d['mode'] == 'DSVGP' and d['mll_type'] == 'PLL':
    d['mode'] = "DPPGPR"
  elif d['mode'] == 'GradSVGP' and d['mll_type'] == 'PLL':
    d['mode'] = "GradPPGPR"
  if "D" in d['mode'] or "Grad" in d['mode']:
    attrib['run'] = d['mode'] + str(d['num_directions'])
  else:
    attrib['run'] = d['mode']

  # reduce points
  #if not np.any(np.isclose(attrib['M'],[800],atol=10)):
  #  continue
  if not np.any(np.isclose(attrib['M'],[200,400,800,1200],atol=10)):
    continue
  # reduce methods
  if not attrib['run'] in ['SVGP','PPGPR','GradSVGP5','GradPPGPR5','DSVGP2','DPPGPR2','DSKI','ExactGradGP']:
    continue
  data.append(attrib)
# make a pandas df
df = pd.DataFrame.from_dict(data,orient='columns')
pd.to_pickle(df,"sin5_plot_data.pickle")
#df = df[df['run']!='GradSVGP3']
print(df)

# plot
rc = {'figure.figsize':(10,5),
      'axes.facecolor':'white',
      'axes.grid' : True,
      'grid.color': '.8',
      'font.family':'Times New Roman',
      'font.size' : 15}
plt.rcParams.update(rc)
#sns.lineplot(x='M',y='nll',hue='run',style='run',palette='colorblind',err_style='band',markers=True,dashes=False,linewidth=5,markersize=12,data=df)
sns.lineplot(x='M',y='rmse',hue='run',style='run',palette='colorblind',err_style='band',markers=True,dashes=False,linewidth=5,markersize=12,data=df)
plt.title("NLL vs Inducing Matrix size")
plt.ylabel("NLL")
plt.xlabel("Inducing Matrix Size")
plt.show()

