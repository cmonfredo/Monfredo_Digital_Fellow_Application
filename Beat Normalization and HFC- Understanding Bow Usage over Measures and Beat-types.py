#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scipy


# In[3]:


def plot_stuff(beats, hfc, measures):
    beats_df = pd.read_csv(beats,header=None,names=["time","value"])
    hfc_df = pd.read_csv(hfc,header=None,names=["time","value"])
    beats_df["beat_length"] =  beats_df.time.shift(-1)-beats_df.time
    
    npoints = 100
    dfs = []
    for i,row in beats_df.iloc[:-1].iterrows():
        l = row.beat_length
        s = row.time
        dfs.append(pd.DataFrame(dict(time=np.linspace(s,s+l,npoints),
                                     idx = range(npoints),
                                     beat_id = row.value)))
        
    new_df = pd.concat(dfs).reset_index(drop=True)
    f = scipy.interpolate.interp1d(hfc_df.time,hfc_df.value,kind="quadratic")
    new_df["value"] = f(new_df.time)
    x = new_df.pivot("idx","beat_id","value")
    fig, axs = plt.subplots(measures)
    fig.set_size_inches(6, 3*measures)
    for i in range(measures):
        x.iloc[:,slice(i*8,(i+1)*8)].plot(colormap="cool",ax=axs[i])


# In[4]:


def plot_stuffbeats(beats, hfc, measures):
    beats_df = pd.read_csv(beats,header=None,names=["time","value"])
    hfc_df = pd.read_csv(hfc,header=None,names=["time","value"])
    beats_df["beat_length"] =  beats_df.time.shift(-1)-beats_df.time
    
    npoints = 21
    dfs = []
    for i,row in beats_df.iloc[:-1].iterrows():
        l = row.beat_length
        s = row.time
        dfs.append(pd.DataFrame(dict(time=np.linspace(s,s+l,npoints),
                                     idx = range(npoints),
                                     beat_id = row.value)))
        
    new_df = pd.concat(dfs).reset_index(drop=True)
    f = scipy.interpolate.interp1d(hfc_df.time,hfc_df.value,kind="quadratic")
    new_df["value"] = f(new_df.time)
    x = new_df.pivot("idx","beat_id","value")
    fig, axs = plt.subplots(8)
    fig.set_size_inches(6, 3*8)
    for i in range(8):
        x.iloc[:,slice(i,measures*8,8)].plot(colormap="cool",ax=axs[i])


# # Wispelwey, Lipkind, Diaz: by measure (8 beats per measure)

# In[5]:


plot_stuff("../Note markers/Wispelwey beats.csv","../HFC files/Wispelwey HFC.csv",8)


# In[6]:


plot_stuff("../Note markers/Lipkind beats.csv","../HFC files/Lipkind HFC.csv",8)


# In[7]:


plot_stuff("../Note markers/Diaz beats.csv","../HFC files/Diaz HFC.csv",8)


# # By beat within measure, Wispelwey, Lipkind, Diaz

# In[8]:


plot_stuffbeats("../Note markers/Wispelwey beats.csv","../HFC files/Wispelwey HFC.csv",8)


# In[8]:


plot_stuffbeats("../Note markers/Lipkind beats.csv","../HFC files/Lipkind HFC.csv",8)


# In[10]:


plot_stuffbeats("../Note markers/Diaz beats.csv","../HFC files/Diaz HFC.csv",8)


# In[ ]:




