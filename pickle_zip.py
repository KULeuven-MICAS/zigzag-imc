import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np
import pandas as pd
import pickle
import os
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import plotly.express as px
import shutil

import zigzag

def get_data(directory):
    ii  = 0
    all_data = []
    for filename in os.listdir(directory):
        ii+=1
        if filename.split(".")[-1] != 'pkl':
            continue
        f = os.path.join(directory, filename)
        
        with open(f, 'rb') as infile:
            data = pickle.load(infile)
        all_data.append(data)
    return all_data



for network in ['resnet8','deepautoencoder','mobilenet_v1','ds_cnn']:
    data = get_data(f"./outputs_{network}/dimc/")
    with open(f"./outputs_{network}/dimc/{network}_all_data_points.pkl","wb") as infile:
        pickle.dump(data,infile)


