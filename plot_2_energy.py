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

with open("all_weights_pareto_resnet8_energy.pkl","rb") as infile:
    df_all_weights = pickle.load(infile)

df_all_weights['type'] = 'all_weights_on_chip'
#df_all_weights['latency_total'] = df_all_weights['latency_cc']

with open("onchip_weights_resnet8_energy.pkl","rb") as infile:
    df_onchip_weights = pickle.load(infile)

df_onchip_weights['type'] = 'sram_weights_on_chip'

with open("offchip_weights_resnet8_energy.pkl","rb") as infile:
    df_offchip_weights = pickle.load(infile)
df_offchip_weights['type'] = 'dram_weights_off_chip'

df = pd.concat([df_all_weights, df_onchip_weights, df_offchip_weights])
fig = px.scatter(df, 'area', 'energy_total', color='type',log_y=True, log_x=True)
fig.show()
