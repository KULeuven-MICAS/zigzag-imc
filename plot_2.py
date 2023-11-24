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


network = "resnet8"
with open(f"imc_weights_{network}.pkl","rb") as infile:
    df_all_weights = pickle.load(infile)

df_all_weights['type'] = 'all_weights_on_chip'
df_all_weights = df_all_weights.sort_values(by=['area','latency_total'],ascending=[True,True],ignore_index=True)
df_aw_pareto = pd.DataFrame()
best_latency = float('inf')
for i,r in df_all_weights.iterrows():
    if r.latency_total < best_latency:
        df_aw_pareto = df_aw_pareto.append(r)
        best_latency = r.latency_total
df_aw_pareto['type'] = 'all_weight_pareto'

with open(f"onchip_weights_{network}.pkl","rb") as infile:
    df_onchip_weights = pickle.load(infile)
df_onchip_weights = df_onchip_weights.sort_values(by=['area','latency_total'],ascending=[True,True],ignore_index=True)

df_onchip_weights['type'] = 'sram_weights_on_chip'
df_onchip_pareto = pd.DataFrame()
best_latency = float('inf')
for i,r in df_onchip_weights.iterrows():
    if r.latency_total < best_latency:
        df_onchip_pareto = df_onchip_pareto.append(r)
        best_latency = r.latency_total
df_onchip_pareto['type'] = 'onchip_pareto'


with open(f"offchip_weights_{network}.pkl","rb") as infile:
    df_offchip_weights = pickle.load(infile)
df_offchip_weights = df_offchip_weights.sort_values(by=['area','latency_total'],ascending=[True,True],ignore_index=True)
df_offchip_weights['type'] = 'dram_weights_off_chip'
df_offchip_pareto = pd.DataFrame()
best_latency = float('inf')
for i,r in df_offchip_weights.iterrows():
    if r.latency_total < best_latency:
        df_offchip_pareto = df_offchip_pareto.append(r)
        best_latency = r.latency_total
df_offchip_pareto['type'] = 'offchip_pareto'


breakpoint()
df = pd.concat([df_all_weights, df_onchip_weights, df_offchip_weights, df_aw_pareto, df_onchip_pareto, df_offchip_pareto])
fig = px.scatter(df, 'area', 'latency_total', color='type',log_y=True, log_x=True)
fig.show()
fig = px.scatter(df, 'area', 'energy_total', color='type',log_y=True, log_x=True)
fig.show()
