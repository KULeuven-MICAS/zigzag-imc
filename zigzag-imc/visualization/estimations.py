import pickle
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd


def dimc_res8_estimations():
    with open('data_output.pickle', 'rb') as infile:
        data_output = pickle.load(infile)
    data_output.sort_values(by=['workload', 'layer_node'], ascending=True, ignore_index=True, inplace=True)
    data_dict = [] 
    for v in data_output['voltage'].unique():
        for pidx in data_output['paper_idx'].unique():
            for i, r in data_output[(data_output.workload == 'resnet8') & (data_output.voltage == v) & (data_output.paper_idx == pidx)].iterrows():
                for t in ['Ewlbl', 'Eadder', 'Emem_O', 'Emem_W', 'Emem_I', 'Emul']:
                    data_dict.append({})
                    data_dict[-1]['idx'] = i
                    data_dict[-1]['workload'] = 'resnet8'
                    data_dict[-1]['layer_node'] = r.layer_node
                    data_dict[-1]['energy'] = r[t]
                    data_dict[-1]['type'] = t
                    data_dict[-1]['voltage'] = v
                    data_dict[-1]['paper_idx'] = pidx
            

    df = pd.DataFrame(data_dict)
    df = df[df.voltage == 0.8]

    fig = px.bar(df, x='layer_node', y='energy',  color='type', facet_row='paper_idx', log_y=True)
    fig.show()
    


if __name__ == "__main__":
    dimc_res8_estimations()
