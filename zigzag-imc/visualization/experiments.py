import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import json
import os
import pandas as pd
import matplotlib.pyplot as plt


def get_data_experiment_1():
    typey = {'A':'Act.buffer','ADC':'ADC','B':'Weight writing','DAC':'DAC','O':'Output buffer','accumulation_energy':'Adder tree','cell':'E$_{cell}$','multiplication_energy':'E$_{mul}$','precharging_cell':
            'E$_{BL}$','I':'Act.buffer','W':'Weight writing'}
    typey = {'A':'I.buffer','ADC':'ADC','B':'W.write','DAC':'DAC','O':'O.buffer','accumulation_energy':'Adder tree','cell':'E$_{cell}$','multiplication_energy':'E$_{mul}$','precharging_cell':
            'E$_{BL}$','I':'I.buffer','W':'W.write'}
    
    dict_data = []
    for accel in ['dimc', 'aimc', 'aimc_dimc', 'dimc2']:
        dir_list = ['../outputs/'+accel+'/mobilenet', '../outputs/'+accel+'/resnet8', '../outputs/'+accel+'/ds-cnn', '../outputs/'+accel+'/ae']
        workload = ['mobilenet', 'resnet8', 'ds-cnn', 'autoencoder']
        for ii_d, directory in enumerate(dir_list):
            for file in os.listdir(directory):
                filename = os.fsdecode(file)
                if filename.endswith(".json"):
                    layer_index = filename.split("-")
                    layer_index = layer_index[0].split("_")
                    layer_index = eval(layer_index[1])
                    dict_datax = []
                    with open(directory + '/' + filename) as infile:
                        values = json.load(infile)
                    for energy_type, energy_val in values['outputs']['energy']['operational_energy_breakdown'].items():
                        dict_datax.append({})
                        dict_datax[-1]['workload'] = workload[ii_d]
                        dict_datax[-1]['layer'] = layer_index
                        dict_datax[-1]['energy'] = energy_val
                        dict_datax[-1]['type'] = typey[energy_type]
                        dict_datax[-1]['accelerator'] = accel
                    for energy_type, energy_val in values['outputs']['energy']['energy_breakdown_per_level'].items():
                        dict_datax.append({})
                        dict_datax[-1]['workload'] = workload[ii_d]
                        dict_datax[-1]['layer'] = layer_index
                        dict_datax[-1]['energy'] = energy_val[0]
                        dict_datax[-1]['type'] = typey[energy_type]
                        dict_datax[-1]['accelerator'] = accel
                    typex = ['A','ADC','B','DAC','O','accumulation_energy','cell','multiplication_energy','precharging_cell','I','W']
                    typex = set(typey.values())
                    for x in typex:
                        if x not in [xx['type'] for xx in dict_datax]:
                            dict_datax.append({'workload':workload[ii_d], 'layer':layer_index,'energy':0,'type':x,'accelerator':accel})
                    dict_data += dict_datax
    df = pd.DataFrame(dict_data)
    
    df.sort_values(by=['layer'], ignore_index=True, ascending=True, inplace=True)
    #fig = px.bar(df[df.accelerator == 'aimc'], x='layer', y='energy', color='type', facet_row='workload')
    #fig.show()

    df = df.groupby(['workload', 'type', 'accelerator']).agg({'energy': 'sum'})
    df = df.reset_index()

    return df

def plot_experiment_1():
    df = get_data_experiment_1()
    df.fillna(0)
    plt.rcParams['text.usetex'] = True
    
    typex = {'A':'I.buffer','ADC':'ADC','B':'W.write','DAC':'DAC','O':'O.buffer','accumulation_energy':'Adder tree','cell':'E$_{cell}$','multiplication_energy':'E$_{mul}$','precharging_cell':
            'E$_{BL}$','I':'I.buffer','W':'W.write'}
    workloads = {'autoencoder':'DeepAutoEncoder','mobilenet':'MobileNetV1', 'ds-cnn':'DS-CNN','resnet8':'ResNet8'}
    plt.rcParams['figure.figsize'] = (9, 4)
    y_peak = {'autoencoder': 528384/2,     
            'mobilenet':14978304/2,
            'ds-cnn':5312000/2,
            'resnet8':25003264/2,
            }
    y_fjMAC = {'aimc':16.5e-9,
            'aimc_dimc':35e-9,
            'dimc':44.44e-9,
            'dimc2':43.35e-9}

    fig, ax = plt.subplots(1,4, sharey=False)
    for ii_w, w in enumerate(df.workload.unique()):
        table = df[df.workload == w]
        ax[ii_w].grid(visible=True, which='major', axis='y')
        bottom = np.zeros(4)
        for t in table.type.unique():
            tablet = table[table.type == t]
            x = tablet['accelerator'].values.tolist()
            y = tablet['energy'].values.tolist()
            y = [yy/1e6 for yy in y]
            ax[ii_w].bar([0,1,2,3], y, width = 0.8, bottom = bottom, label=t)  
            bottom = bottom + np.array(y)
        ax[ii_w].scatter([0,1,2,3], [y_peak[w] * y2 for y2 in y_fjMAC.values()], marker='*', s=220, c='r')

        ax[ii_w].set_title(workloads[w], fontsize=16)
#        plt.yscale('log')
        ax[ii_w].set_xticks(ticks=[0,1,2,3], labels=x, rotation=30, ha='right')
        ax[ii_w].tick_params(axis='both', labelsize=16)

    #plt.xlabel('Design', size=20)
#        for i, nn in enumerate(x):
#            plt.annotate(nn, (nx[i], max(y[i], y2[i])*1.2), ha='center', size=12)
    plt.legend(fontsize=10,loc='center left',ncol=1,bbox_to_anchor=(1.04, 0.5))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_experiment_1()
    pass
