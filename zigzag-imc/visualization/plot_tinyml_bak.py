import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from get_values_sheet import *


def aimc_topsw_topsmm2_plot():
    
    color = ['violet', 'orange', 'lightblue', 'tan', 'steelblue', 'palevioletred']
    tablex = get_values("14sOT8JJPU9aHC3rwLHjFkLKOu9jYdaPon6qBTLBuOeU", "AIMC model!A1:AX27")
    title_color = 'Technology node [nm]'
    plt.rcParams['text.usetex'] = True
    plt.rcParams['figure.figsize'] = (6, 5)
    for ii, l in enumerate(tablex[title_color].unique()):
        table = tablex[tablex[title_color] == l]
        x = table['TOPs/mm2'].values.tolist()
        y = table['TOP/s/W'].values.tolist()
        n = table['Technology node [nm]'].values.tolist()
        n = table['Idx'].values.tolist()
        x = [eval(xx) for xx in x]
        y = [eval(yy) for yy in y]
        n = [f'[{n}]' for n in n]
        plt.scatter(x = x, y = y, s=70, color=color[ii], label = l)
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('TOP/s/mm2', size=15)
        plt.ylabel('TOP/s/W', size=15)


        for i, nx in enumerate(n):
            plt.annotate(nx, (x[i]*1.2, y[i]*1.2))

    plt.legend(title=title_color, loc='lower right')
    plt.grid(visible=True, which='minor', axis='both', c='gainsboro', linestyle='--', zorder=-1.0)
    plt.grid(visible=True, which='major', axis='both', linestyle='-', zorder=-1.0)
    plt.rcParams['axes.axisbelow'] = True
    plt.tight_layout()
    plt.show()


def aimc_topsw_topsmm2_1b_plot():
    
    color = ['violet', 'orange', 'lightblue', 'tan', 'steelblue', 'palevioletred']
    tablex = get_values("14sOT8JJPU9aHC3rwLHjFkLKOu9jYdaPon6qBTLBuOeU", "AIMC model!A1:BH27")
    title_color = 'Technology node [nm]'
    plt.rcParams['text.usetex'] = True
    plt.rcParams['figure.figsize'] = (6, 5)
    for ii, l in enumerate(tablex[title_color].unique()):
        table = tablex[tablex[title_color] == l]
        x = table['TOPs/mm2-1b'].values.tolist()
        y = table['TOPs/W-1b'].values.tolist()
        n = table['Technology node [nm]'].values.tolist()
        n = table['Idx'].values.tolist()
        x = [eval(xx) for xx in x]
        y = [eval(yy) for yy in y]
        n = [f'[{n}]' for n in n]
        plt.scatter(x = x, y = y, s=70, color=color[ii], label = l)
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('1b-TOP/s/mm2', size=15)
        plt.ylabel('1b-TOP/s/W', size=15)


        for i, nx in enumerate(n):
            plt.annotate(nx, (x[i]*1.2, y[i]*1.2))

    plt.legend(title=title_color, loc='lower right')
    plt.grid(visible=True, which='minor', axis='both', c='gainsboro', linestyle='--', zorder=-1.0)
    plt.grid(visible=True, which='major', axis='both', linestyle='-', zorder=-1.0)
    plt.rcParams['axes.axisbelow'] = True
    plt.tight_layout()
    plt.show()



def dimc_topsw_topsmm2_plot():
    
    color = ['violet', 'orange', 'lightblue', 'tan']
    tablex = get_values("14sOT8JJPU9aHC3rwLHjFkLKOu9jYdaPon6qBTLBuOeU", "copy_DIMC_topsw_model!A1:AX24")
    tablex = tablex.sort_values(by=['Technology node [nm]'], ignore_index =True, inplace=False)
    title_color = 'Technology node [nm]'
    plt.rcParams['text.usetex'] = True
    plt.rcParams['figure.figsize'] = (6, 5)
    for ii, lt in enumerate(tablex[title_color].unique()):
        table = tablex[tablex[title_color] == lt]
        if(lt=='28'): table = tablex[tablex[title_color] == lt].sort_values(by=['TOPs/mm2'])
        x = table['TOPs/mm2'].values.tolist()
        y = table['TOP/s/W'].values.tolist()
        pi= table['Input precision'].values.tolist()
        pw= table['Weight precision'].values.tolist()
        v = table['Voltage (V)'].values.tolist()
        n = table['Idx'].values.tolist()
        x = [eval(xx) for xx in x]
        y = [eval(yy) for yy in y]
        pi= [eval(pi) for pi in pi]
        pw= [eval(pw) for pw in pw]
        n = [f'[{n}]' for n in n]
        #plt.scatter(x = x, y = y, s=70, color=color[ii], label = lt)
        x = list(np.multiply(np.multiply(x,pi),pw)) ## normalize to 1b Ops
        y = list(np.multiply(np.multiply(y,pi),pw)) ## normalize to 1b Ops
        plt.plot(x,y, linewidth=2, color=color[ii], label = lt, marker='o', markersize=8)
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('Area Efficiency [1b-TOP/s/mm${^2}$]', size=15)
        plt.ylabel('Energy Efficiency [1b-TOP/s/W]', size=15)
        con = f"{n[ii]} {(table['Input precision'].values)[0]}b, {lt}nm"
        if(lt!='28'):
            plt.annotate(con, (np.average(x)*0.7, np.average(y)*1.5))
        else:
            plt.annotate(con, (np.average(x)*0.7, np.average(y)*0.65))
        for i, vx in enumerate(v):
            plt.annotate(f'{vx}V', (x[i]*1, y[i]*1))

        #for i, nx in enumerate(n):
        #    plt.annotate(nx, (x[i]*1.1, y[i]*1.1))

    plt.legend(title=title_color, loc='upper right')
    plt.grid(visible=True, which='minor', axis='both', c='gainsboro', linestyle='--', zorder=-1.0)
    plt.grid(visible=True, which='major', axis='both', linestyle='-', zorder=-1.0)
    plt.rcParams['axes.axisbelow'] = True
    plt.tight_layout()
    plt.show()


def aimc_vs_dimc_topsw_topsmm2_plot():
    
    color = ['violet', 'orange', 'lightblue', 'tan', 'steelblue', 'palevioletred']
    markers = ['o', '^']
    tablex = get_values("14sOT8JJPU9aHC3rwLHjFkLKOu9jYdaPon6qBTLBuOeU", "AIMC model!A1:AX27")
    title_color = 'Idx'
    plt.rcParams['text.usetex'] = True
    plt.rcParams['figure.figsize'] = (6, 4)
    for ii, l in enumerate(tablex[title_color].unique()):
        
        table = tablex[tablex[title_color] == l]
        x = table['TOPs/mm2'].values.tolist()
        y = table['TOP/s/W'].values.tolist()
        n = table['Idx'].values.tolist()
        v = table['Voltage (V)'].values.tolist()
        pi = table['Input precision'].values.tolist()
        pw = table['Weight precision'].values.tolist()
        pw_cp = table['Weight precision'].astype(float)
        po = table['Output precision'].values.tolist()
        BPBS = table['BPBS'].astype(float)
        x = [eval(xx) for xx in x]
        y = [eval(yy) for yy in y]
        if n[0]=='10':
            idx = np.argsort(pw_cp * BPBS).tolist()
            tmpx = []
            tmpy = []
            for i in idx:
                tmpx.append(x[i])
                tmpy.append(y[i])
            
            
        if ii == 0:
            plt.plot(x, y, color=color[-2], marker = markers[0], label = 'AIMC')
        elif n[0] == '10':
            plt.plot(tmpx[0:2], tmpy[0:2], color=color[-2], marker = markers[0])# 4b/4b
            plt.plot(tmpx[2:4], tmpy[2:4], color=color[-2], marker = markers[0])# 4b/8b, PBPS: 1
            plt.plot(tmpx[4:], tmpy[4:], color=color[-2], marker = markers[0])# 4b/8b, PBPS: 2
            plt.plot(x[0:3], y[0:3], color=color[-2], marker = markers[0], alpha=0.2)# 0.7V
            plt.plot(x[3:], y[3:], color=color[-2], marker = markers[0], alpha=0.2)# 0.9V
        else:
            plt.plot(x, y, color=color[-2], marker = markers[0])
        n = [f'[{n}]' for n in n]

        if n[0] == '[0]':
            plt.annotate(f'{n[0]}, {pi[0]}b/{pw[0]}b', (np.average(x)*1, np.average(y)*1.4), color=color[-2], ha='center')
        elif n[0] == '[8]':
            plt.annotate(f'{n[0]}, {pi[0]}b/{pw[0]}b', (np.average(x)*1, np.average(y)*1.2), color=color[-2], ha='center')
        elif n[0] == '[10]':
            plt.annotate(f'{n[0]}', (np.average(x)*1, np.average(y)*1.1), color=color[-2], ha='center')
            plt.annotate(f'{pi[idx[0]]}b/{pw[idx[0]]}b', (np.average(tmpx[0:2])*0.7, np.average(tmpy[0:2])*0.8), color=color[-2], ha='center')
            plt.annotate(f'{pi[idx[-1]]}b/{pw[idx[-1]]}b', (np.average(tmpx[2:4])*1.3, np.average(tmpy[2:4])*1), color=color[-2], ha='center')
            plt.annotate(f'{pi[idx[-1]]}b/{pw[idx[-1]]}b', (np.average(tmpx[4:])*0.7, np.average(tmpy[4:])*0.8), color=color[-2], ha='center')
        elif n[0] == '[17]':
            plt.annotate(f'{n[0]}, {pi[0]}b/{pw[0]}b', (np.average(x)*1.4, np.average(y)*0.7), color=color[-2], ha='center')
        elif n[0] == '[6]':
            plt.annotate(f'{n[0]}, {pi[0]}b/{pw[0]}b', (np.average(x)*1.2, np.average(y)*0.7), color=color[-2], ha='center')
        elif n[0] == '[11]':
            plt.annotate(f'{n[0]}, {pi[0]}b/{pw[0]}b', (np.average(x)*0.7, np.average(y)*0.3), color=color[-2], ha='center')
        elif n[0] == '[12]':
            plt.annotate(f'{n[0]}, {pi[0]}b/{pw[0]}b', (np.average(x)*0.9, np.average(y)*1.2), color=color[-2], ha='center')
        elif n[0] == '[14]':
            plt.annotate(f'{n[0]}, {pi[0]}b/{pw[0]}b', (np.average(x)*0.3, np.average(y)*0.6), color=color[-2], ha='center')
        else:
            plt.annotate(f'{n[0]}, {pi[0]}b/{pw[0]}b', (np.average(x)*1, np.average(y)*1.2), color=color[-2], ha='center')
        #for i, vx in enumerate(v):
        #    if(i==0):
        #        plt.annotate(f'{pi[i]}b/{pw[i]}b', (x[i]*1, y[i]*0.8), ha='center')
        #    elif(v[i] == v[i-1]):
        #        plt.annotate(f'{pi[i]}b/{pw[i]}b', (x[i]*1, y[i]*0.8))

    tablex = get_values("14sOT8JJPU9aHC3rwLHjFkLKOu9jYdaPon6qBTLBuOeU", "copy_DIMC_topsw_model!A1:AM24")
    tablex = tablex.sort_values(by=['Technology node [nm]'], ignore_index =True, inplace=False)
    title_color = 'Idx'
    for ii, l in enumerate(tablex[title_color].unique()):
        if tablex[title_color].unique()[ii] == '3':
            # TOPs/mm2 for the 3rd paper is not in order. Looks mess when plt.plot
            table = tablex[tablex[title_color] == l].sort_values(by=['TOPs/mm2'])
        else:
            table = tablex[tablex[title_color] == l]
        x = table['TOPs/mm2'].values.tolist()
        y = table['TOP/s/W'].values.tolist()
        n = table['Idx'].values.tolist()
        x = [eval(xx) for xx in x]
        y = [eval(yy) for yy in y]
        v = table['Voltage (V)'].values.tolist()
        pi = table['Input precision'].values.tolist()
        pw = table['Weight precision'].values.tolist()
        n = [f'[{n}]' for n in n]
        if ii == 0:
            plt.plot(x, y, color=color[-1], marker = markers[1], label = 'DIMC')
        else:
            plt.plot(x, y, color=color[-1], marker = markers[1])

        if(n[0] == '[3]'):
            plt.annotate(f'{n[0]}, {pi[0]}b/{pw[0]}b', (np.average(x)*0.5, np.average(y)*1.5), color=color[-1])
        else:
            plt.annotate(f'{n[0]}, {pi[0]}b/{pw[0]}b', (np.average(x)*1.1, np.average(y)*1.1), color=color[-1])
        #for i, vx in enumerate(v):
        #    plt.annotate(f'{vx}V', (x[i]*1.1, y[i]*1.1))

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('TOP/s/mm$^2$', size=15)
    plt.ylabel('TOP/s/W', size=15)

    plt.legend(loc='lower right')
    plt.grid(visible=True, which='minor', axis='both', c='gainsboro', linestyle='--', zorder=-1.0)
    plt.grid(visible=True, which='major', axis='both', linestyle='-', zorder=-1.0)
    plt.rcParams['axes.axisbelow'] = True
    plt.tight_layout()
    plt.show()




def aimc_vs_dimc_topsw_topsmm2_1b_plot():
    
    color = ['violet', 'orange', 'lightblue', 'tan', 'steelblue', 'palevioletred']
    markers = ['o', '^']
    tablex = get_values("14sOT8JJPU9aHC3rwLHjFkLKOu9jYdaPon6qBTLBuOeU", "AIMC model!A1:BH27")
    title_color = 'Idx'
    plt.rcParams['text.usetex'] = True
    plt.rcParams['figure.figsize'] = (6, 4)
    for ii, l in enumerate(tablex[title_color].unique()):
        table = tablex[tablex[title_color] == l]
        x = table['TOPs/mm2-1b'].values.tolist()
        y = table['TOPs/W-1b'].values.tolist()
        #n = table['Technology node [nm]'].values.tolist()
        n = table['Idx'].values.tolist()
        v = table['Voltage (V)'].values.tolist()
        pi = table['Input precision'].values.tolist()
        pw = table['Weight precision'].values.tolist()
        pw_co = table['Weight precision'].astype(float)
        BPBS = table['BPBS'].astype(float)
        po = table['Output precision'].values.tolist()
        x = [eval(xx) for xx in x]
        y = [eval(yy) for yy in y]
        if n[0]=='10':
            idx = np.argsort(pw_co * BPBS).tolist()
            tmpx = []
            tmpy = []
            for i in idx:
                tmpx.append(x[i])
                tmpy.append(y[i])
        if ii == 0:
            plt.plot(x, y, color=color[-2], marker = markers[0], label = 'AIMC')
        elif n[0] == '10':

            plt.plot(tmpx[0:2], tmpy[0:2], color=color[-2], marker = markers[0])# 4b/4b
            plt.plot(tmpx[2:4], tmpy[2:4], color=color[-2], marker = markers[0])# 4b/8b, PBPS: 1
            plt.plot(tmpx[4:], tmpy[4:], color=color[-2], marker = markers[0])# 4b/8b, PBPS: 2
            plt.plot(x[0:3], y[0:3], color=color[-2], marker = markers[0], alpha=0.2)# 0.7V
            plt.plot(x[3:], y[3:], color=color[-2], marker = markers[0], alpha=0.2)# 0.9V
        else:
            plt.plot(x, y, color=color[-2], marker = markers[0])

        n = [f'[{n}]' for n in n]
        #find a suitable location for annotation for below cases
        if n[0] == '[6]':
            plt.annotate(n[0], (np.average(x)*1.1, np.average(y)*1.2), color=color[-2], ha='center')
        elif n[0] == '[2]':
            plt.annotate(n[0], (np.average(x)*0.8, np.average(y)*0.8), color=color[-2], ha='center')
        elif n[0] == '[10]':
            plt.annotate(n[0], (np.average(x)*0.7, np.average(y)*1), color=color[-2], ha='center')
        elif n[0] == '[14]':
            plt.annotate(n[0], (np.average(x)*0.7, np.average(y)*0.8), color=color[-2], ha='center')
        elif n[0] == '[11]':
            plt.annotate(n[0], (np.average(x)*0.4, np.average(y)*0.2), color=color[-2], ha='center')
        elif n[0] == '[0]':
            plt.annotate(n[0], (np.average(x)*0.7, np.average(y)*1), color=color[-2], ha='center')
        elif n[0] == '[8]':
            plt.annotate(n[0], (np.average(x)*0.7, np.average(y)*1.1), color=color[-2], ha='center')
        elif n[0] == '[13]':
            plt.annotate(n[0], (np.average(x)*1.2, np.average(y)*1.2), color=color[-2], ha='center')
        else:
            plt.annotate(n[0], (np.average(x)*1, np.average(y)*1.2), color=color[-2], ha='center')
        #for i, vx in enumerate(v):
        #    plt.annotate(f'{v[i]}V, {pi[i]}, {pw[i]}', (x[i]*0.8, y[i]*1), color=color[-2])

    tablex = get_values("14sOT8JJPU9aHC3rwLHjFkLKOu9jYdaPon6qBTLBuOeU", "copy_DIMC_topsw_model!A1:AX24")
    tablex = tablex.sort_values(by=['Technology node [nm]'], ignore_index =True, inplace=False)
    title_color = 'Idx'
    tablex['TOPs/W-1b'] = tablex['TOP/s/W'].astype(float) * tablex['Input precision'].astype(float) * tablex['Weight precision'].astype(float)
    tablex['TOPs/mm2-1b'] = tablex['TOPs/mm2'].astype(float) * tablex['Input precision'].astype(float) * tablex['Weight precision'].astype(float)
    for ii, l in enumerate(tablex[title_color].unique()):
        if tablex[title_color].unique()[ii] == '3':
            # TOPs/mm2 for the 3rd paper is not in order. Looks mess when plt.plot
            table = tablex[tablex[title_color] == l].sort_values(by=['TOPs/mm2'])
        else:
            table = tablex[tablex[title_color] == l]
        x = table['TOPs/mm2-1b'].values.tolist()
        y = table['TOPs/W-1b'].values.tolist()
        n = table['Idx'].values.tolist()
        v = table['Voltage (V)'].values.tolist()
        #x = [eval(xx) for xx in x]
        #y = [eval(yy) for yy in y]
        n = [f'[{n}]' for n in n]
        if ii == 0:
            #plt.scatter(x = x, y = y, s=70, color=color[-1], marker = markers[1], label = 'DIMC')
            plt.plot(x, y, color=color[-1], marker = markers[1], label = 'DIMC')
        else:
            #plt.scatter(x = x, y = y, s=70, color=color[-1], marker = markers[1])
            plt.plot(x, y, color=color[-1], marker = markers[1])

        #for i, vx in enumerate(v):
        #    plt.annotate(f'{vx}V', (x[i]*0.8, y[i]*1), color=color[-1])
        plt.annotate(n[0], (np.average(x)*1.1, np.average(y)*1.1), color=color[-1])

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('1b-TOP/s/mm$^2$', size=15)
    plt.ylabel('1b-TOP/s/W', size=15)

    plt.legend(loc='lower right')
    plt.grid(visible=True, which='minor', axis='both', c='gainsboro', linestyle='--', zorder=-1.0)
    plt.grid(visible=True, which='major', axis='both', linestyle='-', zorder=-1.0)
    plt.rcParams['axes.axisbelow'] = True
    plt.tight_layout()
    plt.show()


def aimc_vs_dimc_topsw_tops_1b_plot():
    
    color = ['violet', 'orange', 'lightblue', 'tan', 'steelblue', 'palevioletred']
    markers = ['o', '^']
    tablex = get_values("14sOT8JJPU9aHC3rwLHjFkLKOu9jYdaPon6qBTLBuOeU", "AIMC model!A1:BH27")
    title_color = 'Technology node [nm]'
    plt.rcParams['text.usetex'] = True
    plt.rcParams['figure.figsize'] = (6, 4)
    for ii, l in enumerate(tablex[title_color].unique()):
        table = tablex[tablex[title_color] == l]
        x = table['TOPs-1b'].values.tolist()
        y = table['TOPs/W-1b'].values.tolist()
        n = table['Technology node [nm]'].values.tolist()
        n = table['Idx'].values.tolist()
        x = [eval(xx) for xx in x]
        y = [eval(yy) for yy in y]
        n = [f'[{n}]' for n in n]
        if ii == 0:
            plt.scatter(x = x, y = y, s=70, color=color[-2], marker = markers[0], label = 'AIMC')
        else:
            plt.scatter(x = x, y = y, s=70, color=color[-2], marker = markers[0])

        for i, nx in enumerate(n):
            plt.annotate(nx, (x[i]*1.2, y[i]*1.2))

    tablex = get_values("14sOT8JJPU9aHC3rwLHjFkLKOu9jYdaPon6qBTLBuOeU", "copy_DIMC_topsw_model!A1:AM24")
    tablex = tablex.sort_values(by=['Technology node [nm]'], ignore_index =True, inplace=False)
    title_color = 'Technology node [nm]'
    tablex['TOPs/W-1b'] = tablex['TOP/s/W'].astype(float) * tablex['Input precision'].astype(float) * tablex['Weight precision'].astype(float)
    tablex['TOPs-1b'] = tablex['TOPs/mm2'].astype(float) * tablex['Input precision'].astype(float) * tablex['Weight precision'].astype(float)
    for ii, l in enumerate(tablex[title_color].unique()):
        table = tablex[tablex[title_color] == l]
        x = table['TOPs-1b'].values.tolist()
        y = table['TOPs/W-1b'].values.tolist()
        n = table['Idx'].values.tolist()
        #x = [eval(xx) for xx in x]
        #y = [eval(yy) for yy in y]
        n = [f'[{n}]' for n in n]
        if ii == 0:
            plt.scatter(x = x, y = y, s=70, color=color[-1], marker = markers[1], label = 'DIMC')
        else:
            plt.scatter(x = x, y = y, s=70, color=color[-1], marker = markers[1])

        for i, nx in enumerate(n):
            plt.annotate(nx, (x[i]*1.1, y[i]*1.1))

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('1b-TOP/s', size=15)
    plt.ylabel('1b-TOP/s/W', size=15)

    plt.legend(loc='lower right')
    plt.grid(visible=True, which='minor', axis='both', c='gainsboro', linestyle='--', zorder=-1.0)
    plt.grid(visible=True, which='major', axis='both', linestyle='-', zorder=-1.0)
    plt.rcParams['axes.axisbelow'] = True
    plt.tight_layout()
    plt.show()



def aimc_validation_plot():
    color = ['violet', 'orange', 'steelblue', 'palevioletred']
    table = get_values("14sOT8JJPU9aHC3rwLHjFkLKOu9jYdaPon6qBTLBuOeU", "AIMC model!A1:AX27")

    plt.rcParams['text.usetex'] = True
    x = table['Model / Paper'].values.tolist()
    y = table['fJ/MAC'].values.tolist()
    y2 = table['Total'].values.tolist()
    n = table['Idx'].values.tolist()
    x = [f'{abs(eval(xx))}\%' for xx in x]
    y = [eval(yy) for yy in y]
    y2 = [eval(yy) for yy in y2]
    nx = np.array([nn for nn,_ in enumerate(n)])
    n = [f'[{n}]' for n in n]
    
    plt.rcParams['figure.figsize'] = (20, 5)
    plt.grid(visible=True, which='major', axis='y')
    plt.bar(nx-0.2, y, width = 0.3, color=color[2], label='fJ/MAC reported')  
    plt.bar(nx+0.2, y2, width = 0.3, color=color[3], label='fJ/MAC estimated')  
    plt.legend(fontsize=15,loc='lower right')
    plt.yscale('log')
    plt.xticks(ticks=nx, labels=n, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Design', size=15)
    plt.ylabel('fJ / MAC', size=15)
    for i, nn in enumerate(x):
        plt.annotate(nn, (nx[i], max(y[i], y2[i])*1.2), ha='center', size=12)
    plt.tight_layout()
    plt.show()


def dimc_validation_plot():
    color = ['violet', 'orange', 'steelblue', 'palevioletred']
    tablex = get_values("14sOT8JJPU9aHC3rwLHjFkLKOu9jYdaPon6qBTLBuOeU", "copy_DIMC_topsw_model!A1:AX24")
    markers = ['^']
    plt.rcParams['text.usetex'] = True

    plt.rcParams['figure.figsize'] = (6, 5)
    for ii, idx in enumerate(tablex['Idx'].unique()):
        table = tablex[tablex['Idx'] == idx]
        x = table['Voltage (V)'].values.tolist()
        y = table['fJ/MAC'].values.tolist()
        y2 = table['Total'].values.tolist()
        n = table['Idx'].values.tolist()
        x = [eval(xx) for xx in x]
        y = [eval(yy) for yy in y]
        y2 = [eval(yy) for yy in y2]
        nx = np.array([nn for nn,_ in enumerate(n)])
        n = [f'[{n}]' for n in n]
        plt.scatter(x, y, color=color[ii], marker=markers[0], label=f'[{idx}]', s=70)
        plt.plot(x, y2, color=color[ii])

    f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
    handles = [f("s", color[i]) for i in range(3)]
    handles += [f(markers[i], "k") for i in range(1)]
    labels = [f'[{x}]' for x in tablex['Idx'].unique().tolist()] + ["Reported"]

    plt.legend(handles, labels, fontsize=12)
    plt.ylabel('fJ / MAC', fontsize=16)
    plt.xlabel('Vdd [V]', fontsize=16)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.grid()
    plt.show()
    

def dimc_regression_cap():
    x = [5, 22, 28, 65]
    y = [0.24, 0.32, 0.3, 1.56/2]#Cinv
    model = np.poly1d(np.polyfit(x, y, 1))
    polyline = np.linspace(1, 70, 70)
    plt.rcParams['text.usetex'] = True
    plt.rcParams['figure.figsize'] = (4, 4)
    plt.plot(polyline, model(polyline), c='steelblue')
    plt.scatter(x,y,s=100,c='lightblue')
    for i,xx in enumerate(x):
        pass
        if(xx==65):
            plt.annotate(f'{xx}nm', (x[i]-8,y[i]), size=12,ha='center')
        elif(xx==22):
            plt.annotate(f'{xx}nm', (x[i]-5,y[i]+0.03), size=12,ha='center')
        else:
            plt.annotate(f'{xx}nm', (x[i],y[i]+0.03), size=12,ha='center')
    plt.annotate('Cinv = %s' %str(model).lstrip(' \n'), (45,0.2), size=13, ha='center')
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.grid()
    plt.xlabel('Technology node [nm]', fontsize=16)
    plt.ylabel('Cinv [fF]', fontsize=16)
    print(model)
    plt.tight_layout()
    plt.show()


def aimc_output_ratio_plot():
    color = ['violet', 'orange', 'steelblue', 'palevioletred']
    table = get_values("14sOT8JJPU9aHC3rwLHjFkLKOu9jYdaPon6qBTLBuOeU", "AIMC model!A1:BE27")

    x = table['TOPs/mm2'].values.tolist()
    y = table['TOP/s/W'].values.tolist()
    y2 = table['INWOR'].values.tolist()
    n = table['Idx'].values.tolist()
    x = [eval(xx) for xx in x]
    y = [eval(yy) for yy in y]
    y2 = np.array([eval(yy) for yy in y2])
    nx = np.array([nn for nn,_ in enumerate(n)])
    n = [f'[{n}]' for n in n]

    plt.rcParams['figure.figsize'] = (6, 5)
    plt.scatter(x = x, y = y, s=70, c=y2, cmap=cm.viridis, norm=matplotlib.colors.LogNorm())
    plt.rcParams['text.usetex'] = True
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r'TOP/s/mm2', size=15)
    plt.ylabel(r'TOP/s/W', size=15)

    for i, nx in enumerate(n):
        plt.annotate(nx, (x[i]*1.2, y[i]*1.2))
    plt.grid(visible=True, which='minor', axis='both', c='gainsboro', linestyle='--', zorder=-1.0)
    plt.grid(visible=True, which='major', axis='both', linestyle='-', zorder=-1.0)
    plt.rcParams['axes.axisbelow'] = True
    plt.colorbar(label=r'IN $\times$ W $\times$ Output ratio')
    plt.tight_layout()
    plt.show()


def aimc_signal_margin_plot():
    color = ['violet', 'orange', 'steelblue', 'palevioletred']
    table = get_values("14sOT8JJPU9aHC3rwLHjFkLKOu9jYdaPon6qBTLBuOeU", "AIMC model!A1:BD27")

    plt.rcParams['text.usetex'] = True
    plt.rcParams['figure.figsize'] = (6, 5)
    x = table['Signal margin'].values.tolist()
    y = table['TOP/s/W'].values.tolist()
    n = table['Idx'].values.tolist()
    x = [eval(xx) for xx in x]
    y = [eval(yy) for yy in y]
    nx = np.array([nn for nn,_ in enumerate(n)])
    n = [f'[{n}]' for n in n]

    plt.scatter(x = x, y = y, s=70, c=color[2])
    plt.yscale('log')
    plt.xlabel('Signal margin [V]', size=15)
    plt.ylabel('TOP/s/W', size=15)

    for i, nx in enumerate(n):
        plt.annotate(nx, (x[i], y[i]*1.1))


    polyline = np.linspace(0, 0.03, 100)
    for i in [0.05, 0.1, 0.2, 0.4, .6, .8, 1, 1.2, 1.4, 1.6, 1.8, 2]:
        y = i/polyline
        plt.plot(polyline, y, c='lightblue')
 
    plt.grid(visible=True, which='minor', axis='both', c='gainsboro', linestyle='--', zorder=-1.0)
    plt.grid(visible=True, which='major', axis='both', linestyle='-', zorder=-1.0)
    plt.rcParams['axes.axisbelow'] = True
    plt.tight_layout()
    plt.show()

def dimc_cinv_fitting():
    color = ['violet', 'orange', 'steelblue', 'palevioletred']
    tablex = get_values("14sOT8JJPU9aHC3rwLHjFkLKOu9jYdaPon6qBTLBuOeU", "copy_DIMC_topsw_model!A1:AX24")

    tablex = tablex.sort_values(by=['Technology node [nm]'], ignore_index =True, inplace=False)
    title_color = 'Technology node [nm]'
    plt.rcParams['text.usetex'] = True
    plt.rcParams['figure.figsize'] = (6, 5)
    for ii, lt in enumerate(tablex[title_color].unique()):
        table = tablex[tablex[title_color] == lt].sort_values(by=['Voltage (V)'])
        if(table['Idx'].values[0]!='3'):
            Q = table['Weight precision'].astype(float)
        else:
            Q = table['Weight precision'].astype(float) + 1 ## specially for the 3rd paper with booth encoding
        K = table['parallelism'].astype(float)
        LOG2K = np.log2(K)
        LOG2T = np.log2( table['BPBS'].astype(float) )
        x = []
        y = []
        for cinv in np.arange(0.1,0.5,0.01):
            Eadder = 2 * cinv * 5 * table['Voltage (V)'].astype(float) * table['Voltage (V)'].astype(float) * ( Q*K+K-2-(Q+LOG2K-1) )/K * table['Input precision'].astype(float) / table['BPBS'].astype(float)
            if(table['Rows'].values[0] == '1'):
                Ewl = 0
                Ebl = 0
            else:
                Ewl = cinv * table['Voltage (V)'].astype(float) * table['Voltage (V)'].astype(float) * Q
                Ebl = cinv * table['Voltage (V)'].astype(float) * table['Voltage (V)'].astype(float) * Q * table['Rows'].astype(float)
            Emul = 2 * cinv * table['Voltage (V)'].astype(float) * table['Voltage (V)'].astype(float) * Q * table['Input precision'].astype(float) / table['BPBS'].astype(float)
            Edff = 2 * cinv * 5 * (Q+LOG2K+LOG2T) * table['Voltage (V)'].astype(float) * table['Voltage (V)'].astype(float) * table['Input precision'].astype(float) / table['BPBS'].astype(float) / (table['parallelism'].astype(float))
            if(table['Idx'].values[0]!='3'):
                #diff = abs((Eadder + Ewl + Ebl + Emul + Edff) / table['fJ/MAC'].astype(float)-1).mean()
                diff = abs((Eadder + Ewl + Ebl + Emul) / table['fJ/MAC'].astype(float)-1).mean()
            else:
                #diff = abs(0.5 * (Eadder + Ewl + Ebl + Emul + Edff) / table['fJ/MAC'].astype(float)-1).mean()
                diff = abs(0.5 * (Eadder + Ewl + Ebl + Emul) / table['fJ/MAC'].astype(float)-1).mean()
            x.append(cinv)
            y.append(diff * 100)
        minimal=(round(x[y.index(min(y))],2))

        plt.plot(x,y, linewidth=2, color=color[ii], label = lt, marker='o', markersize=8)
        plt.yscale('log')
        plt.xlabel('Cinv [fF]', size=15)
        plt.ylabel('Modeling Mismatch percentage [\%]', size=15)
        if(lt == '5'):
            plt.annotate(f'{minimal}fF', (0.2, min(y)), size=12, ha='center')
        else:
            plt.annotate(f'{minimal}fF', (minimal, 0.8*min(y)), size=12, ha='center')

    plt.legend(title=title_color, loc='upper right')
    plt.grid(visible=True, which='minor', axis='both', c='gainsboro', linestyle='--', zorder=-1.0)
    plt.grid(visible=True, which='major', axis='both', linestyle='-', zorder=-1.0)
    plt.rcParams['axes.axisbelow'] = True
    plt.tight_layout()
    plt.show()

def aimc_dac_fitting():
    color = ['violet', 'orange', 'steelblue', 'palevioletred']
    tablex = get_values("14sOT8JJPU9aHC3rwLHjFkLKOu9jYdaPon6qBTLBuOeU", "AIMC model!A1:BJ27")

    tablex = tablex.sort_values(by=['Idx'], ignore_index =True, inplace=False)
    title_color = 'Idx'
    plt.rcParams['text.usetex'] = True
    plt.rcParams['figure.figsize'] = (6, 5)

    Ecell = tablex['Cell [fJ]'].astype(float)
    Eadc = tablex['ADC/MAC [fJ]'].astype(float)
    Eaccumulation = tablex['Accumulation [fJ]'].astype(float)

    pi = tablex['Input precision'].astype(float)
    BPBS = tablex['BPBS'].astype(float)
    columns = tablex['Columns'].astype(float)
    volt_2 = tablex['Voltage (V)'].astype(float) * tablex['Voltage (V)'].astype(float)

    x = []
    y = []
    for base in np.arange(30,70,1):
        Edac = base * pi * BPBS/columns * volt_2
        diff = abs((Ecell + Eadc + Eaccumulation + Edac) / tablex['fJ/MAC'].astype(float)-1).mean() # %
        x.append(base)
        y.append(diff * 100)
    minimal=(round(x[y.index(min(y))],2))
    plt.plot(x,y, linewidth=2, color=color[-2], label = 'AIMC modeling', marker='o', markersize=8)
    plt.yscale('log')
    plt.xlabel('Edac/bit [fJ]', size=15)
    plt.ylabel('Modeling Mismatch percentage [\%]', size=15)
    plt.annotate(f'{minimal}fJ', (minimal+0.5, 1.01 * min(y)), size=15, ha='center')

    #for ii, lt in enumerate(tablex[title_color].unique()):
    #    table = tablex[tablex[title_color] == lt].sort_values(by=['Voltage (V)'])
    #    Ecell = table['Cell [fJ]'].astype(float)
    #    Eadc = table['ADC/MAC [fJ]'].astype(float)
    #    Eaccumulation = table['Accumulation [fJ]'].astype(float)

    #    pi = table['Input precision'].astype(float)
    #    BPBS = table['BPBS'].astype(float)
    #    columns = table['Columns'].astype(float)
    #    volt_2 = table['Voltage (V)'].astype(float) * table['Voltage (V)'].astype(float)

    #    x = []
    #    y = []
    #    for base in np.arange(0,100,10):
    #        Edac = base * pi * BPBS/columns * volt_2
    #        diff = abs((Ecell + Eadc + Eaccumulation + Edac) / table['fJ/MAC'].astype(float)-1).mean() # %
    #        x.append(base)
    #        y.append(diff * 100)
    #    minimal=(round(x[y.index(min(y))],2))
    #    plt.plot(x,y, linewidth=2, color=color[-1], label = lt, marker='o', markersize=8)
    #    plt.yscale('log')
    #    plt.xlabel('Edac/bit [fJ]', size=15)
    #    plt.ylabel('Modeling Mismatch percentage [\%]', size=15)
    #    plt.annotate(f'{minimal}fJ', (minimal, 0.8*min(y)), size=12, ha='center')

    #plt.legend(title=title_color, loc='upper right')

    plt.grid(visible=True, which='minor', axis='both', c='gainsboro', linestyle='--', zorder=-1.0)
    plt.grid(visible=True, which='major', axis='both', linestyle='-', zorder=-1.0)
    plt.rcParams['axes.axisbelow'] = True
    plt.tight_layout()
    plt.show()

def ml_op_breakdown():
    color = ['violet', 'orange', 'steelblue', 'palevioletred','crimson','forestgreen']
    table = get_values("14sOT8JJPU9aHC3rwLHjFkLKOu9jYdaPon6qBTLBuOeU", "wl_bd!A1:E5")

    plt.rcParams['text.usetex'] = True
    plt.rcParams['figure.figsize'] = (0.2,1)
    x = table['wl'].values.tolist()
    y1 = table['conv'].astype(float)
    y2 = table['batch'].astype(float)
    y3 = table['relu'].astype(float)
    y4 = table['pooling'].astype(float)
    ytotal = y1+y2+y3+y4
    ratio = y1 / ytotal * 100
    #n = table['Idx'].values.tolist()
    #x = [f'{abs(eval(xx))}\%' for xx in x]
    #y = [eval(yy) for yy in y]
    #y2 = [eval(yy) for yy in y2]
    #nx = np.array([nn for nn,_ in enumerate(n)])
    #n = [f'[{n}]' for n in n]
    
    plt.rcParams['figure.figsize'] = (20, 5)
    plt.grid(visible=True, which='major', axis='y')
    plt.bar(x, y1/ytotal * 100, width = 0.3, color=color[-1], label='Conv')  
    plt.bar(x, y2/ytotal * 100, width = 0.3, color=color[1], bottom=y1/ytotal * 100, label='Batch')  
    plt.bar(x, y3/ytotal * 100, width = 0.3, color=color[-2], bottom=(y1+y2)/ytotal * 100, label='Relu')  
    plt.bar(x, y4/ytotal * 100, width = 0.3, color=color[3], bottom=(y1+y2+y3)/ytotal * 100, label='Pooling')  
    plt.legend(fontsize=15,loc='upper center', ncol=4)
    #plt.yscale('log')
    #plt.xticks(ticks=nx, labels=n, fontsize=12)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim([0,120])
    plt.xlabel('Operation breakdown for benchmarking Workload', size=15)
    plt.ylabel('Ratio of operations (\%)', size=15)
    for i, nn in enumerate(x):
        plt.annotate(f'{round(ratio[i],1)}\%', (x[i], 100 * 1.02), color = color[-1], ha='center', size=15)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #aimc_topsw_topsmm2_plot()
    #aimc_topsw_topsmm2_1b_plot()
    #######aimc_vs_dimc_topsw_topsmm2_plot()##updated
    #######aimc_vs_dimc_topsw_topsmm2_1b_plot()##updated
    #aimc_vs_dimc_topsw_tops_1b_plot()
    ##########dimc_topsw_topsmm2_plot()##updated
    #aimc_validation_plot()
    #dimc_validation_plot()
    #######dimc_regression_cap()##updated
    #aimc_output_ratio_plot()
    #aimc_signal_margin_plot()
    #######dimc_cinv_fitting()##updated
    #######aimc_dac_fitting()##updated
    ml_op_breakdown()
    pass
