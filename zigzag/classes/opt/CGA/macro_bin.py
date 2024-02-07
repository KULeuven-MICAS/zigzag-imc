import numpy as np
from ortools.sat.python import cp_model
from loguru import logger

class MacroBin():
    def __init__(self, height, number_of_macros):
        self.height = height
        self.D3 = number_of_macros

    def pack_macrobin(self, layer_list, fsi, zsl, ol, nki):
        model = cp_model.CpModel()
        solver = cp_model.CpSolver()

        n_superitems, n_items = fsi.shape
        n_network_layers, _ = nki.shape

        # Variable definition
        tb, al, ulb = dict(), dict(), dict()
        for b in range(self.D3):
            tb[b] = model.NewBoolVar(f"t_{b}")
        for l in range(len(layer_list)):
            #al[l] = model.NewBoolVar(f'a_{l}')
            for b in range(self.D3):
                ulb[l,b] = model.NewBoolVar(f'u_{l}_{b}')
        
        # Constraints
        # Ensure that every item is included in exactly one layer
        for i in range(n_items):
            model.Add(
                cp_model.LinearExpr.Sum(
                    fsi[s, i] * zsl[s, l] * ulb[l, b] for s in range(n_superitems) for l in range(len(layer_list)) for b in range(self.D3)
                )
                == 1
            )

        for k in range(n_network_layers):
            for b in range(self.D3):
                model.Add(
                    cp_model.LinearExpr.Sum(
                        nki[k, i] * fsi[s, i] * zsl[s, l] * ulb[l, b] for i in range(n_items) for s in range(n_superitems) for l in range(len(layer_list))
                    )
                    <= 1
                )


        # Ensure that height of layer combination does not exceed bin height
        for b in range(self.D3):
            model.Add(
                cp_model.LinearExpr.Sum(
                    ol[l] * ulb[l, b] for l in range(len(layer_list))
                )
                <= self.height
            )
        for l in range(len(layer_list)):
            for b in range(self.D3):
                model.Add(ulb[l, b] <= tb[b])
                #model.Add(ulb[l, b] <= al[l])

        obj = cp_model.LinearExpr.Sum(tb[b] for b in range(self.D3))
        model.Minimize(obj)

        # Set solver parameters
        num_workers = 4
        solver.parameters.num_search_workers = num_workers
        solver.parameters.log_search_progress = False
        solver.parameters.search_branching = cp_model.FIXED_SEARCH

        # Solve
        status = solver.Solve(model)
        bin_dict = {} 

        if solver.StatusName(status) == 'OPTIMAL' or solver.StatusName(status) == 'FEASIBLE':
            for b in range(self.D3):
                bin_dict[b] = []
                for l in range(len(layer_list)):
                    if solver.Value(ulb[l,b]) == 1:
                        bin_dict[b].append(l)
        return bin_dict, solver.StatusName(status)

        

