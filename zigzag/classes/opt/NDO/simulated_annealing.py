import numpy as np

class SimulatedAnnealingOptimizer():
    def __init__(self, init_temperature, cooling_factor, iterations, input_parameters):
        self.temp = init_temperature
        self.alfa = cooling_factor
        self.iterations = iterations
        self.input_parameters = input_parameters


        x = np.linspace(32,256,256-32+1)
        self.input_samples = []
        self.output_samples = []

        # Sample new point
        optimizer_target = self.sample_parameter(init=True)
        self.kwargs['optimizer_target'] = optimizer_target 
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:],**self.kwargs)
        cme_list = []
        for cme, extra_info in sub_stage.run():
            cme_list.append(cme)
            yield cme, extra_info
        self.best_cost = - self.get_cost(cme_list)
        self.output_samples.append(self.best_cost)
        current_dim = [optimizer_target.target_parameters['D1'], optimizer_target.target_parameters['D2']]

        print(f'Cost {self.best_cost:.2e}')
        for i in range(self.optimizer_params['iterations']):
            # Sample new point from distribution g(_) ==> random perturbation
            optimizer_target = self.sample_parameter(init=False, current_dim=current_dim)
            self.kwargs['optimizer_target'] = optimizer_target 
            sub_stage = self.list_of_callables[0](self.list_of_callables[1:],**self.kwargs)
            cme_list = []
            for cme, extra_info in sub_stage.run():
                cme_list.append(cme)
                yield cme, extra_info
            new_cost = - self.get_cost(cme_list)
            print(f'Cost {new_cost:.2e}')
            self.output_samples.append(new_cost)
            delta_cost = new_cost - self.best_cost
            print(f'{delta_cost:.2e}', np.log10(delta_cost))
            # Apply acceptance criterion
            if (delta_cost < 0):
                #accept
                cprint('Accepted because negative', 'green')
                self.best_cost = new_cost
                current_dim = [optimizer_target.target_parameters['D1'], optimizer_target.target_parameters['D2']]
            else:
                r = np.random.random()
                cprint(f'Accept? {r:.2e} < {np.exp(- np.log(delta_cost)/10 / self.temp):.2e}','yellow')
                if r < np.exp(- np.log(delta_cost)/10 / self.temp):
                    #accept
                    cprint('Accepted','green')
                    self.best_cost = new_cost
                    current_dim = [optimizer_target.target_parameters['D1'], optimizer_target.target_parameters['D2']]
                else:
                    # not accept
                    cprint('Refused','red')
                    pass

            print()
            # update temperature
            # Geometric cooling adopted (Also logarithmic, exponential available)
            #self.temp = np.power(self.alfa,i) * self.optimizer_params['init_temperature']
            # Logarithmic cooling
            self.temp = (self.alfa * self.optimizer_params['init_temperature']) / (np.log(2+i))
            print(f'Iteration {i}: Temperature {self.temp:.2e}')
            
        breakpoint()

    def sample_parameter(self, init=False, current_dim=None):
        while(1):
            if init:
                dims = np.random.randint(32, 256, size=2)
            else:
                while(1):
                    dims = [x for x in current_dim]
                    dims[0] = current_dim[0] + int( np.random.normal(scale=self.temp)*256 )
                    dims[1] = current_dim[1] + int( np.random.normal(scale=self.temp)*256 )
                    if dims[0] > 0 and dims[1] > 0:
                        break

            dimensions = {'D1':int(dims[0]),'D2':int(dims[1]),'D3':1}
            group_depth = 1
            imc_array = self.imc_array_dut(dimensions, group_depth)
            if imc_array.total_area <= self.optimizer_params['area_budget']:
                break

        self.input_samples.append(dims)
        print(f'Sampled array dimension: {dims}')
        optimizer_target = OptimizerTarget(target_stage = 'AcceleratorParserStage',
                target_object = [tID('object','accelerator'), tID('obj','cores'), tID('list',0), tID('obj','operational_array')],
                target_modifier = 'set_array_dim',
                target_parameters = dimensions)
        return optimizer_target


