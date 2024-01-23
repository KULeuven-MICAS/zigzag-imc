class tID():
    def __init__(self, target_type, target_id):
        self.target_type = target_type
        self.target_id = target_id

class OptimizerTarget():
    def __init__(self, target_stage, target_object, target_modifier, target_range, target_parameters= None):
        self.target_stage = target_stage
        self.target_object = target_object
        self.target_modifier = target_modifier
        self.target_range = target_range
        self.target_parameters = target_parameters

def find_optimizer_target(tid_list, target_object=None):
    if tid_list == []:
        return target_object
    else:
        if tid_list[0].target_type not in ['list','dict']:
            new_target_object = next((v for k,v in target_object.__dict__.items() if k == tid_list[0].target_id), None)
        elif tid_list[0].target_type == 'list':
            new_target_object = target_object[tid_list[0].target_id]
        end_target = find_optimizer_target(tid_list[1:],new_target_object)
        return end_target
 
class BlackBoxOptimizer():
    def __init__(self):
        pass

    def init_optimizer(self):
        raise NotImplementedError()

    def sample(self):
        raise NotImplementedError()

    def get_cost(self):
        raise NotImplementedError()

    def update_optimizer(self):
        raise NotImplementedError()
    
    def runner(self):
        raise NotImplementedError()
