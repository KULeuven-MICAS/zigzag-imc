class tID():
    def __init__(self, target_type, target_id):
        self.target_type = target_type
        self.target_id = target_id

class OptimizerTarget():
    def __init__(self, target_stage, target_object, target_modifier, target_parameters):
        self.target_stage = target_stage
        self.target_object = target_object
        self.target_modifier = target_modifier
        self.target_parameters = target_parameters


class BlackBoxOptimizer():
    def __init__(self):
        self.targets = []
        pass

    def init_optimizer(self):
        raise NotImplementedError()

    def sample(self):
        raise NotImplementedError()

    def get_cost(self):
        raise NotImplementedError()

    def update_optimizer(self):
        raise NotImplementedError()

