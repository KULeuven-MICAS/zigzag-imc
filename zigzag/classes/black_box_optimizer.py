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

