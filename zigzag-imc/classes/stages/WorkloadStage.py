import networkx as nx

from classes.stages.Stage import Stage
from classes.workload.dnn_workload import DNNWorkload
import classes.io.input_config as inputs
from classes.workload.dummy_node import DummyNode


class WorkloadStage(Stage):
    """
    Class that iterates through the nodes in a given workload graph.
    """
    def __init__(self, list_of_callables, *, workload, **kwargs):
        """
        Initialization of self.workload.
        :param main_inputs: MainInputs, NOT copied
        """
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload

    def run(self):
        for ii_layer, layer in enumerate(nx.topological_sort(self.workload)):
            if type(layer) == DummyNode:
                continue  # skip the DummyNodes
            #if ii_layer != 119:
            #    continue
            kwargs = self.kwargs.copy()
            kwargs['layer'] = layer
            kwargs['workload_mapping'] = self.workload.get_node_with_id(layer.id)
            sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **kwargs)
            for cme, extra_info in sub_stage.run():
                yield cme, (layer, extra_info)
