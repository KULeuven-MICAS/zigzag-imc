import importlib

from zigzag.classes.io.accelerator.parser import AcceleratorParser
from zigzag.classes.stages.Stage import Stage
from zigzag.classes.workload.dnn_workload import DNNWorkload
from zigzag.utils import pickle_deepcopy

import logging
logger = logging.getLogger(__name__)

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
    


## Description missing
class AcceleratorParserStage(Stage):
    def __init__(self, list_of_callables, *, accelerator, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        #if type(accelerator) == 'str':
        self.accelerator_parser = AcceleratorParser(accelerator)
        #else:
        #    self.accelerator_parser = None
        #    self.accelerator_model = accelerator

    def run(self):
        #if self.accelerator_parser is not None:
        self.accelerator_parser.run()
        self.accelerator = self.accelerator_parser.get_accelerator()

        if 'optimizer_target' in self.kwargs.keys():
            optimizer_target = self.kwargs['optimizer_target']
            if type(self).__name__ == optimizer_target.target_stage:
                tid_list = optimizer_target.target_object
                target_object = find_optimizer_target(tid_list, self)
                getattr(target_object, optimizer_target.target_modifier)(optimizer_target.target_parameters)
        #else:
        #    accelerator = self.accelerator_model

        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], accelerator=self.accelerator, **self.kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info


## @ingroup Stages
## Parse the input workload residing in workload_path.
## The "workload" dict is converted to a NetworkX graph.
## @param workload
## @param mapping
def parse_workload_from_path_or_from_module(workload, mapping):
    if isinstance(workload, str):  # load from path
        module = importlib.import_module(workload)
        workload = module.workload

    if isinstance(mapping, str):  # load from path
        module = importlib.import_module(mapping)
        mapping = module.mapping

    # make a copy here to prevent later it is being changed in the following stages
    workload_copy = pickle_deepcopy(workload)
    workload = DNNWorkload(workload_copy, mapping)
    #logger.info(
    #    f"Created workload graph with {workload.number_of_nodes()} nodes and {workload.number_of_edges()} edges.")

    return workload

## Description missing
class WorkloadParserStage(Stage):
    def __init__(self, list_of_callables, *, workload, mapping, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.workload = workload
        self.mapping = mapping

    def run(self):
        workload = parse_workload_from_path_or_from_module(self.workload, self.mapping)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], workload=workload, **self.kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

