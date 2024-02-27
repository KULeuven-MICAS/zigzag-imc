from typing import Generator, Callable, List, Tuple, Any

from zigzag.classes.stages.Stage import Stage
from zigzag.classes.cost_model.cost_model import CostModelEvaluation
from zigzag.classes.cost_model.cost_model_for_sram_imc import CostModelEvaluationForIMC
from zigzag.classes.hardware.architecture.accelerator import Accelerator
from zigzag.classes.mapping.spatial.spatial_mapping import SpatialMapping
from zigzag.classes.mapping.temporal.temporal_mapping import TemporalMapping
from zigzag.classes.workload.layer_node import LayerNode

import logging

logger = logging.getLogger(__name__)

##  Pipeline stage that calls a cost model to evaluate a mapping on a HW config.
class CostModelStage(Stage):

    ## The class constructor 
    # Initializes the cost model stage given main inputs
    # @param list_of_callables
    # @param accelerator
    # @param layer
    # @param spatial_mapping
    # @param temporal_mapping
    # @param access_same_data_considered_as_no_access
    # @param kwargs
    def __init__(
        self,
        list_of_callables: List[Callable],
        *,
        accelerator,
        layer,
        spatial_mapping,
        spatial_mapping_int,
        temporal_mapping,
        access_same_data_considered_as_no_access=True,
        **kwargs
    ):
        super().__init__(list_of_callables, **kwargs)
        (
            self.accelerator,
            self.layer,
            self.spatial_mapping,
            self.spatial_mapping_int,
            self.temporal_mapping,
            self.access_same_data_considered_as_no_access,
        ) = (
            accelerator,
            layer,
            spatial_mapping,
            spatial_mapping_int,
            temporal_mapping,
            access_same_data_considered_as_no_access,
        )
        self.extra_cells = kwargs['extra_cells']
        self.extra_rows = kwargs['extra_rows']

    ## Run the cost model stage by calling the internal zigzag cost model with the correct inputs.
    def run(self) -> Generator[Tuple[CostModelEvaluation, Any], None, None]:
        core_id = self.layer.core_allocation
        core = self.accelerator.get_core(core_id)
        operational_array = core.operational_array
        extra_cells = self.extra_cells[self.layer.id] if self.extra_cells != {} else 0
        extra_rows = self.extra_rows[self.layer.id] if self.extra_rows != {} else 0
        pe_type = getattr(operational_array, "pe_type", None) # return None if it does not exist
        if pe_type is not None and pe_type in ["in_sram_computing"]: # if pe_type exists and in the list
            self.cme = CostModelEvaluationForIMC(
                accelerator=self.accelerator,
                layer=self.layer,
                spatial_mapping=self.spatial_mapping,
                spatial_mapping_int=self.spatial_mapping_int,
                temporal_mapping=self.temporal_mapping,
                # the below parameter is optional
                access_same_data_considered_as_no_access=self.access_same_data_considered_as_no_access,
                extra_cells=extra_cells,
                extra_rows=extra_rows
            )
        else:
            self.cme = CostModelEvaluation(
                accelerator=self.accelerator,
                layer=self.layer,
                spatial_mapping=self.spatial_mapping,
                spatial_mapping_int=self.spatial_mapping_int,
                temporal_mapping=self.temporal_mapping,
                # the below parameter is optional
                access_same_data_considered_as_no_access=self.access_same_data_considered_as_no_access,
            )
        yield (self.cme, None)

    def is_leaf(self) -> bool:
        return True
