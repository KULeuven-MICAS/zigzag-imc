from typing import Dict, Set
import numpy as np
from classes.hardware.architecture.dimension import Dimension
from classes.hardware.architecture.operand_spatial_sharing import OperandSpatialSharing
from classes.hardware.architecture.operational_unit import OperationalUnit, Multiplier
from math import ceil
import math
from abc import ABC, abstractmethod
import pdb
import copy


class OperationalArray(ABC):
    def __init__(self, operational_unit: OperationalUnit, dimensions: Dict[str, int]):
        """
        This class captures multi-dimensional operational array size.

        :param operational_unit: an OperationalUnit object including precision and single operation energy, later we
                           can add idle energy also (e.g. for situations that one or two of the input operands is zero).

        :param dimensions: define the name and size of each multiplier array dimensions, e.g. {'D1': 3, 'D2': 5}.
        """
        self.unit = operational_unit
        self.total_unit_count = int(np.prod(list(dimensions.values()))) 
        base_dims = [Dimension(idx, name, size) for idx, (name, size) in enumerate(dimensions.items())]
        self.dimensions = base_dims
        self.dimension_sizes = [dim.size for dim in base_dims]
        self.nb_dimensions = len(base_dims)
        self.total_area = self.get_area()

    def __jsonrepr__(self):
        """
        JSON Representation of this class to save it to a json file.
        """
        return {"operational_unit": self.unit, "dimensions": self.dimensions}
   


    @abstractmethod
    def get_area(self):
        raise NotImplementedError('get_area() function not implemented for the operational array')

    @abstractmethod
    def get_MAC_cost(self):
        raise NotImplementedError('get_MAC_energy() function not implemented for the operational array')


class MultiplierArray(OperationalArray):
    def get_area(self):
        area = self.unit.area['MAC_unit'] * self.total_unit_count
        return area

    def get_MAC_cost(self, layer, mapping):
        return {'Digital MAC': layer.total_MAC_count * (self.unit.get_1b_multiplier_energy() + self.unit.get_adder_energy())}


class AIMCArray(OperationalArray):
    def __init__(self, operational_unit: OperationalUnit, dimensions: Dict[str, int]):
        super().__init__(operational_unit, dimensions)
        self.type = 'AIMC'

    def get_area(self):
        area = self.unit.area['cell'] * self.total_unit_count + \
            self.unit.area['ADC'] * self.dimension_sizes[0] + \
            self.unit.area['DAC'] * self.dimension_sizes[1]
        return area

    def get_MAC_cost(self, layer, mapping):
        spatial_mapping = copy.deepcopy(layer.user_spatial_mapping)
        spatial_mapping_tot = [v for k,v in spatial_mapping.items() if k in ['D1','D2','D3']]
        spatial_mapping = [v for k,v in spatial_mapping.items() if k in ['D1','D2']]
        const_operand = 'W' if 'W' in mapping.spatial_mapping.mapping_dict_origin.keys() else 'B'
        input_operand = 'I' if const_operand =='W' else 'A'
        spatial_mapping = [j for i in spatial_mapping for j in i]
        spatial_mapping_tot = [j for i in spatial_mapping_tot for j in i]
        
        FXu = np.prod([x[1] for x in spatial_mapping if x[0] == 'FX'])
        FYu = np.prod([x[1] for x in spatial_mapping if x[0] == 'FY'])
        OXu = np.prod([x[1] for x in spatial_mapping if x[0] == 'OX'])
        Cu = np.prod([x[1] for x in spatial_mapping if x[0] == 'C'])
        Ku = np.prod([x[1] for x in spatial_mapping if x[0] == 'K'])
        total_unrollings = np.prod([x[1] for x in spatial_mapping_tot])

        core_depth = self.unit.cost['CORE_ROWS']
        num_rows = Cu * (FYu * (OXu + FXu - 1)) # mapped input channel
        num_cols = OXu * Ku  # mapped output channel
        num_cores =total_unrollings / np.prod([x[1] for x in spatial_mapping])
        max_cols = self.dimension_sizes[0]*self.unit.input_precision[1] # physical cols (D1 in input file)
        total_MAC = layer.total_MAC_count / np.prod([x[1] for x in spatial_mapping])  # how many time macro is triggered (normalized to single macro)
        
        '''
        precharging_cell: energy cost (pJ) for precharing wordline and bitline (of the entire IMC array) only when core_depth > 1
        when core_depth=1, there is no read-out operation.
        '''
        precharging_cell = 0
        if core_depth > 1:
            precharging_cycles = mapping.unit_mem_data_movement[const_operand][0].data_elem_move_count.rd_out_to_low / mapping.ir_loop_size_per_level[const_operand][1]
            # only mapped part will consume energy. For the rest, it's believed it can be power gated.
            precharging_cell = \
                (self.unit.cost['wl_cap'] * (self.unit.cost['vdd']**2) + \
                (self.unit.cost['bl_cap'] * (self.unit.cost['vdd']**2) * (core_depth - 1)) ) * \
                (self.unit.cost['WEIGHT_BITCELL']) * \
                precharging_cycles


        """
        energy cost for multiplier array, during entire layer processing (pJ)
        """
        multiplication_energy = \
            self.unit.get_1b_multiplier_energy() * self.unit.cost['WEIGHT_BITCELL'] * \
            num_rows * self.dimension_sizes[0] * \
            (layer.operand_precision[input_operand] / self.unit.cost['DAC_RES']) * total_MAC # the number of times AIMC macro is triggered

        """
        energy for place-value adder tree (pJ)
        :param input_precision: input precision of an adder tree
        :param number_of_input_adder_tree: number of inputs
        :param adder_depth: adder tree's depth
        """
        input_precision = self.unit.cost['ADC_RES']
        number_of_input_adder_tree = self.unit.cost['WEIGHT_BITCELL']
        adder_depth = math.log(number_of_input_adder_tree, 2)
        num_of_1b_adder_in_adder_tree = input_precision * (number_of_input_adder_tree - 1) + number_of_input_adder_tree * (adder_depth-0.5)
        adder_energy_per_col_each_time = self.unit.get_1b_adder_energy()* num_of_1b_adder_in_adder_tree
        adder_energy_each_macro = num_cols * adder_energy_per_col_each_time
        adder_tree_energy = adder_energy_each_macro * (layer.operand_precision[input_operand] / self.unit.cost['DAC_RES']) * total_MAC

        """
        accumulator energy (pJ) (including regs and accumulators)
        Note: no accumulator if DAC_RES equals to input precision
        """
        accumulator_energy = 0
        if self.unit.cost['DAC_RES'] < layer.operand_precision[input_operand]:
            accumulator_precision  = layer.operand_precision[input_operand] + layer.operand_precision[const_operand] + self.unit.cost['ADC_RES']
            accumulator_adder_energy_each_time = num_cols * self.unit.get_1b_adder_energy() * accumulator_precision
            accumulator_regs_energy_each_time  = num_cols * self.unit.get_1b_reg_energy() * accumulator_precision
            accumulator_energy = (accumulator_adder_energy_each_time+accumulator_regs_energy_each_time) * (layer.operand_precision[input_operand] / self.unit.cost['DAC_RES']) * total_MAC
        
        """
        energy of cell array (bitline accumulation)
        """
        cell = \
            (self.unit.cost['bl_cap'] * (self.unit.cost['bl_v']**2) * num_cols * self.dimension_sizes[1]) * \
            (self.unit.cost['WEIGHT_BITCELL']) * (layer.operand_precision[input_operand] / self.unit.cost['DAC_RES']) * total_MAC
    
        """
        DAC/ADC energy
        """
        DAC_energy = num_rows * 50e-3 * self.unit.cost['DAC_RES'] * (layer.operand_precision[input_operand]/self.unit.cost['DAC_RES']) * (np.power(self.unit.cost['vdd'],2)) * (layer.operand_precision[const_operand] / self.unit.cost['WEIGHT_BITCELL']) * total_MAC
        ADC_energy = num_cols * (100e-3 * self.unit.cost['ADC_RES'] + 1e-6 * (np.power(4, self.unit.cost['ADC_RES']))) * (np.power(self.unit.cost['vdd'],  2)) * (layer.operand_precision[input_operand] / self.unit.cost['DAC_RES']) * self.unit.cost['WEIGHT_BITCELL'] * total_MAC
        # 2023/6/8: previously imc_write_cell represents nb of weight operands to be updated in IMC array and it is used in cost model. Now it's no use. Ready to be removed.
        imc_write_cells = num_rows * num_cores * self.dimension_sizes[0]
    
        mac_cost =  {
                'precharging_cell' : precharging_cell,
                'cell' : cell,
                'multiplication_energy' : multiplication_energy,
                'adder_tree_energy'     : adder_tree_energy,
                'DAC': DAC_energy,
                'ADC': ADC_energy,
                'accumulator_energy' : accumulator_energy}
        return mac_cost, imc_write_cells


class DIMCArray(OperationalArray):
    def __init__(self, operational_unit: OperationalUnit, dimensions: Dict[str, int]):
        super().__init__(operational_unit, dimensions)
        self.type = 'DIMC'


    def get_area(self):
        area = self.unit.area['cell'] * self.total_unit_count + \
            self.unit.area['adder'] * self.dimension_sizes[0]/self.unit.cost['WEIGHT_BITCELL'] + \
            self.unit.area['multiplier'] * self.dimension_sizes[1] * self.dimension_sizes[0]
        return area

    def __jsonrepr__(self):
        """
        JSON Representation of this class to save it to a json file.
        """
        return {"operational_unit": self.unit, "dimensions": self.dimensions, 'accumulation_cycles':self.accumulation_cycles}
 
    def get_MAC_cost(self, layer, mapping):
        spatial_mapping = copy.deepcopy(layer.user_spatial_mapping)
        spatial_mapping_tot = [v for k,v in spatial_mapping.items() if k in ['D1','D2','D3']]
        spatial_mapping = [v for k,v in spatial_mapping.items() if k in ['D1','D2']]
        const_operand = 'W' if 'W' in mapping.spatial_mapping.mapping_dict_origin.keys() else 'B'
        input_operand = 'I' if const_operand =='W' else 'A'
        spatial_mapping = [j for i in spatial_mapping for j in i]
        spatial_mapping_tot = [j for i in spatial_mapping_tot for j in i]
        
        FXu = np.prod([x[1] for x in spatial_mapping if x[0] == 'FX'])
        FYu = np.prod([x[1] for x in spatial_mapping if x[0] == 'FY'])
        OXu = np.prod([x[1] for x in spatial_mapping if x[0] == 'OX'])
        OYu = np.prod([x[1] for x in spatial_mapping if x[0] == 'OY'])
        Cu = np.prod([x[1] for x in spatial_mapping if x[0] == 'C'])
        Ku = np.prod([x[1] for x in spatial_mapping if x[0] == 'K'])
        total_unrollings = np.prod([x[1] for x in spatial_mapping_tot])

        core_depth = self.unit.cost['CORE_ROWS']
        max_cols = self.dimension_sizes[0]*self.unit.input_precision[1] # physical cols (D1 in input file)

        num_rows = Cu * (FYu * (OXu + FXu - 1)) # mapped input channel
        num_rows_adder_tree = Cu * (FYu * (FXu)) # mapped input channel
        num_rows = int(num_rows + 0.5) # round to upper int
        num_cols = OXu * Ku # mapped output channel
        num_cores =total_unrollings / np.prod([x[1] for x in spatial_mapping])
        total_MAC = layer.total_MAC_count / np.prod([x[1] for x in spatial_mapping])  # how many time macro is triggered (normalized to single macro)

        '''
        precharging_cell: energy cost (pJ) for precharing wordline and bitline (of the entire IMC array) only when core_depth > 1
        when core_depth=1, there is no read-out operation.
        :param self.dimension[0].size: maximum_output_channel
        :param self.dimension[1].size: maximum_input_channel
        update 2023/03/18: voltage on wl is vdd/2 -> voltage on wl is vdd.
        update 2023/03/18: energy also multiply with #maximum_output_channel
        '''
        precharging_cell = 0
        if core_depth > 1:
            precharging_cycles = mapping.unit_mem_data_movement[const_operand][0].data_elem_move_count.rd_out_to_low / mapping.ir_loop_size_per_level[const_operand][1]
            # only mapped part will consume energy. For the rest, it's believed it can be power gated.
            precharging_cell = \
                (self.unit.cost['wl_cap'] * (self.unit.cost['vdd']**2) + \
                (self.unit.cost['bl_cap'] * (self.unit.cost['vdd']**2) * (core_depth - 1)) ) * \
                (self.unit.cost['WEIGHT_BITCELL']) * \
                precharging_cycles

        """
        energy cost for multiplier array, during entire layer processing (pJ)
        update 2023/03/18: update calculation formula for multiplication_energy
        update 2023/03/18: energy also multiply with #maximum_output_channel
        """
        if self.unit.imc_type == 'IMC':
            # only mapped part will consume energy.
            multiplication_energy = \
                self.unit.get_1b_multiplier_energy() * self.unit.cost['WEIGHT_BITCELL'] * self.unit.cost['INPUT_BITS_PER_CYCLE'] * \
                num_rows* self.dimension_sizes[0] * \
                (layer.operand_precision[input_operand] / self.unit.cost['INPUT_BITS_PER_CYCLE']) * total_MAC # the number of times DIMC macro is triggered

            """
            energy for adder tree (pJ)
            :param input_precision: input precision of an adder tree
            :param mapped_inputs:   the number of activate inputs for an adder tree (#mapped_input_channel)
            update 2023/03/18: feature added. Now the energy will decrease when inputs are not fully mapped.
            update 2023/03/18: feature added. Now the energy will decrease when columns (output channel) are not fully mapped.
            update 2023/04/23: add nb_1b_adders_in_pv_adders to support 'INPUT_BITS_PER_CYCLE'>1
            update 2023/06/30: change the calculation of nb_1b_adders_in_pv_adders, so pv_adders is put after the normal adders, instead of in the front as previous.
            """
            adder_tree_energy = 0
            accumulator_energy = 0
            self.accumulation_cycles = 0
            input_precision = layer.operand_precision[const_operand] # input precision for an adder tree
            adders_output_precision = input_precision + int(math.log2(self.dimensions[1].size))
            if self.unit.cost['INPUT_BITS_PER_CYCLE'] > 1:
                nb_1b_adders_in_pv_adders = adders_output_precision * (self.unit.cost['INPUT_BITS_PER_CYCLE']-1) + self.unit.cost['INPUT_BITS_PER_CYCLE']*(math.log2(self.unit.cost['INPUT_BITS_PER_CYCLE'])-0.5)
                #nb_1b_adders_in_pv_adders = layer.operand_precision[const_operand]*(self.unit.cost['INPUT_BITS_PER_CYCLE']-1) + self.unit.cost['INPUT_BITS_PER_CYCLE']*(math.log2(self.unit.cost['INPUT_BITS_PER_CYCLE'])-0.5)
                #nb_1b_adders_in_pv_adders = nb_1b_adders_in_pv_adders * num_rows_adder_tree
            else:
                nb_1b_adders_in_pv_adders = 0
            number_of_input = self.dimensions[1].size  # maximum number of inputs for an adder tree
            mapped_inputs = num_rows_adder_tree  # number of used inputs for an adder tree
            adder_tree_depth = int( math.log2(number_of_input) )
            number_of_1b_adder = number_of_input*(input_precision+1)-(input_precision+adder_tree_depth+1) # whole number of 1b adder in an adder tree

            if mapped_inputs >= 1:
                if(mapped_inputs >= number_of_input):
                    """
                    :param fully_activated_number_of_1b_adder: fully activated 1b adder, probably will produce a carry
                    :param half_activated_number_of_1b_adder: only 1 input is activate and the other port is 0, so carry path is not in activate state.
                    """
                    fully_activated_number_of_1b_adder = number_of_1b_adder
                    half_activated_number_of_1b_adder = 0
                else:
                    """
                    find out fully_activated_number_of_1b_adder and half_activated_number_of_1b_adder when inputs are not fully mapped
                    :param left_input: the number of inputs waiting for processing
                    :param baseline: serves as references for left_input
                    .algorithm: iteratively check if left_input is bigger or smaller than baseline, which will /2 each time, until left_input == 1
                    """
                    fully_activated_number_of_1b_adder = 0
                    half_activated_number_of_1b_adder = 0
                    left_input = mapped_inputs
                    baseline = number_of_input
                    while(left_input != 0):
                        baseline = baseline/2
                        activated_depth = int( math.log(baseline, 2) )
                        if(left_input <= 1 and baseline == 1):
                            fully_activated_number_of_1b_adder += 0
                            half_activated_number_of_1b_adder += input_precision
                            left_input = 0
                        elif(left_input > baseline):
                            fully_activated_number_of_1b_adder += baseline*(input_precision+1)-(input_precision+activated_depth+1) + (input_precision + activated_depth)
                            half_activated_number_of_1b_adder += 0
                            left_input = left_input - baseline
                        elif(left_input < baseline):
                            half_activated_number_of_1b_adder += input_precision + activated_depth
                        else: # left_input == baseline
                            fully_activated_number_of_1b_adder += baseline*(input_precision+1)-(input_precision+activated_depth+1)
                            half_activated_number_of_1b_adder += input_precision + activated_depth
                            left_input = left_input - baseline
                
                fully_activated_number_of_1b_adder *= self.unit.cost['INPUT_BITS_PER_CYCLE']
                half_activated_number_of_1b_adder *= self.unit.cost['INPUT_BITS_PER_CYCLE']
                fully_activated_number_of_1b_adder += nb_1b_adders_in_pv_adders
                adder_energy_per_col_each_time = fully_activated_number_of_1b_adder * self.unit.get_1b_adder_energy() + half_activated_number_of_1b_adder * self.unit.get_1b_adder_energy_half_activated()

                adder_energy_each_macro = num_cols * adder_energy_per_col_each_time
                            
                adder_tree_energy += adder_energy_each_macro * (layer.operand_precision[input_operand] / self.unit.cost['INPUT_BITS_PER_CYCLE']) * total_MAC

                """
                accumulator energy (pJ) (including regs and accumulators)
                """
                # only mapped part will consume energy. For the rest, it's believed it can be power gated.
                if self.unit.cost['INPUT_BITS_PER_CYCLE'] == layer.operand_precision[input_operand]:
                    accumulator_energy += 0
                else:
                    accumulator_precision = layer.operand_precision[const_operand]+layer.operand_precision[input_operand]+int(math.log(self.dimensions[1].size,2))
                    accumulator_adder_energy_each_time = num_cols * self.unit.get_1b_adder_energy() * accumulator_precision
                    accumulator_regs_energy_each_time  = num_cols * self.unit.get_1b_reg_energy() * accumulator_precision
                    accumulator_energy += (accumulator_adder_energy_each_time+accumulator_regs_energy_each_time) * (layer.operand_precision[input_operand] / self.unit.cost['INPUT_BITS_PER_CYCLE']) * total_MAC
                self.accumulation_cycles += total_MAC

        # 2023/6/8: previously imc_write_cell represents nb of weight operands to be updated in IMC array and it is used in cost model. Now it's no use. Ready to be removed.
        imc_write_cells = num_rows * num_cores * self.dimension_sizes[0]
        mac_cost =  {'precharging_cell' : precharging_cell,\
                'multiplication_energy' : multiplication_energy,
                'adder_tree_energy'     : adder_tree_energy,
                'accumulator_energy'   : accumulator_energy }
        return mac_cost, imc_write_cells



def multiplier_array_example1():
    '''Multiplier array variables'''
    multiplier_input_precision = [8, 8]
    multiplier_energy = 0.5
    multiplier_area = 0.1
    dimensions = {'D1': 14, 'D2': 3, 'D3': 4}
    operand_spatial_sharing = {'I1': {(1, 0, 0)},
                       'O': {(0, 1, 0)},
                       'I2': {(0, 0, 1), (1, 1, 0)}}
    multiplier = Multiplier(multiplier_input_precision, multiplier_energy, multiplier_area)
    multiplier_array = MultiplierArray(multiplier, dimensions, operand_spatial_sharing)

    return multiplier_array


def multiplier_array_example2():
    '''Multiplier array variables'''
    multiplier_input_precision = [8, 8]
    multiplier_energy = 0.5
    multiplier_area = 0.1
    dimensions = {'D1': 14, 'D2': 12}
    operand_spatial_sharing = {'I1': {(1, 0)},
                             'O': {(0, 1)},
                             'I2': {(1, 1)}}
    multiplier = Multiplier(multiplier_input_precision, multiplier_energy, multiplier_area)
    multiplier_array = MultiplierArray(multiplier, dimensions, operand_spatial_sharing)

    return multiplier_array


if __name__ == "__main__":
    multiplier_array = multiplier_array_example1()
    for os in multiplier_array.operand_spatial_sharing:
        print(f'{os}\tdirection: {os.direction} operand: {os.operand} instances: {os.instances}')
