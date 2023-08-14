from typing import List, Dict
import numpy as np


class OperationalUnit:
    def __init__(self, input_precision: List[int], output_precision: int, unit_cost: Dict[str, float], unit_area: Dict[str, float], technology: int):
        """
        General class for a unit that performs a certain operation. For example: a multiplier unit.

        :param input_precision: The bit precision of the operation inputs.
        :param output_precision: The bit precision of the operation outputs.
        :param unit_cost: The energy cost of performing a single operation.
        :param unit_area: The area of a single operational unit.
        """
        self.input_precision = input_precision
        self.output_precision = output_precision
        self.precision = input_precision + [output_precision]
        self.cost = unit_cost
        self.technology = technology
        self.area = unit_area
        if 'nand2_per_adder' in unit_cost.keys():
            self.nand2_per_adder = unit_cost['nand2_per_adder']
        else:
            self.nand2_per_adder = 3
        if 'xor2_per_adder' in unit_cost.keys():
            self.xor2_per_adder = unit_cost['xor2_per_adder']
        else:
            self.xor2_per_adder = 2
        if 'mac_clock_domain' in unit_cost.keys():
            self.cost['mac_clock_domain'] = unit_cost['mac_clock_domain']
        else:
            self.cost['mac_clock_domain'] = 1



    def __jsonrepr__(self):
        """
        JSON Representation of this class to save it to a json file.
        """
        return self.__dict__

    def get_cap_gate_nand2(self):
        """
        get the capacitance value for single nand2 gate
        (comment: formula changes a bit to get 0.7 @ 28nm)
        """
        return (self.technology * 0.019 + 0.168) * 1e-3 # pF
        # return (self.technology * 0.019 + 0.228)*1e-3 # pF

    def get_cap_gate_xor2(self):
        """
        get the capacitance value for single xor2 gate
        """
        return self.get_cap_gate_nand2()*1.5 # 1.5x bigger than nand2

    def get_cap_inv(self):
        """
        get the capacitance value of wordline and bitline of single SRAM cell
        """
        return self.get_cap_gate_nand2()/2

    def get_1b_adder_energy(self):
        """
        energy value for 1-b full adder (pJ)
        """
        nand2_per_adder = self.nand2_per_adder
        xor2_per_adder = self.xor2_per_adder
        return (self.cost['vdd']**2) * (self.get_cap_gate_nand2()*nand2_per_adder + self.get_cap_gate_xor2() * xor2_per_adder)

    def get_1b_adder_energy_half_activated(self):
        """
        energy value for 1-b full adder when one input is always 0 (carry path inactivate) (pJ)
        """
        xor2_per_adder = self.xor2_per_adder
        return (self.cost['vdd']**2) * (self.get_cap_gate_xor2() * xor2_per_adder)

   # def get_adder_energy(self):
   #     gate_per_adder = self.gate_per_adder
   #     return  self.get_cap_gate() * (self.cost['vdd']**2) * gate_per_adder

    def get_1b_multiplier_energy(self):
        """
        energy value for 1-b multiplier (pJ)
        Comment: .during computation, weight stays the same, so 0.5
        """
        cap_nor2 = self.get_cap_gate_nand2()
        return  0.5*cap_nor2 * (self.cost['vdd']**2) 

    def get_1b_reg_energy(self):
        """
        energy for 1b register (pJ)
        for the use of accumulator
        """
        cap_dff = 3 * self.get_cap_gate_nand2()
        return cap_dff * (self.cost['vdd']**2)



class Multiplier(OperationalUnit):
    def __init__(self, input_precision: List[int], energy_cost: Dict[str, float], area: Dict[str, float]):
        """
        Initialize the Multiplier object.

        :param input_precision: The bit precision of the multiplication inputs.
        :param output_precision: The bit precision of the multiplication outputs.
        :param energy_cost: The energy cost of performing a single multiplication.
        :param area: The area of a single multiplier.
        """
        output_precision = sum(input_precision)
        super().__init__(input_precision, output_precision, energy_cost, area)


class AIMC(OperationalUnit):
    def __init__(self, input_precision: List[int], energy_cost: Dict[str, float], area: Dict[str, float], technology: int):
        """
        Class for a single AIMC macro
        :param input_precision: The bit precision of AIMC [list: activation precision, weight precision].
        :param energy_cost: The dict containing how many rows per input channel, input precision per input channel, storage precision
        :param area: the area cost for cell, adder and multiplier.
        :param technology: technology node (e.g., 22, 28)
        """
        output_precision = sum(input_precision)
        super().__init__(input_precision, output_precision, energy_cost, area, technology)

        if 'wl_cap' not in self.cost:
            self.cost['wl_cap'] = self.get_cap_inv()
        if 'bl_cap' not in self.cost:
            self.cost['bl_cap'] = self.get_cap_inv()
        if 'wl_v' not in self.cost:
            self.cost['wl_v'] = self.cost['vdd']
        if 'bl_v' not in self.cost:
            self.cost['bl_v'] = self.cost['vdd']

        if 'WEIGHT_BITCELL' not in self.cost:
            self.cost['WEIGHT_BITCELL'] = self.input_precision[1] # default: storage precisoin = weight precision

class DIMC(OperationalUnit):
    def __init__(self, input_precision: List[int], energy_cost: Dict[str, float], area: Dict[str, float], technology: int):
        """
        Class for a single DIMC macro
        :param input_precision: The bit precision of DIMC [list: activation precision, weight precision].
        :param energy_cost: The dict containing how many rows per input channel, input precision per input channel, storage precision
        :param area: the area cost for cell, adder and multiplier.
        :param technology: technology node (e.g., 22, 28)
        """
        output_precision = sum(input_precision)
        super().__init__(input_precision, output_precision, energy_cost, area, technology)
        self.imc_type = 'IMC'
        if 'wl_cap' not in self.cost:
            self.cost['wl_cap'] = self.get_cap_inv() # Cwl/cell = 1 inv
        if 'bl_cap' not in self.cost:
            self.cost['bl_cap'] = self.get_cap_inv() # Cbl/cell = 1 inv
        if 'WEIGHT_BITCELL' not in self.cost:
            self.cost['WEIGHT_BITCELL'] = self.input_precision[1] # default: storage precisoin = weight precision

