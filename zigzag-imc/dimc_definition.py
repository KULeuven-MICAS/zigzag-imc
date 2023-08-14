import math
from dimc_basics import *
import aimc_basics

class DimcMacro:
    def __init__(self, activation_precision: int, storage_precision: int, input_precision: int, input_channel: int, output_channel: int, mapped_input_channel: int, mapped_output_channel: int, rows: int, cols: int, w_bw: int, array_delay: float, array_w_energy: float, array_area: float, unit_area: float, unit_delay: float, unit_cap: float, vdd: float):
        """
        Class for a single DIMC macro
        :param activation_precision:The bit precision of the input operand (unit: bit)
        :param storage_precision:   The bit precision of the operand stored in DIMC macro (unit: bit)
        :param input_precision:     The bit precision of each input channel (unit: bit)
        :param input_channel:       The number of input channels
        :param output_channel:      The number of output channels
        :param mapped_input_channel:The number of activated input channels
        :param mapped_output_channel:The number of activated output channels
        :param rows:                The rows of memory cell array in macro
        :param cols:                The columns of memory cell array in macro (should equal to storage_precision*output_channel if no column mux) (memory bandwidth is supposed to equal to cols)
        :param array_delay:         The write delay for updating data (unit: ns) (from CACTI)
        :param array_w_energy:      The write energy of the cell array (unit: fJ) (from CACTI)
        :param array_area:          The area cost of the cell array (unit: mm2) (from CACTI)
        :param unit_area:           The area cost for a basic gate (unit: um2) (0.696 for 28nm)
        :param unit_delay:          The access delay for a basic gate (unit: ns) (0.0394 for 28nm)
        :param unit_cap:            The capacitance for a basic gate (unit: fF) (0.7 for 28nm)
        :param vdd:                 The supply voltage (unit: V)
        NOTE: currently, no sparsity for input and weight is considered in the code.
        """
        if mapped_input_channel > input_channel:
            raise ValueError(f"DimcMacro: mapped_input_channel should not exceed input_channel. mapped_input_channel: {mapped_input_channel}, input_channel: {input_channel}")
        if mapped_output_channel > output_channel:
            raise ValueError(f"DimcMacro: mapped_output_channel should not exceed output_channel. mapped_output_channel: {mapped_output_channel}, output_channel: {output_channel}")
        self.unit_area = unit_area
        self.unit_delay = unit_delay
        self.unit_cap = unit_cap
        unit_reg = UnitDff(unit_area, unit_delay, unit_cap)
        self.activation_precision = activation_precision
        self.input_precision = input_precision
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.mapped_input_channel = mapped_input_channel
        self.mapped_output_channel = mapped_output_channel
        self.w_bw = w_bw
        self.array_delay = array_delay * 0 # neglect array delay
        self.vdd = vdd
        
        """
        multiplier array for each output channel
        """
        self.mults = MultiplierArray(vdd=vdd,input_precision=int(cols/output_channel), number_of_multiplier=input_channel*input_precision, number_of_mapped_multiplier=mapped_input_channel*input_precision, unit_area=unit_area, unit_delay=unit_delay, unit_cap=unit_cap)
        """
        single adder_tree for each output channel
        """
        adder_tree_input_precision = storage_precision
        if int(math.log(input_channel * input_precision, 2)) != math.log(input_channel * input_precision, 2):
            raise ValueError("DimcMacro: input_precision of an adder tree must be in the power of 2! Not supported for other cases.")
        self.adder_tree = AdderTree(vdd=vdd, input_precision=adder_tree_input_precision, number_of_input=input_channel, number_of_mapped_input=mapped_input_channel, unit_area=unit_area, unit_delay=unit_delay, unit_cap=unit_cap)
        """
        single place value adder tree for each in/out channel when input_precision > 1
        """
        if input_precision > 1:
            self.pv_adders = aimc_basics.PlaceValueAdderTree(vdd=vdd,input_precision=self.adder_tree.output_precision,number_of_input=input_precision,unit_area=unit_area, unit_delay=unit_delay,unit_cap=unit_cap)
        """
        accumulator for each output channel
        """
        self.accumulator_input_precision = storage_precision + +int(math.log(input_channel, 2))+activation_precision # output precision from adder tree + required shift bits
        self.accumulator = Adder(vdd=vdd, input_precision=self.accumulator_input_precision, unit_area=unit_area, unit_delay=unit_delay, unit_cap=unit_cap)
        """
        memory instance (delay unit: ns, energy unit: fJ, area unit: mm2)
        unitbank: sram bank
        regs_accumulator: register files inside accumulator for each output channel
        """
        self.unitbank = MemoryInstance(name='unitbank', size=rows*cols, r_bw=cols, w_bw=w_bw, r_cost=0, w_cost=array_w_energy, area=array_area, r_port=1, w_port=1, rw_port=0, latency=0)
        self.regs_accumulator = MemoryInstance(name='regs_output', size=self.accumulator_input_precision, r_bw=self.accumulator_input_precision, w_bw=self.accumulator_input_precision, r_cost=0, w_cost=unit_reg.calculate_cap() * vdd**2 * self.accumulator_input_precision, area=unit_reg.calculate_area()*self.accumulator_input_precision, r_port=1, w_port=1, rw_port=0, latency=0)
    
    def calculate_area(self):
        """
        area: The area cost (unit: mm2)
        """
        self.area_bank = self.unitbank.area
        self.area_mults = self.output_channel * self.mults.calculate_area()
        self.area_adder_tree = self.output_channel * self.adder_tree.calculate_area() * self.input_precision
        if self.input_precision > 1:
            self.area_pv_adders = self.output_channel * self.pv_adders.calculate_area()
        else:
            self.area_pv_adders = 0
        if self.input_precision == self.activation_precision:
            self.area_accumulator = 0
        else:
            self.area_accumulator = self.output_channel * self.accumulator.calculate_area()
        self.area_regs_accumulator = self.output_channel * self.regs_accumulator.area
        area = self.area_bank + self.area_mults + self.area_pv_adders + self.area_adder_tree + self.area_accumulator + self.area_regs_accumulator
        return area
        
    def calculate_computing_delay(self):
        """
        computing_delay: The computing access time (unit: ns) (=Tclk)
        """
        if self.input_precision > 1:
            pv_last_adder = Adder(vdd=self.vdd, input_precision=(self.pv_adders.output_precision-self.pv_adders.input_precision), unit_area=self.unit_area, unit_delay=self.unit_delay, unit_cap=self.unit_cap)
            self.pv_adders_delay = (self.pv_adders.depth-1) * self.pv_adders.unit_adder.calculate_delay_lsb() + pv_last_adder.calculate_delay_msb()
        else:
            self.pv_adders_delay = 0
        if self.input_precision == self.activation_precision:
            self.accumulator_delay = 0
        elif self.input_precision > 1:
            adder_1b_carry_delay = 2*UnitNand2(self.unit_area, self.unit_delay, self.unit_cap).calculate_delay()
            self.accumulator_delay = self.accumulator.calculate_delay_lsb() + adder_1b_carry_delay*(self.accumulator_input_precision-self.pv_adders.output_precision)
        else:
            adder_1b_carry_delay = 2*UnitNand2(self.unit_area, self.unit_delay, self.unit_cap).calculate_delay()
            self.accumulator_delay = self.accumulator.calculate_delay_lsb() + adder_1b_carry_delay*(self.accumulator_input_precision-self.adder_tree.output_precision)

        computing_delay = self.array_delay + self.mults.calculate_delay() + self.adder_tree.calculate_delay() + self.accumulator_delay + self.pv_adders_delay
        return computing_delay
        
    def calculate_computing_energy(self, input_toggle_rate, weight_sparsity):
        """
        computing_energy: The computing energy cost per clock cycle (each time it is triggered) (unit: fJ)
        """
        if input_toggle_rate < 0 or input_toggle_rate > 1:
            raise ValueError(f'input toggle rate should be in [0, 1]')
        if weight_sparsity < 0 or weight_sparsity > 1:
            raise ValueError(f'weight sparsity should be in [0, 1]')
        self.energy_bank = self.unitbank.r_cost * 0 # set to 0, because this energy is mostly amortized during peak performance evaluation
        self.energy_mults = input_toggle_rate * self.mapped_output_channel * self.mults.calculate_energy()
        self.energy_adder_tree = input_toggle_rate * (1-weight_sparsity) * self.mapped_output_channel * self.adder_tree.calculate_energy() * self.input_precision
        if self.input_precision > 1:
            self.energy_pv_adders = self.mapped_output_channel * self.pv_adders.calculate_energy()
        else:
            self.energy_pv_adders = 0 
        if self.mapped_input_channel == 0 or self.input_precision == self.activation_precision:
            self.energy_accumulator = 0
        else:
            self.energy_accumulator = self.mapped_output_channel * self.accumulator.calculate_energy()
        self.energy_regs_accumulator = self.output_channel * self.regs_accumulator.w_cost * (self.input_precision != self.activation_precision) # all regs of output channels still consume energy
        computing_energy = self.energy_bank + self.energy_mults + self.energy_adder_tree + self.energy_accumulator + self.energy_regs_accumulator + self.energy_pv_adders
        return computing_energy
    
    def calculate_number_of_macs(self):
        """
        mac_number: The number of executed MACs per cycle when mapped (fully or partially)
        """
        mac_number = self.mapped_output_channel*self.mapped_input_channel*self.input_precision/self.activation_precision
        return mac_number
        
    def calculate_w_delay(self):
        """
        w_delay: The write access delay for updating data (unit: ns)
        """
        w_delay = self.array_delay
        return w_delay
    
    def calculate_w_energy(self):
        """
        w_energy: The write energy cost for updating data (unit: fJ)
        """
        w_energy = self.unitbank.w_cost
        return w_energy
    

# definition example
# unit_area_28nm = 0.696
# unit_delay_28nm = 0.0394
# unit_cap_28nm = 0.7
# vdd=0.9
# dimc_macro = DimcMacro(activation_precision=8, storage_precision=8, input_precision=2, input_channel=16, output_channel=6, mapped_input_channel=16, mapped_output_channel=6, rows=32, cols=48, w_bw=48, array_delay=0.0669052, array_w_energy=328.423/64*81, array_area=0.00065545, unit_area=unit_area_28nm, unit_delay=unit_delay_28nm, unit_cap=unit_cap_28nm, vdd=vdd)
# regs_input = MemoryInstance(name='regs_input', size=32*2, r_bw=32*2, w_bw=32*2, r_cost=0, w_cost=32*2*UnitDff(unit_area_28nm, unit_delay_28nm, unit_cap_28nm).calculate_cap() * vdd**2, area=32*2*UnitDff(unit_area_28nm, unit_delay_28nm, unit_cap_28nm).calculate_area(), r_port=1, w_port=1, rw_port=0, latency=1)
# regs_output = MemoryInstance(name='regs_output', size=8*6, r_bw=8*6, w_bw=8*6, r_cost=0, w_cost=8*6*UnitDff(unit_area_28nm, unit_delay_28nm, unit_cap_28nm).calculate_cap() * vdd**2, area=8*6*UnitDff(unit_area_28nm, unit_delay_28nm, unit_cap_28nm).calculate_area(), r_port=1, w_port=1, rw_port=0, latency=1)
