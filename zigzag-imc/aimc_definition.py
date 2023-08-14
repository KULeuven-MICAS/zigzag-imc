from dimc_basics import *
from aimc_basics import *

class AimcMacro:
    def __init__(self, activation_precision: int, storage_precision: int, input_precision: int, input_channel: int, output_channel: int, mapped_input_channel: int, mapped_output_channel: int, rows: int, cols: int, w_bw: int, array_delay: float, array_w_energy: float, array_area: float, unit_area: float, unit_delay: float, unit_cap: float, adc_resolution: int, dac_resolution: int, vdd: float):
        """
        Class for a single AIMC macro
        :param activation_precision:The bit precision of the input operand (unit: bit)
        :param storage_precision:   The bit precision of the operand stored in AIMC macro (unit: bit)
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
        :param adc_resolution:      The resolution of used ADC (unit: bit)
        :param dac_resolution:      The resolution of used DAC (unit: bit)
        :param vdd:                 The supply voltage (unit: V)
        NOTE: currently, no sparsity for input and weight is considered in the code.
        """
        if mapped_input_channel > input_channel:
            raise ValueError(f"AimcMacro: mapped_input_channel should not exceed input_channel. mapped_input_channel: {mapped_input_channel}, input_channel: {input_channel}")
        if mapped_output_channel > output_channel:
            raise ValueError(f"AimcMacro: mapped_output_channel should not exceed output_channel. mapped_output_channel: {mapped_output_channel}, output_channel: {output_channel}")
        if input_precision > activation_precision:
            raise ValueError(f"AimcMacro: input_precision cannot exceed activation_precision. input_precision: {input_precision}, activation_precision: {activation_precision}")
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
        self.array_delay = array_delay*0 # neglect array delay
        self.adc_resolution = adc_resolution
        self.dac_resolution = dac_resolution
        self.vdd = vdd
        self.storage_precision = storage_precision
        self.unit_cap = unit_cap

        """
        multiplier array for each output channel
        """
        self.mults = MultiplierArray(vdd=vdd,input_precision=int(cols/output_channel), number_of_multiplier=input_channel, number_of_mapped_multiplier=mapped_input_channel, unit_area=unit_area, unit_delay=unit_delay, unit_cap=unit_cap)
        """
        single ADC
        """
        self.adc = ADC(resolution=adc_resolution,input_channel=input_channel)
        """
        single DAC
        """
        self.dac = DAC(resolution=dac_resolution)
        """
        adder_tree for each output channel
        """
        adder_tree_input_precision = adc_resolution
        self.place_value_adder_tree = PlaceValueAdderTree(vdd=vdd, input_precision=adder_tree_input_precision, number_of_input=storage_precision, unit_area=unit_area, unit_delay=unit_delay, unit_cap=unit_cap)
        """
        accumulator for each output channel (only exist when input_precision < activation_precision)
        """
        self.accumulator_input_precision = storage_precision + adc_resolution +activation_precision # output precision from adder tree + required shift bits (small approximation for shift bits)
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
        self.area_dacs = self.input_channel * self.dac.calculate_area()
        self.area_adcs = self.output_channel * self.storage_precision * self.adc.calculate_area()
        self.area_adder_tree = self.output_channel * self.place_value_adder_tree.calculate_area()
        if self.input_precision < self.activation_precision:
            self.area_accumulator = self.output_channel * self.accumulator.calculate_area()
            self.area_regs_accumulator = self.output_channel * self.regs_accumulator.area
        else:
            self.area_accumulator = 0
            self.area_regs_accumulator = 0
        self.area = self.area_bank + self.area_mults + self.area_dacs + self.area_adcs + self.area_adder_tree + self.area_accumulator + self.area_regs_accumulator
        return self.area

    def calculate_computing_delay(self):
        """
        computing_delay: The computing access time (unit: ns) (=Tclk)
        """
        if self.input_precision == self.activation_precision:
            self.accumulator_delay = 0
        else:
            adder_1b_carry_delay = 2*UnitNand2(self.unit_area, self.unit_delay, self.unit_cap).calculate_delay()
            self.accumulator_delay = self.accumulator.calculate_delay_lsb() + adder_1b_carry_delay*(self.accumulator_input_precision-self.place_value_adder_tree.output_precision)

        self.computing_delay = self.array_delay + self.mults.calculate_delay() + self.dac.calculate_delay() + self.adc.calculate_delay() + self.place_value_adder_tree.calculate_delay() + self.accumulator_delay
        return self.computing_delay

    def calculate_computing_energy(self, input_toggle_rate, weight_sparsity):
        """
        computing_energy: The computing energy cost per clock cycle (each time it is triggered) (unit: fJ)
        """
        if input_toggle_rate < 0 or input_toggle_rate > 1:
            raise ValueError(f'input toggle rate should be in [0, 1]')
        if weight_sparsity < 0 or weight_sparsity > 1:
            raise ValueError(f'weight sparsity should be in [0, 1]')
        energy_bl = self.input_channel * self.unit_cap/2 * self.vdd**2 * self.storage_precision # per output channel
        self.energy_bank = (1-weight_sparsity) * self.mapped_output_channel * energy_bl # during peak performance evaluation, energy spent on bitlines accumulation
        self.energy_mults = input_toggle_rate * self.mapped_output_channel * self.mults.calculate_energy()
        self.energy_dacs = input_toggle_rate * self.mapped_input_channel * self.dac.calculate_energy(self.vdd)
        if self.mapped_input_channel == 0:
            self.energy_adcs = 0
            self.energy_adder_tree = 0
        else:
            self.energy_adcs = (1-weight_sparsity) * self.mapped_output_channel * self.storage_precision * self.adc.calculate_energy(self.vdd)
            self.energy_adder_tree = input_toggle_rate * self.mapped_output_channel * self.place_value_adder_tree.calculate_energy()

        if self.mapped_input_channel == 0 or self.input_precision == self.activation_precision:
            self.energy_accumulator = 0
        else:
            self.energy_accumulator = self.mapped_output_channel * self.accumulator.calculate_energy()

        if self.input_precision == self.activation_precision:
            self.energy_regs_accumulator = 0
        else:
            self.energy_regs_accumulator = self.mapped_output_channel * self.regs_accumulator.w_cost # all regs of output channels still consume energy

        self.computing_energy = self.energy_bank + self.energy_mults + self.energy_dacs + self.energy_adcs + self.energy_adder_tree + self.energy_accumulator + self.energy_regs_accumulator
        return self.computing_energy

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

        



if __name__ == "__main__":
    unit_area_28nm = 0.614
    unit_delay_28nm = 0.0394
    unit_cap_28nm = 0.7
    aimc = AimcMacro(activation_precision=8, storage_precision=8, input_precision=8, input_channel=16, output_channel=6, mapped_input_channel=16, mapped_output_channel=6, rows=32, cols=48, w_bw=48, array_delay=0.0669052, array_w_energy=328.423/64*81, array_area=0.00065545, unit_area=unit_area_28nm, unit_delay=unit_delay_28nm, unit_cap=unit_cap_28nm, adc_resolution=8, dac_resolution=8, vdd=0.9)
    breakpoint()
