from dimc_basics import Adder
import math

class ADC:
    """
    Class for a single ADC.
    :param resolution:      ADC resolution
    :param vdd:             The supply voltage (unit: V)
    :param input_channel:   The number of input channels on ADC input node
    """
    def __init__(self, resolution:int, input_channel: int):
        if resolution <= 0:
            raise ValueError(f"ADC: resolution must be a positive integer. Now: {resolution}")
        self.resolution = resolution
        self.input_channel = input_channel
    def calculate_area(self):
        """
        self.area: the area cost of a single ADC (unit: mm2)
        """
        if self.resolution == 1:
            self.area = 0
        else:
            k1 = -0.0369
            k2 = 1.206
            self.area = 10**(k1*self.resolution+k2) * 2**self.resolution * (10**-6) # unit: mm2
        #elif self.resolution < 12:
        #    k1 = -0.0369
        #    k2 = 1.206
        #    self.area = 10**(k1*self.resolution+k2) * 2**self.resolution * (10**-6) # unit: mm2
        #else: # resolution >=12
        #    # equation below has not be validated. From: https://ieeexplore.ieee.org/abstract/document/9218657
        #    self.area = 5 * 10**-7 * 2**self.resolution # unit: mm2
        return self.area

    def calculate_delay(self):
        """
        self.delay: the delay cost of a single ADC (unit: ns)
        """
        k3 = 0.00653 # ns
        k4 = 0.640   # ns
        self.delay = self.resolution * (k3 * self.input_channel + k4) # unit: ns
        return self.delay

    def calculate_energy(self, vdd:float):
        """
        self.energy: the energy cost of a single ADC (unit: fJ)
        """
        #if self.resolution == 1:
        #    self.energy = 0
        #else:
        k5 = 100 # fF
        k6 = 0.001 # fF
        self.energy = (k5 * self.resolution + k6 * 4**self.resolution) * vdd**2 # unit: fJ
        return self.energy

class DAC:
    """
    Class for a single DAC
    :param resolution:  DAC resolution
    :param vdd:         the supply voltage (unit: V)
    """
    def __init__(self, resolution: int):
        if resolution <= 0:
            raise ValueError(f"DAC: resolution must be a positive integer. Now: {resolution}")
        self.resolution = resolution
    
    def calculate_area(self):
        """
        self.area: the area of a single DAC (unit: mm2)
        """
        self.area = 0
        return self.area

    def calculate_delay(self):
        """
        self.delay: the delay of a single DAC (unit: ns)
        """
        self.delay = 0
        return self.delay

    def calculate_energy(self, vdd:float):
        """
        self.energy: the energy of a single DAC (unit: fJ)
        """
        if self.resolution == 1:
            self.energy = 0
        else:
            k0 = 50 # fF
            self.energy = k0 * self.resolution * vdd**2
        return self.energy

class PlaceValueAdderTree:
    """
    Class for a adder tree with each input having 1 place value (for AIMC)
    Note: for this adder tree, the input/output precision of n-th stage is: input_precision + 2**n
    Note: for AIMC, number_of_input here corresponds to Bw (weight precision in bit).
    :param vdd: supply voltage (unit: V)
    :param input_precision: input precision (unit: bit)
    :param number_of_input: the number of inputs
    :param unit_area: The area cost of single NAND2 gate (unit: mm2)
    :param unit_delay: The delay cost of single NAND2 gate (unit: ns)
    :param unit_cap: The input capacitance of single NAND2 gate (unit: fF)
    """
    def __init__(self, vdd: float, input_precision: int, number_of_input: int,  unit_area: float, unit_delay: float, unit_cap: float):
        if(math.log(number_of_input,2)%1 != 0):
            raise ValueError("PlaceValueAdderTree: the number of input for the adder tree is not in the power of 2. Currently it is: %s" %number_of_input)
        if(vdd < 0 or input_precision < 0 or number_of_input < 0 or unit_area < 0 or unit_delay < 0 or unit_cap < 0):
            raise ValueError("PlaceValueAdderTree: there should be no negtive value for parameters. Please recheck.")
        self.vdd = vdd
        self.input_precision = input_precision
        self.number_of_input = number_of_input
        self.depth = int( math.log(number_of_input, 2) ) # convert to int for looking nicer
        self.output_precision = input_precision + number_of_input
        self.number_of_1b_adder = input_precision * (number_of_input-1) + number_of_input * (self.depth-0.5)
        self.unit_area = unit_area
        self.unit_delay = unit_delay
        self.unit_cap = unit_cap
        self.unit_adder = Adder(vdd=self.vdd, input_precision=1, unit_area=self.unit_area, unit_delay=self.unit_delay, unit_cap=self.unit_cap)

    def calculate_area(self):
        """
        area: the area cost (unit: mm2)
        """
        self.area = self.number_of_1b_adder * self.unit_adder.calculate_area()
        return self.area

    def calculate_delay(self):
        """
        delay: the delay cost (unit: ns)
        """
        last_adder = Adder(vdd=self.vdd, input_precision=self.output_precision, unit_area=self.unit_area, unit_delay=self.unit_delay, unit_cap=self.unit_cap)
        self.delay = last_adder.calculate_delay_lsb() * (self.depth-1) + last_adder.calculate_delay_msb()
        return self.delay

    def calculate_energy(self):
        """
        energy: The energy cost (each time it is triggered) (unit: fJ)
        """
        self.energy = self.unit_adder.calculate_energy() * self.number_of_1b_adder
        return self.energy




