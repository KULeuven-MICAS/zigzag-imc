from zigzag.classes.hardware.architecture.ImcArray import ImcArray
import numpy as np

def imc_array_dut(dimensions, group_depth):

    """Multiplier array variables"""
    tech_param = { # 28nm
        "tech_node": 0.028,             # unit: um
        "vdd":      0.9,                # unit: V
        "nd2_cap":  0.7/1e3,            # unit: pF
        "xor2_cap": 0.7*1.5/1e3,        # unit: pF
        "dff_cap":  0.7*3/1e3,          # unit: pF
        "nd2_area": 0.614/1e6,          # unit: mm^2
        "xor2_area":0.614*2.4/1e6,      # unit: mm^2
        "dff_area": 0.614*6/1e6,        # unit: mm^2
        "nd2_dly":  0.0478,             # unit: ns
        "xor2_dly": 0.0478*2.4,         # unit: ns
        # "dff_dly":  0.0478*3.4,         # unit: ns
    }
    hd_param = {
        "pe_type":              "in_sram_computing",     # for in-memory-computing. Digital core for different values.
        "imc_type":             "digital",  # "digital" or "analog"
        "input_precision":      8,          # activation precision expected in the hardware
        "weight_precision":     8,          # weight precision expected in the hardware
        "input_bit_per_cycle":  1,          # nb_bits of input/cycle/PE
        "group_depth":          group_depth,          # group depth in each PE
        "wordline_dimension": "D1",         # hardware dimension where wordline is (corresponds to the served dimension of input regs)
        "bitline_dimension": "D2",          # hardware dimension where bitline is (corresponds to the served dimension of output regs)
        "enable_cacti":         True,       # use CACTI to estimated cell array area cost (cell array exclude build-in logic part)
        "adc_resolution": 1#max(1, 0.5 * np.log2(dimensions['D1']))
        # Energy of writing weight. Required when enable_cacti is False.
        # "w_cost_per_weight_writing": 0.08,  # [OPTIONAL] unit: pJ/weight.
    }

    imc_array = ImcArray(
        tech_param, hd_param, dimensions
    )

    return imc_array


