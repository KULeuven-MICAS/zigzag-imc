
import math


def get_adder_depth_structure(adder_depth):
    adder_depth_structure = []
    while adder_depth > 0:
        print(adder_depth, math.floor(math.log2(adder_depth)))
        adder_depth_structure.append(2**math.floor(math.log2(adder_depth)))
        adder_depth -= 2**(math.floor(math.log2(adder_depth)))

    return adder_depth_structure

aa = get_adder_depth_structure(24)
print(aa)
breakpoint()
