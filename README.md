# ZigZag-IMC
This repository presents the extended version of ZigZag, a HW Architecture-Mapping Design Space Exploration (DSE) Framework.
This extension is specifically designed to support In-Memory Computing (IMC).
ZigZag-IMC bridges the gap between algorithmic DL decisions and their IMC acceleration cost on specialized accelerators through a fast and accurate HW cost estimation. 

## Important Notice

#### **2025/6/3**:

- Since the [ZigZag framework](https://github.com/KULeuven-MICAS/zigzag) has undergone several maintenance updates, some features and supporting materials related to IMC (In-Memory Computing) modeling are missing. Therefore, please feel free to use this repository for posting issues and running IMC simulations.
- The links to the IMC papers/videos/slides/posters have been removed from the [ZigZag framework](https://github.com/KULeuven-MICAS/zigzag). If you are looking for details on the modeling methodology, please refer to the resources linked at the bottom of this page.

The comparison between this ZigZag-IMC repository and the ZigZag framework is tabulated below.

| Framework                                                    | Same                                         | Diff                                                                                                                                                  |
|:-------------------------------------------------------------|:---------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------|
| [ZigZag framework](https://github.com/KULeuven-MICAS/zigzag) | support both digital and IMC architecture    | does not support diagonal spatial unrolling and multi-dimensional spatial unrolling (can only unroll C or FX or FY along the same hardware dimension) |
| ZigZag-IMC                                                   | support both digital and IMC architecture    | support diagonal spatial unrolling and multi-dimensional spatial unrolling (e.g., unroll C, FX, FY in together along the same hardware dimension)     |

#### **2024/3/14**:

- Considering ZigZag-IMC framework is an enhanced version of ZigZag framework, we have merged this ZigZag-IMC repository with the [ZigZag framework](https://github.com/KULeuven-MICAS/zigzag) for an easy maintenance. 
Please refer to the [ZigZag framework](https://github.com/KULeuven-MICAS/zigzag), which will have regular updates and maintenance, if you are looking for ZigZag-IMC framework. This repository for ZigZag-IMC will no longer be maintained in the future.

#### Literature comparison

| Framework                                                | Pros                                                               | Cons                                                                                                                       |
|:---------------------------------------------------------|:-------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------|
| [Cimloop](https://arxiv.org/abs/2405.07259) (ISPASS2024) | extended with NeuroSim, support NVM-based IMC                      | model validated to 1 chip per technology, ADC models extracted from pure ADC works (higher energy and area cost, no delay) |
| ZigZag-IMC                                               | models validated against 7 IMC chips on 28nm, higher reliability   | only support SRAM-based IMC                                                                                                |

## Prerequisite

To get started, you can install all packages directly through pip using the pip-requirements.txt with the command:

`$ pip install -r requirements.txt`

## Getting Started
The main script is `main_onnx.py`, which takes into 3 input files:
- .onnx: workload definition.
- mapping: user-provided spatial mapping or spatial mapping restriction.
- accelerator: hardware definition.

The repository includes three examples provided in the `run.sh` script, which are:
- an example for a pure digital PE-based hardware template.
- an example for an SRAM-based Digital In-Memory Computing hardware template.
- an example for an SRAM-based Analog In-Memory Computing hardware template.

The output will be saved under `outputs/` folder.

API is also created in `zigzag/api.py`, which can be directly called externally.

**Note**: Since CACTI is not supported on Windows, please run these example on Linuxs, or else user-defined memory cost value should be provided in the input hardware file.

## New features
In this novel version, in addition to the features found in the basic zigzag, we have introduced several new capabilities:
- **New cost models**: Added support for SRAM-based Analog In-Memory Computing and Digital In-Memory Computing (28nm).
- **Dataflow optimization***: Inner layer data will remain in lower memory if it is expected to be used by the next layer along the same branch.
- **Mix Spatial Mapping***: Mix user-defined spatial mapping is now allowed (refer to `inputs/examples/mapping/default_imc.py`).
- **Diagonal OX/OY Mapping***: Computing array now supports diagonal OX/OY mapping.
- **Automatic Mix Spatial Mapping***: Spatial mapping will be autogenerated if a `spatial_mapping_hint` is provided in the mapping file.
- **Simulation Speedup**: Only the three spatial mappings with the highest hardware utilization will be assessed to ensure fast simulation speed.

*: features have been integrated into the base ZigZag framework.

## In-Memory Computing Hardware Cost Model Description
Our SRAM-based In-Memory Computing model is a versatile, parameterized model designed to cater to both Analog IMC and Digital IMC.
Since hardware costs are technology-node dependent, we have performed special calibration for the 28nm technology node. The model has been validated against 7 chips from the literature. 
A summary of the hardware settings for these chips is provided in the following table.

| source                                                          | label | B<sub>i</sub>/B<sub>o</sub>/B<sub>cycle</sub> | macro size     | #cell_group | nb_of_macros |
|-----------------------------------------------------------------|-------|-----------------------------------------------|----------------|-------------|--------------|
| [paper](https://ieeexplore.ieee.org/abstract/document/9431575)  | AIMC1 | 7 / 2 / 7                                     | 1024&times;512 | 1           | 1            |
| [paper](https://ieeexplore.ieee.org/abstract/document/9896828)  | AIMC2 | 8 / 8 / 2                                     | 16&times;12    | 32          | 1            |
| [paper](https://ieeexplore.ieee.org/abstract/document/10067289) | AIMC3 | 8 / 8 / 1                                     | 64&times;256   | 1           | 8            |
| [paper](https://ieeexplore.ieee.org/abstract/document/9731762)  | DIMC1 | 8 / 8 / 2                                     | 32&times;6     | 1           | 64           |
| [paper](https://ieeexplore.ieee.org/abstract/document/9731545)  | DIMC2 | 8 / 8 / 1                                     | 32&times;1     | 16          | 2            |
| [paper](https://ieeexplore.ieee.org/abstract/document/10067260) | DIMC3 | 8 / 8 / 2                                     | 128&times;8    | 8           | 8            |
| [paper](https://ieeexplore.ieee.org/abstract/document/10067779) | DIMC4 | 8 / 8 / 1                                     | 128&times;8    | 2           | 4            |

B<sub>i</sub>/B<sub>o</sub>/B<sub>cycle</sub>: input precision/weight precision/number of bits processed per cycle per input.
#cell_group: the number of cells sharing one entry to computation logic.

The validation results are displayed in the figure below (assuming 50% input toggle rate and 50% weight sparsity are assumed). 
The gray bar represents the reported performance value, while the colored bar represents the model estimation.
The percent above the bars is the ratio between model estimation and the chip measurement results.

<p align="center">
<img src="https://github.com/KULeuven-MICAS/zigzag-imc/blob/master/imc_model_validation/model_validation.png" width="100%" alt="imc model validation plot">
</p>

- AIMC1 incurs additional area costs due to repeaters/decaps.
- Sparsity information is not available for AIMC2, DIMC2, DIMC4.
- AIMC1, AIMC3 were fabricated using 22nm technology, therefore the cost estimation was scaled accordingly.


**Note:**

The current integrated IMC model has certain limitations and is applicable only under the following conditions:
- The SRAM cell is a 6T memory cell.
- The adder tree follows a RCA (Ripple Carry Adder) structure without any approximation logic.
- The operands are of integer type rather than floating point.
- The voltage used for the delay estimation is fixed at 0.9 V.
- Sparsity impact is not included in the estimated energy cost.


## Publication pointers
- J. Sun, P. Houshmand and M. Verhelst, "Analog or Digital In-Memory Computing? Benchmarking through Quantitative Modeling," Proceedings of the IEEE/ACM Internatoinal Conference On Computer Aided Design (ICCAD), October 2023. [paper](https://ieeexplore.ieee.org/document/10323763), [poster](https://drive.google.com/file/d/1EVdua-y2Wg8WL-ovUIw7KUR9kpnpN4AS/view?usp=sharing), [slides](https://docs.google.com/presentation/d/19OXRDh6NCBUIOVGneO3lrZfVT58xh06U/edit?usp=sharing&ouid=108247328431603587200&rtpof=true&sd=true), [video](https://drive.google.com/file/d/10-k4XEPan-O-QAH4Q0uvone36qfNRCpK/view?usp=sharing)

- P. Houshmand, J. Sun and M. Verhelst, "Benchmarking and modeling of analog and digital SRAM in-memory computing architectures," arXiv preprint arXiv:2305.18335 (2023). [paper](https://arxiv.org/abs/2305.18335)



