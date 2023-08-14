AIMC:
To get mismatch value and cost breakdown for AIMC under 28nm, run "python aimc_validation.py" under aimc_validation/22-28nm/.

DIMC:
To extract parameters for DIMC under 28nm, run "python model_extraction_28nm.py" in dimc_validation/28nm/, then it will print out the best fitting value for energy/area/delay (tclk) AND the average mismatch. (note: paper2 is not used for energy validation because it's not reported)

Run "dimc_validation.py", it will get the mismatch value and cost breakdown for each paper.


