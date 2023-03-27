# PASS

This code implements bi-PASS Sync, a synchronized version of bisection Parallel Adaptive Survivor Selection, based on these resources: 

Pei, L., Nelson, B. L., and Hunter, S. R. (2022). ``Parallel Adaptive Survivor Selection.'' Operations Research.
Pei, L., Hunter, S., Nelson, B. (2020). ``Evaluation of bi-PASS for Parallel Simulation Optimization.'' Proceedings of the Winter Simulation Conference 2020.
Pei, L., Hunter, S., Nelson, B. (2018). ``A New Framework for Ranking and Selection Using an Adaptive Standard.'' Proceedings of the Winter Simulation Conference 2018.

bi-PASS Sync uses simulation optimization to maximize an unconstrained objective function.

To run the example file, run
sh example.sh

To run bi-PASS on another problem, create your own config file based on example_config.py
The main function to adjust in the config file is simulation_model()
If the true standard, true means, or true variances are not known, these can be set to -np.inf.

The boundary function is calibrated for a 5\% expected false elimination rate based on an estimated variance using $10$ observations.

Proper documentation coming soon, check back April 2023.
