# ML-for-quantum-annealing-penalties
In this repo, I implemented a regressor that fits synthetic data on a simplified green hydrogen production pipeline.
The goal is to predict the penalties that are to be used in the restrictions of a QUBO problem before feeding it into a Quantum annealer. 
After making the prediction, a comparison is made between a QUBO using uniform penalties for each restriction and a QUBO whose penalties have been refinded by an ML algorithm.

On notebooks usage:

- Data generator.ipynb generated the main dataset, later used for ML predictions and for the testing of the optimization algorithms.

- Regressor.ipynb tales the generated data and build a simple regressor that estimates the penalties for this specific problem 

- Baseline.ipynb is used for the execution and testing of MINLP optimization with Pyomo and Dwave annealer

- Annealer.ipynb notebook is used for the execution of annealing schedules in Dwave

- execution.ipynb executes the final test, building a dataframe containing all the results.

- comparison.ipynb takes the final dataframe containing the results and showcases the results

