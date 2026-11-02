
from src.classes import AntibodyOptimizer

optimizer = AntibodyOptimizer(filepath="datasets/initial_conditions.csv")

# load the 30 selected batch runs
optimizer.load_data(number_of_batches=30)

# train and compare the regression models
comparison = optimizer.train_and_compare()

# optimize operating conditions and harvest time
optimizer.optimize(n_simulations=200)

# display the recommended conditions
optimizer.print_summary()

print("\nMODEL COMPARISON")
print(comparison.round(4).to_string(index=False))