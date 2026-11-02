
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV


class AntibodyOptimizer:

    def __init__(self, filepath):
        self.filepath = filepath

        self.features = [
            "Glucose [mM]",
            "Glutamine [mM]",
            "Lactate [mM]",
            "Ammonia [mM]",
            "VCD [cells/mL]",
            "TCD [cells/mL]",
            "mAb [mg/L]"
        ]

        self.initial_conditions = None
        self.data = None
        self.results = None
        self.best_model = None
        self.best_model_name = None
        self.residuals = None

        self.x_lag = None
        self.x_cur = None
        self.b_cur = None

        self.optimization_results = None
        self.optimal_conditions = None

        self.models = None
        self.poly_validation_results = None
        self.knn_validation_results = None

        self.p = 2

    def load_data(self, number_of_batches=30):

        filepath = os.path.join("datasets", "initial_conditions.csv")
        initial_conditions = pd.read_csv(filepath)

        # remove duplicate initial condition combinations
        condition_columns = [
            "Glucose [mM]",
            "Glutamine [mM]",
            "VCD [cells/mL]"
        ]

        unique_conditions = initial_conditions.drop_duplicates(
            subset=condition_columns
        )

        # sort conditions
        unique_conditions = unique_conditions.sort_values(
            by=condition_columns
        ).reset_index(drop=True)

        # select 30 evenly spaced rows from initial_conditions.csv
        selected_indices = np.linspace(
            0,
            len(unique_conditions) - 1,
            number_of_batches,
            dtype=int
        )

        # store and print the selected batches
        self.initial_conditions = (
            unique_conditions.iloc[selected_indices].copy()
        )

        print("\nSelected batch IDs:")
        print(
            self.initial_conditions["Batch ID"].astype(int).tolist()
        )

        # load the time series data for each selected batch
        batch_data = []

        for _, row in self.initial_conditions.iterrows():
            batch_id = int(row["Batch ID"])

            filename = "batch_" + str(batch_id).zfill(4) + ".csv"
            filepath_batch = os.path.join(
                "datasets",
                filename
            )

            batch = pd.read_csv(filepath_batch)

            batch.insert(0, "Batch ID", batch_id)

            batch_data.append(batch)

        # to combine the batch data into a single data frame
        self.data = pd.concat(
            batch_data,
            ignore_index=True
        )

        self.data = self.data.sort_values(
            by=["Batch ID", "Time [h]"]
        ).reset_index(drop=True)

        print("\nLoaded", len(batch_data), "batch files.")
        print("Total time-series observations:", len(self.data))

        # store the combined batch data to a csv file
        self.data.to_csv(
            os.path.join("tables", "combined_batch_data.csv"),
            index=False
        )

        return self.data

    def generate_lagged_data(self):
        x_lag = []
        x_cur = []
        b_cur = []

        # transform time series data to inputs and targets
        for batch_id in self.data["Batch ID"].unique():
            batch = self.data[
                self.data["Batch ID"] == batch_id
                ].sort_values("Time [h]")

            values = batch[self.features].to_numpy()

            for i in range(self.p, len(values)):

                lag_values = []

                for lag in range(1, self.p + 1):
                    lag_values.extend(values[i - lag])

                x_lag.append(lag_values)
                x_cur.append(values[i])
                b_cur.append(batch_id)

        return (
            np.asarray(x_lag),
            np.asarray(x_cur),
            np.asarray(b_cur)
        )

    def train_and_compare(self):

        # generate lagged data
        x_lag, x_cur, b_cur = self.generate_lagged_data()

        self.x_lag = x_lag
        self.x_cur = x_cur
        self.b_cur = b_cur

        # define the models
        models = {
            "Polynomial NVAR": make_pipeline(
                StandardScaler(),
                PolynomialFeatures(include_bias=False),
                LinearRegression()
            ),

            "KNN NVAR": make_pipeline(
                StandardScaler(),
                KNeighborsRegressor()
            )
        }

        # define hyperparameter grids for cross val.
        param_grids = {
            "Polynomial NVAR": {
                "polynomialfeatures__degree": [2, 3, 4, 5]
            },

            "KNN NVAR": {
                "kneighborsregressor__n_neighbors": [2, 3, 4, 5],
                "kneighborsregressor__weights": [
                    "uniform",
                    "distance"
                ]
            }
        }

        # cross val. to compare models
        comparison = []
        tuned_models = {}

        for name, model in models.items():

            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grids[name],
                cv=4,
                scoring="neg_root_mean_squared_error",
                refit=True,
                return_train_score=True,
                n_jobs=-1
            )

            grid_search.fit(x_lag, x_cur)

            # save results
            cv_results = grid_search.cv_results_

            if name == "Polynomial NVAR":
                self.poly_validation_results = pd.DataFrame({
                    "Polynomial Degree": cv_results[
                        "param_polynomialfeatures__degree"
                    ].astype(int),

                    "Train RMSE": -cv_results["mean_train_score"],

                    "Valid RMSE": -cv_results["mean_test_score"]
                })

                self.poly_validation_results = (
                    self.poly_validation_results
                    .sort_values("Polynomial Degree")
                    .reset_index(drop=True)
                )

            if name == "KNN NVAR":
                self.knn_validation_results = pd.DataFrame({
                    "Neighbours": cv_results[
                        "param_kneighborsregressor__n_neighbors"
                    ].astype(int),

                    "Weights": cv_results[
                        "param_kneighborsregressor__weights"
                    ].astype(str),

                    "Train RMSE": -cv_results["mean_train_score"],

                    "Valid RMSE": -cv_results["mean_test_score"]
                })

                self.knn_validation_results = (
                    self.knn_validation_results
                    .sort_values(["Weights", "Neighbours"])
                    .reset_index(drop=True)
                )

            # save best model from grid search
            best_model = grid_search.best_estimator_
            tuned_models[name] = best_model

            best_index = grid_search.best_index_

            mean_cv_rmse = -cv_results["mean_test_score"][best_index]

            # to format things nicer
            if name == "Polynomial NVAR":

                best_parameters = (
                    f"Polynomial Degree = "
                    f"{grid_search.best_params_['polynomialfeatures__degree']}"
                )

            else:

                best_parameters = (
                    f"Neighbours = "
                    f"{grid_search.best_params_['kneighborsregressor__n_neighbors']}, "
                    f"Weights = "
                    f"{grid_search.best_params_['kneighborsregressor__weights']}"
                )

            # compare models by table
            comparison.append({
                "Model": name,
                "Best Parameters": best_parameters,
                "Mean CV RMSE": mean_cv_rmse
            })

        # store tuned models and comparison results
        self.models = tuned_models
        self.results = pd.DataFrame(comparison)

        # select model with lowest cross val. rmse
        best_index = self.results["Mean CV RMSE"].idxmin()

        self.best_model_name = self.results.loc[
            best_index, "Model"
        ]

        self.best_model = self.models[
            self.best_model_name
        ]

        # calculate residuals for uncertainty simulations
        predicted_states = self.best_model.predict(x_lag)

        self.residuals = (
                np.asarray(x_cur) - np.asarray(predicted_states)
        )

        # export results
        self.results.to_csv(
            os.path.join("tables", "model_comparison.csv"),
            index=False,
            float_format="%.5f"
        )

        return self.results

    def predict(self, initial_state, number_of_steps, add_uncertainty=False):
        history = [
            np.asarray(initial_state, dtype=float)
        ]

        # to predict the next state
        for _ in range(self.p - 1):
            history.insert(0, history[0].copy())

        for _ in range(number_of_steps):

            lag_values = []

            # arrange lagged states from newest to oldest
            for lag in range(1, self.p + 1):
                lag_values.extend(history[-lag])

            x_next = np.asarray(lag_values).reshape(1, -1)

            next_state = self.best_model.predict(x_next)[0]

            # adds error to the predicted values using residuals
            if add_uncertainty:
                residual_index = np.random.randint(
                    0,
                    len(self.residuals)
                )

                next_state = (
                    next_state + self.residuals[residual_index]
                )

            # to prevent negative values for conc. and # of cells
            next_state = np.maximum(next_state, 0)

            history.append(next_state)

        return np.asarray(history)

    def optimize(self, n_simulations=200):
        time_values = np.sort(
            self.data["Time [h]"].unique()
        )

        number_of_steps = len(time_values) - 1
        records = []

        for _, condition in self.initial_conditions.iterrows():
            glucose = condition["Glucose [mM]"]
            glutamine = condition["Glutamine [mM]"]
            vcd = condition["VCD [cells/mL]"]

            # to define the initial conditions
            initial_state = [
                glucose,
                glutamine,
                0,       # initial lactate
                0,       # initial ammonia
                vcd,     # initial viable cell density
                vcd,     # initial total cell density
                0        # initial antibody conc.
            ]

            # perform simulations for initial conditions
            trajectories = []

            for _ in range(n_simulations):
                trajectory = self.predict(
                    initial_state=initial_state,
                    number_of_steps=number_of_steps,
                    add_uncertainty=True
                )

                trajectories.append(trajectory)

            trajectories = np.asarray(trajectories)

            # calculate culture cost ($/L)
            culture_cost = (
                10 * glucose
                + 20 * glutamine
                + 0.0005 * vcd
            )

            for step in range(1, number_of_steps + 1):
                harvest_time_days = time_values[step] / 24 #convert to days
                denominator = harvest_time_days + 1 # +1 day of downtime

                mab_predictions = trajectories[:, step, 6]

                # calculate culture value ($/L)
                culture_values = 1000 * mab_predictions

                # calculate profit ($/L)
                profits = culture_values - culture_cost

                # calculate space time profit ($/L/day)
                space_time_profits = profits / denominator

                #calculate space time yield ([mAb]/day)
                space_time_yields = mab_predictions / denominator

                # to organise
                records.append({
                    "Batch ID": int(condition["Batch ID"]),
                    "Glucose [mM]": glucose,
                    "Glutamine [mM]": glutamine,
                    "Initial VCD [cells/mL]": vcd,
                    "Harvest Time [d]": harvest_time_days,

                    #calculate the central 95% prediction interval
                    "Culture Value Mean [$/L]":
                        np.mean(culture_values),

                    "Culture Value 2.5% [$/L]":
                        np.percentile(culture_values, 2.5),

                    "Culture Value 97.5% [$/L]":
                        np.percentile(culture_values, 97.5),

                    "Space-Time Profit Mean [$/L/d]":
                        np.mean(space_time_profits),

                    "Space-Time Profit 2.5% [$/L/d]":
                        np.percentile(space_time_profits, 2.5),

                    "Space-Time Profit 97.5% [$/L/d]":
                        np.percentile(space_time_profits, 97.5),

                    "Space-Time Yield Mean [mg/L/d]":
                        np.mean(space_time_yields),

                    "Space-Time Yield 2.5% [mg/L/d]":
                        np.percentile(space_time_yields, 2.5),

                    "Space-Time Yield 97.5% [mg/L/d]":
                        np.percentile(space_time_yields, 97.5)
                })

        # save results to a csv file
        self.optimization_results = pd.DataFrame(records)

        self.optimization_results.to_csv(
            os.path.join(
                "tables",
                "optimization_results.csv"
            ),
            index=False,
            float_format="%.5f"
        )

        # select conditions with highest mean spacetime profit
        best_index = self.optimization_results[
            "Space-Time Profit Mean [$/L/d]"
        ].idxmax()

        self.optimal_conditions = (
            self.optimization_results.loc[[best_index]].copy()
        )

        self.optimal_conditions.to_csv(
            os.path.join(
                "tables",
                "optimal_conditions.csv"
            ),
            index=False,
            float_format="%.5f"
        )

        # plot mean space time profit vs harvest time
        plt.figure(figsize=(8, 6))

        plt.scatter(
            self.optimization_results["Harvest Time [d]"],
            self.optimization_results["Space-Time Profit Mean [$/L/d]"],
            marker="o",
            color="tab:blue"
        )

        plt.xlabel("Harvest Time [d]")
        plt.ylabel("Mean Space-Time Profit [$/L/d]")
        plt.title("Space-Time Profit vs. Harvest Time")
        plt.tight_layout()

        plt.savefig(os.path.join(
                "figures",
                "profit_vs_harvest_time.png"
            ),
            dpi=400
        )

        plt.close()

        return self.optimization_results

    def print_summary(self):
        print("\nSelected model:")
        print(self.best_model_name)

        print("\nRecommended operating conditions:")
        print(
            self.optimal_conditions.round(4).to_string(index=False)
        )




