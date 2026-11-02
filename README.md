# Monoclonal Antibody Production Optimizer
A python based tool that optimizes antibody production based on initial conditions and harvest times
that maximize space-time profit. Polynomial regression and KNN regression models are trained using time series data
to predict culture performance and estimate uncertainty in outcomes.

## Highlights
- Developed an optimization program for mAb production
- Implemented polynomial NVAR and KNN regression models
- Optimized model hyperparameters using grid search cross validation
- Simulated batch trajectories using residual based uncertainty estimation
- Identified operating conditions that maximize predicted space-time profit

## Environment
- Python 3.14.7
- matplotlib 3.11.0
- numpy 2.5.2
- pandas 3.0.5
- sklearn 1.9.0

## Code Design
The major stages when main.py is executed are as follows:
1. data loading and batch selection
2. lagged state generation and model building
3. model training and tuning
4. prediction and uncertainty estimation
5. operating conditions and harvest time optimization

### 1. Data loading and batch selection
The program begins by loading data contained in the "initial_conditions.csv" file, removing duplicate combinations of 
the initial conditions, and selecting 30 batch runs that are evenly spaced throughout the remaining data. 
The corresponding time series data for each selected batch was manually moved to the "datasets" folder for later use.\
The selected batch data is combined into a single data frame sorted by batch dentification number and time.

### 2. Lagged state generation and model building
A nonlinear vector autoregressive (NVAR) model is used to predict the state of the cell culture, where previous time 
series data observations are used to predict the next state. There are 7 variables included in this: glucose concentration,
lactate concentration, ammonia concentration, viable cell density (VCD), total cell density (TCD), and monoclonal antibody (mAb) concentration.\
For each batch, the time series measurements are arranged into lagged input vectors containing the most recent observation
and the one before it. The corresponding current measurements become prediction targets for the model.\
Two NVAR models, polynomial and KNN, were evaluated. Polynomial regression allows nonlinear relations and interactions
between lagged input variables, and KNN regression allows prediction of future states based on close observations.

### 3. Model training and tuning
Both models were evaluated using 4-fold cross validation using the soot mean squared error (RMSE) as the scoring metric.
The model with the lowest mean cross validation RMSE is selected as the final model, with comparison results exported 
to the "tables" folder./
For polynomial regression, the polynomial degree was tested. For KNN regression, the number of neighbours as well as the 
type of weighting (distance or uniform) was tested.

### 4. Prediction and uncertainty estimation
After selecting the best model, residuals are calculated by comparing the model's predictions to the actual observations.
Prediction begins with an initial state from the selected batch glucose concentration, glutamine concentration, and VCD. 
The initial lactate concentration, ammonia concentration, and mAb concentration are set to 0, and the initial TCD is
assumed to be equal to the VCD.\
The model predicts the next culture state using the previous 2 observations, and repeats this process until the desired
number of steps has been completed. To account for prediction uncertainty, a residual value previously calculated is 
selected and added to the predicted state.\
Multiple trajectories are generated for each set of initial conditions, producing distributions which are used to estimate
approximate 95% prediction intervals for the economic and productivity metrics.

### 5. Operating conditions and harvest time optimization
The predicted performance of each initial condition at every harvest time is evaluated by calculation of the culture 
cost, culture value, and culture profit, and then the space time profit, and space time yield. For each combination
of initial conditions and harvest time, the mean, 2.5 percentile, and 97.5 percentile of the calculated culture value, 
space time profit, and space time yield are calculated. The percentiles provide an approximate 95% prediction interval.\
All combinations are evaluated, and the operating conditions and harvest time associated with the highest predicted mean
space time profit are identified.

## Analysis
### Selection of Initial Conditions 
The initial variables focused on were [glucose], [glutamine], and VCD. Duplicate combinations of these variables were 
removed from the available data, and the remaining combinations were sorted by their initial conditions.
The initial conditions selected to conduct the 30 batch runs were then chosen by equal spacing of data.\ 
This method of batch selection ensures a range of different operating conditions without replicate data,
and also allows reproducibility.

### Model Architecture
Two nonlinear regression models, polynomial regression and k-nearest neighbours (KNN) regression were implemented and evaluated to
identify a suitable model for prediction of future culture states. A nonlinear autoregressive model was chosen due to 
the time dependence of the culture variables and possible nonlinear relations.\
Polynomial regression was chosen due to its capability of representing non-linear relationships between variables while
KNN regression was chosen due to its prediction of target values based on observations with similar input features.\
The model hyperparameters were optimized using grid search with a 4-fold cross validation. The resulting scores give a 
better estimation of model performance in comparison to a standard train-validation split.\
For polynomial regression, polynomial degrees of 2, 3, 4, and 5 were tested. 
For KNN regression, neighbour counts of 2, 3, 3, amd 5 were tested, with uniform and distance-based weighting.

### Model Comparison and Selection
Models were compared using a 4-fold grid search cross validation. The root mean squared error (RMSE) was used as the 
scoring metric, where a lower value indicates a closer prediction to the actual culture state. Model comparison produced
the following results:
|Model          |Best Parameters                   |Mean CV RMSE|
|---------------|----------------------------------|------------|
|Polynomial NVAR|Polynomial Degree = 2             |28633.63233 |
|KNN NVAR       |Neighbours = 5, Weights = distance|117183.1283 |

Polynomial NVAR resulted in a significantly lower CV RMSE tha KNN NVAR, therefore polynomial NVAR with a degree of 2 
was selected as the final model.\
The results suggest that a polynomial model was more representative of the relationship between the lagged variables and 
culture states. An advantage of polynomial regression is that it is capable of representing nonlinear trends while 
remaining relatively simple to evaluate. At higher degrees, however, polynomial regression can increase model complexity
and may result in overfitting. 


### Influence of Initial Conditions on Culture Performance
The initial glucose concentration influences the available energy at the beginning of the culture process. Glucose 
availability can promote cell growth and metabolism, contributing to biomass accumulation and antibody production. 
Increasing the initial glucose concentration also increases culture cost, which limits the space-time profit achieved.\
The initial glutamine concentration has similar effects to glucose concentration, where readily available glutamine
may promote cell growth and metabolism, however higher concentrations increase the initial cost. Additionally, glutamine 
consumption can result in accumulation of ammonia as a metabolic byproduct, which could have negative effects on culture
growth in the long term.\
The initial VCD determines the number of viable cells at the beginning of the batch process. A higher initial VCD allows 
the culture to begin producing mAb with a larger cell population, however this increases the culture cost.\
The combined effect of these variables determines the overall economic performance of the culture. Culture value depends 
on the final mAb concentration, and the culture cost depends on the initial [glucose], [glutamine], and VCD. Space time 
profit depends on both economic performance and harvesting time

### Recommended Initial Operating Conditions
The optimal initial operating conditions are as follows:
|Initial Condition     |Optimal Value|
|----------------------|-------------|
|Glucose [mM]          |15.000       |
|Glutamine [mM]        |2.000        |
|Initial VCD [cells/mL]|400000.000   |
|Harvest Time [d]      |4.667        |

The estimated culture cost, culture value, space time profit, and space time yield are as follows:
|Parameter                                |Value                 |
|-----------------------------------------|----------------------|
|Culture Value Mean [$/L]                 |799.61761             |
|Culture Value 95% Prediction Interval    |[586.02441 ; 993.03079|
|Space-Time Profit Mean [$/L/d]           |72.28546              |
|Space-Time Profit 95% Prediction Interval|[34.59254 ; 106.41720]|
|Space-Time Yield Mean [mg/L/d]           |0.14111               |
|Space-Time Yield 95% Prediction Interval |[0.10342 ; 0.17524]   |

### Validation of Results
The recommended initial operating conditions are those which resulted in the maximum predicted mean space time profit.
A table containing the optimization results (found in "tables"/"optimization_results.csv") contains the estimated values
for culture value, space time profit, and space time yield for each of the 30 batches of initial operating conditions.
The combination of [glucose], [glutamine], and VCD which results in the largest space time profit from this data file
is chosen as the recommended conditions.\
The recommended harvest time was found in a similar manner, where the time which resulted in teh largest profit was chosen
as the recommended time. This can be seen graphically below:
<img width="3200" height="2400" alt="profit_vs_harvest_time" src="https://github.com/user-attachments/assets/8421efb3-09b1-4a14-9c50-2cf24db0b18d" />
This graph demonstrates the relationship between harvest time and profit, allowing graphical determination of the optimal
harvest time. The harvest time corresponding to the point with most profit was chosen as the recommended harvest time.

