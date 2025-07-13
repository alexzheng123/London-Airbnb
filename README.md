# London Airbnb Price Forecasting

This project compares econometric and machine learning (ML) models to forecast Airbnb listing prices in London using real-world data from Inside Airbnb. The aim is to evaluate model performance and identify the most influential factors affecting listing prices.

## Dataset

- Source: [Inside Airbnb](https://insideairbnb.com/get-the-data/)
- Initial size: 95,144 observations and 75 variables
- Final cleaned dataset: 60,358 observations and 18 features after preprocessing

Key variables include:
- Price (target)
- Room type, Property type, Neighbourhood
- Accommodates, Bathrooms, Bedrooms, Beds
- Host experience, Identity verification, Reviews
- Availability (30 and 365 days)

Extreme outliers were removed (listings above $1,000), and categorical features were one-hot encoded.

## Project Highlights

- Focused only on "Entire home/apt" listings to examine factors (such as location) that can influence prices
- Conducted geospatial and 3D exploratory visualisation:
  - [Interactive Mapbox Plot](https://alexzheng123.github.io/London-Airbnb/mapbox.html)
  - [3D Scatter Plot](https://alexzheng123.github.io/London-Airbnb/3d_scatter.html)
- Identified Westminster, Kensington and Chelsea as the most expensive neighbourhoods

## Preprocessing

- One-hot encoding of categorical variables
- Removal of zero-variance features
- Train/Validation/Test split: 70%/20%/10%
- Standardisation and PCA for clustering
- Clustering performed using k-means with k=12, though cluster labels were not used in final models due to poor impact on performance

## Models Compared

| Model                          | Validation MAE | Test MAE |
|--------------------------------|----------------|----------|
| Baseline (Mean Price)          | 97.23          | 97.93    |
| Ridge Regression               | 61.82          | 61.88    |
| LASSO Regression               | 62.23          | 62.23    |
| Random Forest (Bayesian tuned) | **45.79**          | **45.40**    |
| Feedforward Neural Network     | 52.94          | 53.62    |

### Summary
- Random Forest achieved the best performance.
- Ridge and LASSO provided good interpretability, revealing that location features (e.g. Westminster) had the most impact.
- FNN handled complex nonlinear patterns well and ranked second in accuracy.

## Implementation Details

### Random Forest
- Hyperparameter tuning using Bayesian Optimisation
- Optimal `mtry`: 13, `ntree`: 443

### Feedforward Neural Network
- Architecture: 3 layers (64 → 32 → 1)
- Dropout: 0.3
- Optimiser: Adam
- Early stopping used (patience: 10)
- Trained for 100 epochs, batch size 32

## Setup Instructions

Create and activate a new environment using conda:

```
conda create --name ec349_env python=3.9
conda activate ec349_env
conda install tensorflow keras numpy pandas matplotlib seaborn
```

For users working with R and Python together (e.g. R Markdown or Quarto):

```r
library(reticulate)
use_condaenv("ec349_env", required = TRUE)
library(tensorflow)
library(keras)
```

## Insights

- Location, availability, and size-related variables were the most predictive of price.
- Host characteristics like profile picture and verification had little effect.
- Clustering (k-means) failed to improve prediction accuracy when added as a feature.

