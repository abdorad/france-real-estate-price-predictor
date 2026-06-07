# France Real Estate Price Predictor

This project is a machine learning web app that estimates residential property prices in France and classifies a listing as undervalued, fairly priced, or overvalued.

The model is trained on the French DVF dataset, a public government dataset containing real estate transactions across France.

This repository contains both the trained model and the training code. That makes the project directly usable while still showing the full machine learning pipeline for academic review.

## Features

- Predicts an estimated market price for a property.
- Compares the estimated price with a user-provided listing price.
- Classifies the listing as:
  - Undervalued
  - Fair price
  - Overvalued
- Displays price per square meter for both the predicted price and listing price.
- Uses a saved Scikit-learn model loaded with Joblib.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

## Project Structure

```text
france-real-estate-price-predictor/
  app.py
  train_model.py
  requirements.txt
  README.md
  .gitignore
  france_real_estate_price_model.joblib
```

The trained model is included:

```text
france_real_estate_price_model.joblib
```

The app can run directly after installing the dependencies. The training script is also included so the model can be reproduced from the DVF dataset.

## Train the Model

The included model can be used immediately, but the full training code is provided for reproducibility.

If you already have the DVF text file:

```bash
python train_model.py --data-file ValeursFoncieres-2024.txt
```

You can also download the 2024 DVF dataset directly:

```bash
python train_model.py --download
```

After training, the script creates:

```text
france_real_estate_price_model.joblib
```

This is the file used by the Streamlit app.

## Model Input Features

The app expects the saved model to use the same feature format as the training code:

| Feature | Description |
| --- | --- |
| `Surface reelle bati` | Living surface in square meters |
| `Nombre pieces principales` | Number of main rooms |
| `surface_log` | Log-transformed living surface |
| `terrain_log` | Log-transformed land surface |
| `rooms_per_m2` | Room density |
| `month` | Transaction month |
| `Type local` | Property type, either `Appartement` or `Maison` |
| `departement` | French department extracted from postal code |
| `cp_model` | Postal-code feature used by the model |

## Model Performance

Final model performance after cleaning and feature engineering:

```text
Rows used: 869,761
MAE: 73,090.51 EUR
Median absolute error: 44,534.34 EUR
RMSE: 122,376.68 EUR
R2: 0.6448
```

These results are reasonable for a national real estate prediction model using limited public transaction features. The prediction should be treated as a market estimate, not an exact valuation.

For a master application, this is the key point of the project: it is not only a prediction app, but an end-to-end machine learning workflow with data preparation, feature engineering, model training, evaluation, model saving, and deployment through Streamlit.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/france-real-estate-price-predictor.git
cd france-real-estate-price-predictor
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

Start the Streamlit app:

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Example

Input:

```text
Property type: Appartement
Surface: 65 m2
Rooms: 3
Postal code: 75015
Listing price: 280,000 EUR
```

Example output:

```text
Predicted price: 605,076.84 EUR
Difference: -53.72%
Status: Undervalued
```

The listing is considered undervalued because the listing price is much lower than the model's estimated market price.

## Price Classification Logic

The app compares the listing price to the predicted price:

```text
difference = (listing_price - predicted_price) / predicted_price * 100
```

Default rule:

| Difference | Status |
| --- | --- |
| Greater than +10% | Overvalued |
| Between -10% and +10% | Fair price |
| Less than -10% | Undervalued |

The threshold can be changed in the app sidebar.

## Limitations

This model does not know every factor that affects a real property price. Important missing factors may include:

- Property condition
- Floor number
- Elevator
- Energy rating
- Exact address
- View
- Renovation quality
- Neighborhood micro-location
- Local demand at the time of sale

Because of this, the prediction should be used as a decision-support estimate, not as a certified appraisal.

## Data Source

DVF dataset from the French government open data platform:

https://www.data.gouv.fr/


