# Customer Churn Prediction

A small customer churn prediction project built with TensorFlow and Streamlit. The saved neural network estimates the probability that a customer will leave based on their account and profile details.

## What’s included

- `app.py` — Streamlit interface for entering customer details and viewing a prediction.
- `model.h5` — trained Keras model.
- `scaler.pkl`, `label_encoder_gender.pkl`, `onehot_encoder_geo.pkl` — preprocessing objects used during training.
- `Churn_Modelling.csv` — dataset used by the project.
- `prediction.ipynb` — notebook showing the prediction workflow.
- `experiments.ipynb` — notebook for model experiments.

## Requirements

Python 3.10 or newer is recommended. Install the dependencies from the project directory:

```bash
python -m venv .venv
```

Activate the environment, then run:

```bash
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate it with `.venv\Scripts\Activate.ps1`. On macOS or Linux, use `source .venv/bin/activate`.

## Run the app

From the `annclassification` directory, start the Streamlit app:

```bash
streamlit run app.py
```

Streamlit will print a local URL to open in your browser. Enter the customer's geography, gender, age, credit score, balance, salary, tenure, product count, and account activity details. The app displays a churn probability and classifies probabilities above `0.5` as likely to churn.

Keep `app.py`, `model.h5`, and all three `.pkl` preprocessing files together in the same directory; the app loads these files by relative path.

## Prediction workflow

The app encodes gender and geography, assembles the features, scales them with the saved training scaler, and passes them to the model. The preprocessing artifacts and model must come from the same training run and be used with the same feature order for predictions to be meaningful.

To explore the notebook-based workflow, open `prediction.ipynb` in Jupyter or VS Code. The notebooks are for exploration; `app.py` is the interactive application entry point.

## Dependencies

The main dependencies are TensorFlow, pandas, NumPy, scikit-learn, and TensorBoard. The complete pinned/unpinned dependency list is in [`requirements.txt`](requirements.txt).

## License

No license file is currently included. Contact the project owner before reusing or redistributing this project.
