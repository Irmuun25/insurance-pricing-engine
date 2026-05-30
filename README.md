```markdown
# 🏥 Actuarial Health Insurance Pricing Engine

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Data%20Pipeline-F7931E.svg)

## 📋 Project Overview
This project bridges deep learning engineering with quantitative risk management. It implements an end-to-end predictive pipeline that determines individualized health insurance premiums based on underlying medical risk. 

By capturing complex, non-linear interactions within demographic and health behavior data, a custom PyTorch feedforward neural network estimates expected annual medical claims. Traditional actuarial premium principles—specifically the **Expected Value Principle**—are then programmatically applied to incorporate a risk loading margin for financial insolvency safety.

The production model is deployed via a live, interactive **Streamlit web application**, allowing users to perform real-time risk evaluations and premium calculations.

---

## ⚙️ Architecture & Methodology

**Data Flow:**
`[User Input UI]` ➔ `[StandardScaler]` ➔ `[PyTorch Deep Regression Model (64 ➔ 32 ➔ 1)]` ➔ `[Expected Annual Claims E(X)]` ➔ `[Actuarial Loading: E(X) * (1 + θ)]` ➔ `[Monthly Premium Output]`

1. **Feature Interaction Extraction:** Standard linear models struggle with compound medical risks (e.g., the compounding risk of high BMI combined with a positive tobacco usage profile). This architecture leverages fully connected layers utilizing `nn.ReLU()` activation functions, batch normalization, and dropout layers to effectively model these non-linear feature crossings.
2. **Data Pipeline & Preprocessing:** Categorical variables (Sex, Smoker Status, Region) are mapped using one-hot encoding. Continuous variables (Age, BMI) are passed through a Scikit-Learn `StandardScaler` to ensure stable gradients and smooth convergence during training.
3. **Actuarial Premium Principles:** To safeguard against claim volatility, the application uses the actuarial Expected Value Principle:
   
   **$Premium = E(X) \cdot (1 + \theta)$**
   
   Where $E(X)$ represents the pure expected claims output from the PyTorch model, and $\theta$ is the safety loading factor (set at 20%) to account for administrative expenses and statistical variance.

---

## 🚀 Live Application
The model is actively deployed and can be tested interactively.
* **Streamlit App:** *(Insert your Streamlit share link here once deployed)*

**Key Features:**
* **Interactive Risk Modeling:** Sidebars allowing real-time tuning of continuous values (Age, BMI, dependants) and categorical risks.
* **Live Actuarial Valuation:** Instant calculation breakdown showing the Pure Expected Claims, the Added Safety Margin, and the final scheduled Monthly Premium.
* **Production Pipeline:** Seamless synchronization between the pre-trained neural network weights (`model.pth`) and the training feature scaler (`scaler.pkl`).

---

## 💻 Local Installation & Usage

### Prerequisites
* Python 3.9+
* Git

### Setup
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/insurance-pricing-engine.git](https://github.com/your-username/insurance-pricing-engine.git)
   cd insurance-pricing-engine

```

2. Install the required dependencies:
```bash
pip install -r requirements.txt

```


3. (Optional) Retrain the model:
*If you want to pull the latest dataset from Kaggle and generate new weights.*
```bash
python train.py

```


4. Run the Streamlit web application:
```bash
streamlit run app.py

```



---

## 📁 Repository Structure

* `app.py`: The main Streamlit application script containing the UI and inference logic.
* `train.py`: The model training pipeline, including Kaggle data fetching and preprocessing.
* `model.pth`: The saved state dictionary (weights) of the trained PyTorch neural network.
* `scaler.pkl`: The serialized Scikit-Learn StandardScaler fitted to the training data.
* `requirements.txt`: Python package dependencies.
* `.gitignore`: Standard Git ignore file for Python/Streamlit environments.

---

## 👨‍💻 Author

**Munkh-Irmuun Munkhbat**

```

```
