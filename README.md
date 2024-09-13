
# AI based maize farming advisory rystem & rainy season onset prediction
**Author**: Jemal S. Ahmed  

**Contact Information**:  
- Email: JemalSeid.Ahmed@santannapisa.it 
- LinkedIn: https://www.linkedin.com/in/jemal-seida/ 

---

## Project Overview

This repository contains multiple Jupyter notebooks focused on integrating AI, machine learning, and climate information to support agricultural decision-making. The primary objectives are to provide **data-driven agronomic recommendations** and to predict the **onset of the rainy season** — a critical event for planting decisions.

The project revolves around two key themes:
1. **SMART Maize Farming Advisory**: A system that combines weather forecasts with AI to give personalized advice to maize farmers.
2. **Rainy Season Onset Prediction**: Implementing traditional machine learning and advanced deep learning models (such as LSTM) to predict the start of the rainy season using historical climate data.

This system is designed to help farmers make informed decisions regarding planting dates, input usage, and other agronomic practices, especially in regions where climate variability can significantly impact crop yields.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Notebooks Overview](#notebooks-overview)
- [Requirements](#requirements)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Author Information](#author-information)

---

## Notebooks Overview

### 1. **[SMART Maize Farming Advisory System Using Weather Forecasts and AI](AI_Based_farmAdvisory_SMT_I.ipynb)**

This notebook implements a **SMART farming advisory system** using weather forecasts and AI-based optimization techniques. The system provides:
- **Weather-based recommendations**: Suggests planting dates, irrigation schedules, and fertilizer applications based on weather predictions.
- **AI-driven insights**: Machine learning models tailor advice specific to each field based on climatic conditions and soil types.

### 2. **[LSTM-Based Rainy Season Onset Detection](Onset4PD_LSTM_model.ipynb)**

This notebook focuses on **detecting the rainy season onset** using Long Short-Term Memory (LSTM) networks. Key features include:
- **Data preparation**: Climate data preprocessing and augmentation for better model performance.
- **LSTM model**: A deep learning approach to predict when the rainy season will start, leveraging time-series patterns.
- **Model tuning**: Optimization techniques like early stopping, bidirectional LSTMs, and learning rate scheduling to enhance model accuracy.

### 3. **[Predicting Rainy Season Onset Using Machine Learning](Rainfall_onset_PD.ipynb)**

This notebook provides two options for predicting the onset of the rainy season:
- **Option I: Traditional Machine Learning Approach**: Involves preprocessing, feature engineering, and using classic ML models like Random Forests to predict onset.
- **Option II: LSTM Approach**: An advanced deep learning method using LSTMs to handle sequential climate data and make predictions on rainy season onset.

Both approaches aim to provide accurate and actionable information to improve planting decisions.

---

## Requirements

To run the notebooks in this repository, install the following dependencies:

- **TensorFlow/Keras**: For building and training the LSTM models.
- **Scikit-learn**: For traditional machine learning tasks.
- **Pandas**: For data manipulation and preprocessing.
- **NumPy**: For numerical operations and array handling.
- **Matplotlib/Seaborn**: For data visualization and performance plots.

You can install the required packages by running:

```bash
pip install -r requirements.txt
```

---

## Usage

1. **SMART Maize Advisory System**:
   - Open and execute the `AI_Based_farmAdvisory_SMT_I.ipynb` notebook.
   - Customize it with your weather forecast data and regional maize farming details.
   - This notebook will generate tailored recommendations for maize farmers based on input weather conditions and agronomic data.

2. **Rainy Season Onset Detection (LSTM)**:
   - Open `Onset4PD_LSTM_model.ipynb` to train and predict rainy season onset using an LSTM model.
   - Ensure that your climate data is properly preprocessed and the LSTM parameters are adjusted for your specific use case.

3. **Rainy Season Onset Detection (ML & Randomforest)**:
   - Use the `Rainfall_onset_PD.ipynb` notebook to explore two approaches:
     - Option I: Traditional machine learning methods.
     - Option II: An Randomforest-based approach for onset prediction.
   - This notebook is flexible and allows you to compare the performance of traditional approach with different ML models.

---

