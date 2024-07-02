# Project Description:
## Overview:
- This project consists of various models implemented using classical Machine Learning algorithms on open source datasets.

## Tech Stack:
- Used Google Colaboratory for implementaion.
- Language: Python.
- Also used various Python libraries like NumPy, Pandas, Matplotlib & Scikit-Learn.

## Algorithms Used:
### Linear Regression:
- In linear regression, we obtain an estimate of the unknown variable (denoted by y; the output of our model) by computing a weighted sum of our known variables (denoted by xᵢ; the inputs) to which we add a bias term.

- <img width="200" alt="image" src="https://user-images.githubusercontent.com/77923668/232190479-f4759ae7-1282-4c66-8635-6be6483f32d2.png"> => <img width="180" alt="image" src="https://user-images.githubusercontent.com/77923668/232190565-c7de94a8-e16a-472f-a810-7691e5228667.png">
- We can obtain the weight matrix using 2 methods:
    1. Normal Equation Method (Linear Algebra approach)
    2. Gradient Descent Method (Using Calculus to  minimize the error function)

### Logistic Regression:
- Logistic regression is a statistical/regression analysis method used when the dependent variable is dichotomous (binary).

- <img width="150" alt="image" src="https://user-images.githubusercontent.com/77923668/232193101-1a0e768b-1bf0-451f-b4aa-0de7b98fc332.png">
- Logistic Regression converts the straight best fit line in linear regression to an S-curve using the sigmoid function.

### Neural Networks:
- Neural networks, also known as artificial neural networks (ANNs), are comprised of a node layers, containing an input layer, one or more hidden layers, and an output layer.
- Each node, or artificial neuron, connects to another and has an associated weight and threshold.
    - If the output of any individual node is above the specified threshold value, that node is activated, sending data to the next layer of the network.
    - Otherwise, no data is passed along to the next layer of the network.
- Neural networks rely on training data to learn and improve their accuracy over time.
- <img width="250" alt="image" src="https://user-images.githubusercontent.com/77923668/232193901-fbf5f1ab-3267-47fd-8456-b589673cdb53.png">

## Files:
- *bin_log_reg_cancerdata.ipynb*
    - Implemented a binary classifier model using logistic Regression on the Breast Cancer dataset.
    - About Dataset:
        - Breast Cancer dataset (imported using scikit-learn).
        - [More Information on the Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
    - Assumed baseline model (y = 1): 62.3% accuracy on test data
    - Training Plot:
        - <img width="250" alt="image" src="https://user-images.githubusercontent.com/77923668/232194475-9d566d9d-a0e1-4b4b-ad93-54c87336702f.png">
    - Accuracy: 0.956 on test set

- *linear_reg_GD_housingdata.ipynb*
    - Implemented a linear regression model using Gradient Descent method on California Housing dataset.
    - About Dataset:
        - California Housing Dataset (available on Google Colab).
        - [More Information on the Dataset](https://developers.google.com/machine-learning/crash-course/california-housing-data-description)
    - Training Plot: 
        - <img width="250" alt="image" src="https://user-images.githubusercontent.com/77923668/232196110-e47bfe82-8250-4aa5-b8af-96b468c6374b.png">
    - Accuracy: 0.61 on test set using R2 Error method

- *linear_reg_normal_eq.ipynb*
    - Implemented a linear regression model using Normal Equation method on California Housing dataset.
    - About Dataset:
        - California Housing Dataset (available on Google Colab).
        - [More Information on the Dataset](https://developers.google.com/machine-learning/crash-course/california-housing-data-description)

- *neural_net_iris_classifier.ipynb* 
    - Implemented neural networks (with 3 layers) to perform multi-class classification on Iris dataset.
    - About Dataset:
        - Iris Dataset (imported using scikit-learn)
        - [More Information on the Dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set)
    - Error Plot: 
        - <img width="250" alt="image" src="https://user-images.githubusercontent.com/77923668/232195942-e532b95e-5e81-4d4c-b8bc-9ebb087cbc72.png">
    - Accuracy: 1.0 (train), 0.956 (test)
