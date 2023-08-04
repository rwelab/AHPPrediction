# AHP Prediction

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Description

Welcome to the AHPPrediction repository! 
This project is entitled "Reducing diagnostic delays in Acute Hepatic Porphyria using electronic health records data and machine learning: a multicenter development and validation study". 



 **Introduction**: 

KEY POINTS

Question

Can machine learning help identify undiagnosed patients with Acute Hepatic Porphyria (AHP), a group of rare diseases?

Findings

Using electronic health records (EHR) data from two centers we developed models to predict: 1) who will be referred for AHP testing, and 2) who will test positive. The best models achieved 89-93% accuracy on the test set. These models appeared capable of recognizing 71% of the cases earlier than their true diagnosis date, reducing diagnostic delays by an average of 1.2 years.

Meaning

Machine learning models trained using EHR data can help reduce diagnostic delays in rare diseases like AHP.

 **Features**: 
Importance

Acute Hepatic Porphyria (AHP) is a group of rare but treatable conditions associated with diagnostic delays of fifteen years on average. The advent of electronic health records (EHR) data and machine learning (ML) may help improve the timely recognition of rare diseases like AHP. However, prediction models can be difficult to train given the limited case numbers, unstructured EHR data, and selection biases intrinsic to healthcare delivery.

Objective

To train and characterize models for identifying patients with AHP.


 **Installation**: 
Check, reuirements.txt and installation and dependencies.txt files





## Installation

Here are the steps to install and set up the project locally:

1. Clone the repository: `https://github.com/rwelab/AHPPrediction.git`
2. Navigate to the project directory: `cd AHPPrediction`
3. Install dependencies: `pip install` or `yarn install` (depending on the package manager you use)

## Usage


```bash
# Run the script
python Data_Preprocessing.py.py #preprocessing data from electronic health records
python Feature Selection.py #Feature Selection using various algorithms
python Porphyria_Classifier.py #generating prediction models
python Referral_Diagnosis_Model.py #Applying the best performing models to new cohort for prediction
```

## Contributing

We welcome contributions from the community! If you want to contribute to the project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix: `git checkout -b feature/your-feature-name`.
3. Commit your changes: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature/your-feature-name`.
5. Submit a pull request to the `main` branch.

Please ensure that your code follows the project's coding style and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

