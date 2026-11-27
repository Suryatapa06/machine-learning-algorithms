# Machine Learning Algorithms

Python implementations of core machine learning algorithms with scikit-learn and
TensorFlow/Keras: clustering, decision trees, neural networks and support vector machines.
The notebooks follow standard course and tutorial exercises, with sources listed below.
My original research work is in
[medical-image-retrieval-se-resnet](https://github.com/Suryatapa06/medical-image-retrieval-se-resnet).

| Notebook | Algorithm | Task | Dataset (not included)
|---|---|---|---|---|
| `01_k_means_clustering.ipynb` | K-Means, elbow method | Customer segmentation | Mall Customers (Kaggle)
| `02_decision_tree_loan_default.ipynb` | Decision tree | Loan repayment prediction | LendingClub `loan_data.csv`
| `03_ann_customer_churn.ipynb` | Artificial neural network (Keras) | Bank customer churn | Churn_Modelling (Kaggle)
| `04_svm_diabetes_prediction.ipynb` | Support vector machine | Diabetes prediction, model saved with pickle | PIMA Indians Diabetes

## Topics covered
- Data loading, exploration and visualisation (pandas, matplotlib, seaborn)
- Preprocessing: categorical encoding, train/test split, feature scaling
- Training and evaluating models: clustering (WCSS), classification
  (accuracy, confusion matrix, classification report), neural network training
- Saving a trained model and predicting on a new input

## Running
Datasets are not included. Download each from its source, place the CSV next to the
notebook (or fix the path), then:
```bash
pip install -r requirements.txt
jupyter notebook
```
