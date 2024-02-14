# Binary Classification

Binary classification is a type of classification problem in machine learning where the goal is to categorize instances into one of two classes or categories. The two classes are often referred to as the positive class (denoted as 1) and the negative class (denoted as 0). The objective is to build a model that can learn the patterns and features in the data to accurately predict the class of new, unseen instances.

Here are some key concepts and components of binary classification:

1. **Positive Class (1) and Negative Class (0):** In binary classification, the two classes are typically denoted as 1 and 0. The positive class is the class of interest, and the negative class is the alternative.

2. **True Positive (TP):** Instances that are correctly predicted as belonging to the positive class.

3. **True Negative (TN):** Instances that are correctly predicted as belonging to the negative class.

4. **False Positive (FP):** Instances that are incorrectly predicted as belonging to the positive class (Type I error).

5. **False Negative (FN):** Instances that are incorrectly predicted as belonging to the negative class (Type II error).

6. **Accuracy:** The overall correctness of the model, calculated as (TP + TN) / (TP + TN + FP + FN).

7. **Precision (Positive Predictive Value):** The proportion of instances predicted as positive that are actually positive, calculated as TP / (TP + FP).

8. **Recall (Sensitivity, True Positive Rate):** The proportion of actual positive instances that are correctly predicted, calculated as TP / (TP + FN).

9. **Specificity (True Negative Rate):** The proportion of actual negative instances that are correctly predicted, calculated as TN / (TN + FP).

10. **F1 Score:** The harmonic mean of precision and recall, calculated as 2 * (Precision * Recall) / (Precision + Recall).

Common algorithms used for binary classification :
1. logistic regression 
2. support vector machines
3. decision trees
4. random forests 
5. neural networks. 
6. 

Evaluation metrics like precision, recall, accuracy, and F1 score are used to assess the performance of the model.

Binary classification problems are prevalent in various domains, such as spam detection, fraud detection, medical diagnosis (disease or no disease), sentiment analysis (positive or negative sentiment), and more.