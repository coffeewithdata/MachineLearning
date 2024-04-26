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
# Binary Classification Algorithms in Machine Learning

Binary classification is a type of supervised learning task where the goal is to categorize instances into one of two classes or categories. Here are some popular binary classification algorithms commonly used in machine learning:

## 1. Logistic Regression
- A simple and widely used algorithm for binary classification.
- Applies the logistic function to a linear combination of input features.
- Outputs probabilities that can be thresholded to make binary predictions.

## 2. Support Vector Machines (SVM)
- A powerful algorithm for both binary and multiclass classification.
- Finds the hyperplane that maximally separates the two classes in the feature space.
- Can use different kernel functions for non-linear separation.

Support Vector Machines (SVMs) are a versatile class of machine learning algorithms used for classification, regression, and outlier detection tasks. They excel in handling complex data or datasets with high dimensionality.

### How SVMs Work:

**1. Finding the Hyperplane:**
   - SVMs aim to find a hyperplane in the data space that best separates data points belonging to different classes. This hyperplane serves as a decision boundary, facilitating the classification of new data points.

**2. Maximizing the Margin:**
   - SVMs don't settle for just any hyperplane; they seek the one with the widest margin between the closest data points from each class. This margin acts as a buffer zone, enhancing the model's ability to generalize well to unseen data.

**3. Support Vectors:**
   - Support vectors are the data points lying closest to the hyperplane on either side. They are crucial for defining the hyperplane and the SVM model itself.

### Advantages of SVMs:

- **Effective in High Dimensions:** SVMs handle high-dimensional data efficiently, making them suitable for complex problems.
- **Good Performance:** They are known for achieving high accuracy on various classification tasks.
- **Memory Efficiency:** SVMs utilize only a subset of training data (support vectors) for classification, making them memory-efficient.

### Limitations of SVMs:

- **Complexity for Large Datasets:** Training SVMs on massive datasets can be computationally expensive.
- **Tuning Required:** SVMs can be sensitive to the choice of kernel function, requiring optimization for optimal performance.

### Conclusion:

Support Vector Machines are a potent tool in the machine learning toolbox, particularly effective for classification tasks involving complex data. Their capability to handle high dimensionality and discern optimal separation boundaries has made them a favored choice across numerous applications. However, practitioners must carefully consider their computational requirements and the selection of kernel functions to maximize SVM effectiveness.

  Sample code link :  https://github.com/coffeewithdata/MachineLearning/blob/main/MLCourse/SVC.ipynb 
  Sample code Link :  https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Code/Day%2013%20SVM.md#predicting-the-test-set-results 

## 3. Decision Trees
- Builds a tree-like structure where each internal node represents a decision based on a feature.
- Can be used for binary classification by assigning classes to leaves.

## 4. Random Forest
- An ensemble method that builds multiple decision trees and combines their predictions.
- Reduces overfitting and increases robustness compared to individual trees.

## 5. Gradient Boosting (e.g., XGBoost, LightGBM)
- An ensemble method that builds trees sequentially, with each tree correcting errors made by the previous ones.
- Often highly effective and widely used for binary classification tasks.

## 6. Naive Bayes
- Based on Bayes' theorem and assumes independence between features.
- Simple and efficient, particularly for text classification tasks.

## 7. K-Nearest Neighbors (KNN)
- Classifies an instance based on the majority class of its k nearest neighbors in the feature space.
- Sensitive to the choice of distance metric and the value of k.

## 8. Neural Networks
- Deep learning models with multiple layers of interconnected nodes (neurons).
- Can be used for binary classification tasks with appropriate architecture and training.

## 9. Adaptive Boosting (AdaBoost)
- An ensemble method that combines weak classifiers to create a strong classifier.
- Gives higher weight to misclassified instances in each iteration.

## 10. Gaussian Mixture Models (GMM)
- A probabilistic model that represents the data as a mixture of Gaussian distributions.
- Can be used for clustering and binary classification tasks.

The choice of algorithm depends on factors such as the nature of the data, interpretability of the model, computational efficiency, and specific task requirements. It's recommended to experiment with multiple algorithms and select the one that performs well on a given problem.

Evaluation metrics like precision, recall, accuracy, and F1 score are used to assess the performance of the model.

Binary classification problems are prevalent in various domains, such as spam detection, fraud detection, medical diagnosis (disease or no disease), sentiment analysis (positive or negative sentiment), and more.
