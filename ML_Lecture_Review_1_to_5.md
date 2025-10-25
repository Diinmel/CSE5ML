This response draws on the provided lecture excerpts (Lectures 01
through 05) and revision guidelines. The answers are written in simple
English, focusing on key concepts to be easy to remember.

------------------------------------------------------------------------

### 1. What is machine learning?

**Machine learning (ML) is about letting computers learn from data to
make better predictions**.

We design algorithms that use data to programmatically learn a model.
The computer improves its performance at a task based on its experience
(which is the data). The rules are not hard-coded; the model adapts as
it gets more data.

### 2. List several machine learning tasks

The broad machine learning tasks are:

1.  **Classification**.
2.  **Regression**.
3.  **Clustering**.
4.  **Dimensionality Reduction**.

### 3. What is the difference between classification and regression?

Both are supervised learning problems:

-   **Classification** predicts categories or classes. The goal is to
    predict which of two or more classes an input belongs to, like
    classifying spam or finding a separator.
-   **Regression** predicts a continuous, **real-valued output**. The
    goal is to model the relationship between input and output, like
    fitting a line or curve to predict house prices.

### 4. What is the difference between supervised learning and unsupervised learning?

This difference is based on the data used:

-   **Supervised Learning** requires **labeled data**. The data already
    has target labels (or correct answers) that the model learns from.
-   **Unsupervised Learning** uses data that **does not have labels**.
    The goal is to find hidden patterns or group inputs based on their
    similarities, like in Clustering.

### 5. What is the aim of the linear regression?

The aim is to **fit the best straight line or hyperplane** to the data.

It is used to model the relationship between input variables and a
real-valued output. We want to find the parameters that minimize the
deviations from this predictor line.

### 6. What is the loss function of the linear regression?

The standard loss function used in linear regression is the **Mean
Square Error (MSE)**.

This process is often called **Least Squares Regression**, where the
goal is to minimize the squared difference between the predicted and
actual values.

### 7. What is the underfitting and overfitting?

Both terms describe issues with how well a model generalizes:

-   **Overfitting** happens when the model learns the training data
    *too* well, including the noise and random variations. It performs
    poorly on new, unseen data.
-   **Underfitting** happens when the model is too simple and cannot
    capture the main trend of the data. It performs poorly even on the
    training data.

### 8. What is the general idea of binary classification?

The general idea is to **sort items into exactly two possible groups**
(classes), usually labeled as 0 and 1, or Positive and Negative.

The model tries to learn a rule (a linear or non-linear separator) that
differentiates between these two classes.

### 9. What are logistic regression and its loss function?

-   **Logistic Regression** is a model primarily used for **binary
    classification**. It models the probability that an input belongs to
    a certain class.
-   The loss function often associated with Logistic Regression, which
    measures the difference between two distributions, is **Cross
    Entropy**.

### 10. What is the difference between GD and SGD?

These are optimization algorithms (Gradient Descent methods):

-   **GD (Gradient Descent)** calculates the gradient (slope) using
    **all the training data** in every iteration. This is
    computationally heavy but stable.
-   **SGD (Stochastic Gradient Descent)** calculates the gradient using
    only a **small batch of data** (or a single sample). This is faster
    and generally more noisy, but helps escape local minima.

### 11. What is the confusion matrix?

A **Confusion Matrix helps us check how well our classification model
works**.

It is an $N \times N$ matrix used to evaluate a classification model's
performance by comparing the **actual target values** with the
**predicted values**. For binary problems, it gives us four key values:
True Positive (TP), True Negative (TN), False Positive (FP - Type 1
error), and False Negative (FN - Type 2 error).

### 12. What is the key idea of SVM?

The key idea of **Support Vector Machines (SVM)** is to find the
separating line (hyperplane) that **maximizes the margin** (the largest
possible gap) between the different classes.

This is called the "optimal margin classifier" because it is the most
stable classification boundary.

### 13. What is the loss function of SVM?

The loss function specific to SVM is not explicitly named in the
provided materials but is part of the optimization process used to
maximize the margin.

*The SVM uses optimization to find the maximum margin hyperplane.*

### 14. What is the difference between hard-SVM and soft-SVM?

This difference relates to how the SVM handles the data boundaries:

-   **Hard SVM** requires the data to be perfectly linearly separable;
    it finds a maximum margin hyperplane with **no misclassifications**
    allowed within the margin.
-   **Soft SVM** is more flexible and is used when data is messy or
    non-linearly separable. It **allows a few misclassifications**
    (using slack variables) to still achieve a wide margin and a better
    generalizing model.

### 15. What is the usage of the softmax function?

The **Softmax function is used in Multi-class Classification**.

Its main job is to **convert the model's scores into probabilities** for
each class. The output values are between 0 and 1 and they all sum up to
1.

### 16. What are the key steps of K-means clustering?

K-means clustering is an unsupervised learning method. The key steps
are:

1.  Choose the number of clusters, $k$.
2.  Select $k$ **random points** from the data as initial cluster
    centroids.
3.  **Assign all data points** to the closest cluster centroid.
4.  **Recompute the centroids** (move the centers) based on the average
    of the points in the new cluster.
5.  **Repeat steps 3 and 4** until the centroids stop changing or a
    maximum number of iterations is reached.

### 17. What is the difference between L1 and L2 regularizations?

Both L1 and L2 regularization are methods used to prevent overfitting by
adding a penalty term to the loss function:

-   **L1 Regularization** (Lasso) uses the **sum of the absolute
    values** of the weights. This penalty can force some weights to
    become exactly zero, making it useful for **feature selection**.
-   **L2 Regularization** (Ridge) uses the **sum of the squared values**
    of the weights. This penalty shrinks the weights toward zero but
    usually does not set them exactly to zero.
