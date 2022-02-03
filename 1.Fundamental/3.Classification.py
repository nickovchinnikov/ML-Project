# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, Image, display
from sklearn.base import BaseEstimator, clone
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def display_markdown(markdown: str) -> None:
    """Display markdown on the output

    Args:
        markdown (str): markdown data string
    """
    display(Markdown(markdown))


def display_image(filepath: str) -> None:
    """Approach for draw an image in Python Interactive window and for notebook
    I tired to fight against images outputs bugs, and just use python to fix

    Args:
        filepath (str): image path
    """
    display(Image(filename=filepath))


# %% [markdown]
"""
# Classification

## MNIST

> By default Scikit-Learn caches downloaded datasets
  in a directory called `$HOME/scikit_learn_data`.
"""

# %%
mnist = fetch_openml("mnist_784", version=1)
mnist.keys()

# %% [markdown]
"""
* A `DESCR` key describing the dataset
* A `data` key containing an array with one row per instance and one column per feature
* A `target` key containing an array with the labels
"""

# %%
display_markdown(mnist.DESCR)

# %%
X, y = mnist.data, mnist.target

X.shape, y.shape

# %% [markdown]
"""
There are 70,000 images, and each image has 784 features. This is because each image
is 28×28 pixels, and each feature simply represents one pixel’s intensity, from 0
(white) to 255 (black). Let’s take a peek at one digit from the dataset. All you need to
do is grab an instance’s feature vector, reshape it to a 28×28 array, and display it using
Matplotlib’s `imshow()` function:
"""

# %%
some_digit = X[0]

some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap=mpl.cm.binary, interpolation="nearest")

plt.axis("off")
plt.show(block=True)

# %%
y[0]

# %%
y = y.astype(np.uint8)
y[0]

# %% [markdown]
"""
### Several images from dataset
"""

# %%
size = 10

fig, axis = plt.subplots(size, size, figsize=(15, 15), facecolor='white')

for i in range(size):
    for j in range(size):
        shift = i * size
        some_digit = X[shift + j]
        some_digit_image = some_digit.reshape(28, 28)
        ax = axis[i][j]
        ax.imshow(some_digit_image, cmap=mpl.cm.binary,
                  interpolation="nearest")
        ax.axis("off")

plt.show(block=True)

# %% [markdown]
"""
But wait! You should always create a test set and set it aside before inspecting the data
closely. The MNIST dataset is actually already split into a training set
(the first 60000 images) and a test set (the last 10,000 images):
"""

# %%
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# %% [markdown]
"""
## Training a Binary Classifier
### Setup array of 5 in labels
"""

# %%
y_train_5 = y_train == 5  # True for all 5s, False for all other digits.
y_test_5 = y_test == 5
y_train_5[:5]

# %%
y_test_5[5:10]

# %% [markdown]
"""
Okay, now let’s pick a classifier and train it. A good place to start is with a
Stochastic Gradient Descent (SGD) classifier, using Scikit-Learn’s `SGDClassifier` class.
This classifier has the advantage of being capable of handling very large datasets
efficiently. This is in part because SGD deals with training instances independently,
one at a time (which also makes SGD well suited for online learning), as we will see later.
Let’s create an `SGDClassifier` and train it on the whole training set
"""

# %%
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# %% [markdown]
"""
> The `SGDClassifier` relies on randomness during training (hence
the name “stochastic”). If you want reproducible results, you
should set the `random_state` parameter.

#### Problem with first digit, probably difference between my laptop and author env
"""

# %%
sgd_clf.predict([some_digit])

# %% [markdown]
"""
#### But for 11's item it works fine
"""

# %%
sgd_clf.predict([X[11]])

# %% [markdown]
"""
## Performance Measures
### Measuring Accuracy Using Cross-Validation
"""

# %%

skfolds = StratifiedKFold(n_splits=3, shuffle=False)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)

    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]

    X_test_folds = X_train[test_index]
    y_test_folds = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_predict = clone_clf.predict(X_test_folds)
    n_correct = sum(y_predict == y_test_folds)

    print(n_correct / len(y_predict))

# %% [markdown]
"""
The `StratifiedKFold` class performs stratified sampling (as explained in Chapter 2)
to produce folds that contain a representative ratio of each class. At each iteration the
code creates a clone of the classifier, trains that clone on the training folds, and makes
predictions on the test fold. Then it counts the number of correct predictions and
outputs the ratio of correct predictions

Let’s use the `cross_val_score()` function to evaluate your `SGDClassifier` model 
using K-fold cross-validation, with three folds. Remember that K-fold cross-validation 
means splitting the training set into K-folds (in this case, three), then making 
predictions and evaluating them on each fold using a model trained on the remaining folds
"""

# %%
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")

# %% [markdown]
"""
#### very dumb classifier that just classifies every single image in the “not-5” class
"""
# %%


class Never5Classifier(BaseEstimator):
    """Dummy class that never detect 5

    Args:
        BaseEstimator ([BaseEstimator]): base estimator parent class
    """

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> None:
        """Fit function that do nothing

        Args:
            X (np.ndarray): values for training that need to fit
            y (np.ndarray, optional): values for prediction test. Defaults to None.
        """
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predictor method

        Args:
            X (np.ndarray): Data for predictor

        Returns:
            np.ndarray: Prediction result data
        """
        return np.zeros((len(X), 1), dtype=bool)


never_5_clf = Never5Classifier()

cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")

# %% [markdown]
"""
> That’s right, it has over 90% accuracy! This is simply because only about 10% of the
images are 5s, so if you always guess that an image is not a 5, you will be right about
90% of the time. Beats Nostradamus.
"""

# %% [markdown]
"""
### Confusion Matrix

A much better way to evaluate the performance of a classifier is to look at the confusion matrix.
The general idea is to count the number of times instances of class A are classified as class B.
For example, to know the number of times the classifier confused images of 5s with 3s, you would
look in the $5^{th}$ row and $3^{rd}$ column of the confusion matrix.

To compute the confusion matrix, you first need to have a set of predictions, so they can be
compared to the actual targets. You could make predictions on the test set, but let’s keep it
untouched for now (remember that you want to use the test set only at the very end of your project,
once you have a classifier that you are ready to launch).
Instead, you can use the `cross_val_predict()` function:
"""

# %%
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

# %% [markdown]
"""
Now you are ready to get the confusion matrix using the `confusion_matrix()` function. Just pass it the target classes (`y_train_5`) and the predicted classes (`y_train_pred`)
"""

# %%
confusion_matrix(y_train_5, y_train_pred)

# %% [markdown]
"""
Each row in a confusion matrix represents an actual class, while each column represents a
predicted class. The first row of this matrix considers non-5 images (the negative class):
53,057 of them were correctly classified as non-5s (they are called true negatives),
while the remaining 1,522 were wrongly classified as 5s (false positives). The second row
considers the images of 5s (the positive class): 1,325 were wrongly classified as non-5s
(false negatives), while the remaining 4,096 were correctly classified as 5s (true positives).
A perfect classifier would have only true positives and true negatives, so its confusion
matrix would have nonzero values only on its main diagonal (top left to bottom right):
"""
# %%
y_train_perfect_predictions = y_train_5  # pretend we reached perfection

confusion_matrix(y_train_5, y_train_perfect_predictions)
# %%
display_image("./assets/2022-01-31-18-06-10.png")

# %% [markdown]
"""
`TP` is the number of true positives, and `FP` is the number of false positives.
A trivial way to have perfect precision is to make one single positive
prediction and ensure it is correct (precision = 1/1 = 100%).
This would not be very useful since the classifier would ignore all but one
positive instance. So precision is typically used along with another metric named
recall, also called sensitivity or true positive rate (`TPR`): this is the ratio
of positive instances that are correctly detected by the classifier.

`FN` is of course the number of false negatives.
"""

# %%
display_image("./assets/2022-01-31-18-33-43.png")
display_image("./assets/2022-01-31-18-35-19.png")

# %% [markdown]
"""
### Precision and Recall

Scikit-Learn provides several functions to compute classifier metrics,
including **precision** and **recall**:
"""

# %%
precision_score(y_train_5, y_train_pred)

# %%
recall_score(y_train_5, y_train_pred)

# %% [markdown]
"""
It is often convenient to combine precision and recall into a single 
metric called the $F_1$ score
"""

# %%
display_image("./assets/2022-01-31-18-40-32.png")

# %% [markdown]
"""
[Harmonic mean](https://en.wikipedia.org/wiki/Harmonic_mean)
"""

# %%
display_image("./assets/2022-01-31-18-42-03.png")

# %%
f1_score(y_train_5, y_train_pred)

# %%
display_image("./assets/2022-01-31-18-45-10.png")

# %% [markdown]
"""
Scikit-Learn does not let you set the threshold directly, but it does give
you access to the decision scores that it uses to make predictions. Instead of calling the classifier’s `predict()` method, you can call its
`decision_function()` method, which returns a score for each instance,
and then make predictions based on those scores using any threshold you want:
"""

# %%
y_scores = sgd_clf.decision_function([some_digit])

y_scores = abs(y_scores)

y_scores

# %%
threshold = 0

y_some_digit_pred = y_scores > threshold

y_some_digit_pred

# %%
threshold = 8000

y_some_digit_pred = y_scores > threshold

y_some_digit_pred

# %% [markdown]
"""
Now how do you decide which threshold to use? For this you will first need to get the
scores of all instances in the training set using the `cross_val_predict()` function
again, but this time specifying that you want it to return decision scores instead of predictions:
"""

# %%
y_scores = cross_val_predict(
    sgd_clf, X_train, y_train_5, cv=3, method="decision_function"
)

# %%


def plot_precision_recall_vs_threshold(
        precisions: np.ndarray,
        recalls: np.ndarray,
        threshold: int) -> None:
    """Plot precision and recall vs threshold

    Args:
        precisions (np.ndarray): precisions data
        recalls (np.ndarray): recalls data
        threshold (int): treshold that we draw on x axe
    """
    # On black background I see nothing, make it contrast
    plt.figure(facecolor='white')
    plt.plot(threshold, precisions[:-1], "b--", label="Precision")
    plt.plot(threshold, recalls[:-1], "g-", label="Recall")
    plt.title("Tresholds")
    plt.grid()
    plt.legend()
    plt.show()


# %%
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

# %%
plt.plot(recalls[:-1], precisions[:-1], "b-")
plt.grid()
plt.show()

# %%
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]

y_train_pred_90 = y_scores >= threshold_90_precision

y_train_pred_90

# %%
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]

y_train_pred_90 = y_scores >= threshold_90_precision

precision_score(y_train_5, y_train_pred_90)

# %%
recall_score(y_train_5, y_train_pred_90)

# %% [markdown]
"""
Great, you have a 90% precision classifier! As you can see, it is fairly easy to create a
classifier with virtually any precision you want: just set a high enough threshold, and
you’re done. Hmm, not so fast. A high-precision classifier is not very useful if its
recall is too low!

**If someone says “let’s reach 99% precision,” you should ask, “at
what recall?”**
"""
# %% [markdown]
"""
## The ROC Curve

The *receiver operating characteristic* (ROC) curve is another common tool used with
binary classifiers. It is very similar to the precision/recall curve, but instead of 
plotting precision versus recall, the ROC curve plots the *true positive rate* (another name
for recall) against the *false positive rate*. The FPR is the ratio of negative instances that
are incorrectly classified as positive. It is equal to one minus the *true negative rate*,
which is the ratio of negative instances that are correctly classified as negative. The
TNR is also called *specificity*. Hence the ROC curve plots *sensitivity* (recall) versus
1 – *specificity*.
To plot the ROC curve, you first need to compute the TPR and FPR for various threshold
values, using the `roc_curve()` function:
"""
# %%
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, label: str = None):
    """Plot receiver operating characteristic (ROG) curve

    Args:
        fpr (np.ndarray): false positive rate
        tpr (np.ndarray): true positive rate (recall)
        label (str, optional): plot label. Defaults to None.
    """
    # On black background I see nothing, make it contrast
    plt.figure(facecolor='white')
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')  # dashed diagonal
    plt.grid()
    plt.xlabel('False positive rate (FPR)')
    plt.ylabel('True positive rate (recall)')


plot_roc_curve(fpr, tpr)
plt.show()

# %% [markdown]
"""
Once again there is a tradeoff: the higher the recall (TPR), the more false positives
(FPR) the classifier produces. The dotted line represents the ROC curve of a purely
random classifier; a good classifier stays as far away from that line as possible (toward
the top-left corner).
One way to compare classifiers is to measure the *area under the curve* (AUC).
A perfect classifier will have a ROC AUC equal to 1, whereas a purely random classifier will
have a ROC AUC equal to 0.5. Scikit-Learn provides a function to compute the ROC
AUC:
"""

# %%
roc_auc_score(y_train_5, y_scores)

# %% [markdown]
"""
Let’s train a `RandomForestClassifier` and compare its ROC curve and ROC AUC
score to the `SGDClassifier`. First, you need to get scores for each instance in the
training set. But due to the way it works (see Chapter 7), the `RandomForestClassifier`
class does not have a `decision_function()` method. Instead it has a `predict_proba()`
method. Scikit-Learn classifiers generally have one or the other. The `predict_proba()`
method returns an array containing a row per instance and a column per class, each
containing the probability that the given instance belongs to the given class
(e.g., 70% chance that the image represents a 5):
"""

# %%
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(
    forest_clf, X_train, y_train_5, cv=3, method='predict_proba')

y_probas_forest
# %% [markdown]
"""
But to plot a ROC curve, you need scores, not probabilities. A simple solution is to
use the positive class’s probability as the score:
"""
# %%
y_scores_forest = y_probas_forest[:, 1]  # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(
    y_train_5, y_scores_forest)

# %% [markdown]
"""
Now you are ready to plot the ROC curve. It is useful to plot the first ROC curve as
well to see how they compare
"""

# %%
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.plot(fpr, tpr, "g:", label="SGD")
plt.legend(loc="lower right")
plt.show()

# %% [markdown]
"""
As you can see in plot, the `RandomForestClassifier` ROC curve looks much
better than the `SGDClassifier`: it comes much closer to the top-left corner. As a
result, its ROC AUC score is also significantly better:
"""

# %%
roc_auc_score(y_train_5, y_scores_forest)

# %% [markdown]
"""
Try measuring the precision and recall scores
"""
# %%
y_train_pred_forest = cross_val_predict(
    forest_clf, X_train, y_train_5, cv=3)

# %%
precision_score(y_train_5, y_train_pred_forest)

# %%
recall_score(y_train_5, y_train_pred_forest)

# %% [markdown]
"""
Or I can make banary just to setup the treshold and then call scores methods
Should be the same result
"""
# %%
y_train_pred_forest_bin = y_scores_forest > 0.7

precision_score(y_train_5, y_train_pred_forest_bin)

# %%
recall_score(y_train_5, y_train_pred_forest)

# %% [markdown]
"""
## Multiclass Classification

Scikit-Learn detects when you try to use a binary classification algorithm
for a multiclass classification task, and it automatically runs OvA (except
for SVM classifiers for which it uses OvO).
Let’s try this with the `SGDClassifier`
"""
# %%
sgd_clf.fit(X_train, y_train)  # y_train, not y_train_5
# %%
# Fail to classify, as before
sgd_clf.predict([some_digit]), y[0]
# %% [markdown]
"""
#### But for 11's item it should works fine
"""
# %%
sgd_clf.predict([X[11]]), y[11]
# %% [markdown]
"""
That was easy! This code trains the `SGDClassifier` on the training set using
the original target classes from 0 to 9 (`y_train`), instead of the 5-versus-all
target classes (`y_train_5`). Then it makes a prediction (a correct one in this
case). Under the hood, Scikit-Learn actually trained 10 binary classifiers,
got their decision scores for the image, and selected the class with the
highest score. To see that this is indeed the case, you can call the
`decision_function()` method.
Instead of returning just one score per instance, it now returns 10 scores,
one per class:
"""

# %%
some_digit_scores = sgd_clf.decision_function([some_digit])
some_digit_scores

# %% [markdown]
"""
The highest score is indeed the one corresponding to class 3, error is still here
Example with match bellow
"""
# %%
some_digit_scores2 = sgd_clf.decision_function([X[11]])
some_digit_scores2
# %%
np.argmax(some_digit_scores2)
# %%
classes = sgd_clf.classes_
classes, classes[5]
# %% [markdown]
"""
When a classifier is trained, it stores the list of target classes in its
`classes_` attribute, ordered by value. In this case, the index of each
class in the `classes_` array conveniently matches the class itself
(e.g., the class at index 5 happens to be class 5), but in general you
won’t be so lucky

If you want to force `ScikitLearn` to use one-versus-one or one-versus-all,
you can use the `OneVsOneClassifier` or `OneVsRestClassifier` classes.
Simply create an instance and pass a binary classifier to its constructor.
For example, this code creates a multiclass classifier using the OvO strategy,
based on a `SGDClassifier`
"""
# %%
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, y_train)
# %% [markdown]
"""
Changes of strategy fix the prediction problem! Both digits is classified right!
"""
# %%
ovo_clf.predict([some_digit]), ovo_clf.predict([X[11]])
# %%
len(ovo_clf.estimators_)
# %% [markdown]
"""
Training a `RandomForestClassifier` is just as easy:
"""
# %%
forest_clf.fit(X_train, y_train)
# %%
forest_clf.predict([some_digit])
# %%
forest_clf.predict_proba([some_digit])
# %% [markdown]
"""
Now of course you want to evaluate these classifiers. As usual, you want to use 
cross-validation. Let’s evaluate the `SGDClassifier`’s accuracy using the 
`cross_val_score()` function:
"""
# %%
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring='accuracy')
# %% [markdown]
"""
It gets over 84% on all test folds. If you used a random classifier, you would get 10%
accuracy, so this is not such a bad score, but you can still do much better.
For example, simply scaling the inputs (as discussed in Chapter 2)
increases accuracy above 89%:
"""
# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# %%
y_train_score = cross_val_score(
    sgd_clf, X_train_scaled, y_train, cv=3, scoring='accuracy')

# %%
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
# %% [markdown]
"""
## Error Analysis

First, you can look at the confusion matrix. You need to make predictions using the
`cross_val_predict()` function, then call the `confusion_matrix()` function, just like
you did earlier:
"""
# %%
len(y_train), len(y_train_pred)
# %%
y_train_pred
# %%
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx
# %% [markdown]
"""
You can run any job in parallel mode, just use `n_jobs` parameter
-1 means using all processors. Sometimes it can push performance a lot
"""
# %%
y_train_pred_parallel_jobs = cross_val_predict(
    sgd_clf, X_train_scaled, y_train, cv=3, n_jobs=3)
# %%
y_train_pred_parallel_jobs
# %% [markdown]
"""
That’s a lot of numbers. It’s often more convenient to look at an image
representation of the confusion matrix, using Matplotlib’s `matshow()` function:
"""
# %%
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
# %% [markdown]
"""
Let’s focus the plot on the errors. First, you need to divide each value in the confusion matrix by the number of images in the corresponding class, so you can
compare error rates instead of absolute number of errors (which would make abundant classes look unfairly bad):
"""
# %%
# calculate sum per every column (digit)
row_sums = conf_mx.sum(axis=1, keepdims=True)
row_sums
# %%
# Make a norm conf matrix
norm_conf_mx = conf_mx / row_sums
norm_conf_mx[1]
# %% [markdown]
"""
Remember that rows
represent actual classes, while columns represent predicted classes

Now let’s fill the diagonal with zeros to keep only the errors, and let’s plot the result:
"""
# %%
np.fill_diagonal(norm_conf_mx, 0)
fig, ax = plt.subplots(figsize=(5, 5), facecolor="white")
ax.matshow(norm_conf_mx, cmap=plt.cm.gray)

# %% [markdown]
"""
## Multilabel Classification

Until now each instance has always been assigned to just one class. In some cases you
may want your classifier to output multiple classes for each instance. For example,
consider a face-recognition classifier: what should it do if it recognizes several
people on the same picture? Of course it should attach one tag per person it recognizes.
Say the classifier has been trained to recognize three faces, Alice, Bob, and Charlie; then
when it is shown a picture of Alice and Charlie, it should output `[1, 0, 1]` (meaning
“Alice yes, Bob no, Charlie yes”). Such a classification system that outputs multiple
binary tags is called a multilabel classification system.
We won’t go into face recognition just yet, but let’s look at a simpler example, just for
illustration purposes:
"""
# %%
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier(n_jobs=3)
knn_clf.fit(X_train, y_multilabel)
# %%
knn_clf.predict([some_digit])
# %% [markdown]
"""
And it gets it right! The digit 5 is indeed not large (False) and odd (True).
There are many ways to evaluate a multilabel classifier, and selecting the right metric
really depends on your project. For example, one approach is to measure the F1 score
for each individual label (or any other binary classifier metric discussed earlier), then
simply compute the average score. This code computes the average $F_1$ score across all
labels:
"""
# %%
y_train_knn_predict = cross_val_predict(
    knn_clf, X_train, y_multilabel, cv=3, n_jobs=3)
f1_score(y_multilabel, y_train_knn_predict, average="macro")
# %% [markdown]
"""
## Multioutput Classification
"""
# %%
noise = np.random.randint(0, 10, (len(X_train), 784))
X_train_mod = X_train + noise

noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise

y_train_mod = X_train
y_test_mod = X_test

some_index = 0

fig, axis = plt.subplots(1, 2, figsize=(15, 15), facecolor='white')

axis[0].imshow(X_test_mod[some_index].reshape(28, 28),
               cmap=mpl.cm.binary)
axis[0].set_axis_off()

axis[1].imshow(y_test_mod[some_index].reshape(28, 28),
               cmap=mpl.cm.binary)
axis[1].set_axis_off()
# %% [markdown]
"""
On the left is the noisy input image, and on the right is the clean target image. Now
let’s train the classifier and make it clean this image:
"""
# %%
knn_clf.fit(X_train_mod, y_train_mod)
# %%
clean_image = knn_clf.predict([X_test_mod[some_index]])
plt.imshow(clean_image.reshape(28, 28),
           cmap=mpl.cm.binary)
