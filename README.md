#   K Nearest Neighbors from scratch!

#### Goal:
Write the K Nearest Neighbors classifier from scratch in Python. Classify spam emails using the two provided datasets. 

(a) Report test accuracies when k = 1; 5; 11; 21; 41; 61; 81; 101; 201; 401 without normalizing the features.
(b) Report test accuracies when k = 1; 5; 11; 21; 41; 61; 81; 101; 201; 401 with z-score normalization applied to the features.
(c) In the (b) case, generate an output of KNN predicted labels for the first 50 instances (i.e. t1 - t50) when k = 1; 5; 11; 21; 41; 61; 81; 101; 201; 401 (in this order). For example, if t5 is classified as class "spam" when k = 1; 5; 11; 21; 41; 61 and classified as class "no-spam" when k = 81; 101; 201; 401, then your output line for t5 should be:
t5 spam, spam, spam, spam, spam, spam, no, no, no, no
(d) What can you conclude by comparing the KNN performance in (a) and (b)?
(e) Describe a method to select the optimal k for the KNN algorithm.

#### Data:
The provided datasets are email spam classification datasets in csv format. The data is split into train and test datasets  named spam_train and spam_test.Train dataset consists of 2300  instances  and test dataset consists of 2301 instances.  Each dataset has 57 features and class/label or predicted class/label. Label "1" means spam and label "0" means not spam. 

#### Tools:
The code is written in Python using Visual Studio Code. Libraries used: pandas, numpy, statistics, scipy, datetime.
