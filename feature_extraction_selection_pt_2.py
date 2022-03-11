# from sklearn.datasets import fetch_mldata
from sklearn.datasets import fetch_openml
# mnist = fetch_mldata('MNIST original') # fetch_mldata website died replaced with fetch_openml
mnist = fetch_openml('mnist_784')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import time

#split data between train and test
train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state=None)
scaler = StandardScaler()
# fit training set only
scaler.fit(train_img)

# apply tranform to both training set and test set.
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

# make an instance of the model
pca = PCA(.85)

#PCA Fit
pca.fit(train_img)

# how many components did PCA choose?
print("\nNumber of components: %s\n" % pca.n_components_)

train_img = pca.transform(train_img)
test_img = pca.transform(test_img)

logisticRegr = LogisticRegression(solver = 'lbfgs')

#train the model
start_time = time.time()
logisticRegr.fit(train_img, train_lbl)
finish_time = time.time()

#Predict for Ten Observations (image)
logisticRegr.predict(test_img[0:10])

# Measure performance
logisticRegr.score(test_img, test_lbl)


run_time = finish_time - start_time
run_time