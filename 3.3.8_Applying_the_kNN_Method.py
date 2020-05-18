'''SciKitLearn is an open source machine learning library

  HERE WILL BE PROOF THE METHOD IN A FLOWERS CASE, THE DATA IS:
  for each flower we have the following info:
  Sepal length
  Sepal width
  Petal length
  Petal width
  '''
import numpy as np
import matplotlib.pyplot as plt

def distance(p1, p2):
    '''
    :param p1:
    :param p2:
    :return: distance between p1 and p2:
    '''
    return np.sqrt(np.sum(np.power(p2 - p1, 2)))

def majority_vote(votes):
    '''
    :param votes: list to count the elements
    :return: mode: mean count of the list
    '''
    import scipy.stats as ss
    mode, count = ss.mstats.mode(votes)
    return mode

def find_nearest_neighbors(p, points, k=5):
    '''
    :param p: point selected.
    :param points: array of the rest of points.
    :param k: number of nearest neighbors looking for.
    :return: the index of the neighbors looked.
    '''
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = distance(p, points[i])
    ind = np.argsort(distances)
    return ind[:k]

def knn_predict(p, points, outcomes, k=5):
    ind = find_nearest_neighbors(p, points, k)
    return majority_vote(outcomes[ind])

def make_predict_grid(predictors, outcomes, limits, h, k):
    '''
    Classify each point on the prediction grid.
    :param predictors: array of points
    :param outcomes: list of elements
    :param limits: Tuple(x_min, x_max, y_min, y_max)
    :param h: step-size
    :param k: number of nearest neighbors
    :return Tuple: (x_array, y_array, prediction_grid)
    '''
    (x_min, x_max, y_min, y_max) = limits
    xs = np.arange(x_min, x_max, h)
    ys = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(xs, ys)

    prediction_grid = np.zeros(xx.shape, dtype=int)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p = np.array([x, y])
            prediction_grid[j, i] = knn_predict(p, predictors, outcomes, k)
    return (xx, yy, prediction_grid)

def plot_prediction_grid (xx, yy, prediction_grid, filename):
    """
    Plot KNN predictions for every point on the grid.
    :param xx: 'x' array provided by 'meshgrid'
    :param yy: 'y' array provided by 'meshgrid'
    :param prediction_grid: matrix filled with 'knn_predict'
    :param filename: name to save the file
    :return:
    """
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap (["hotpink","lightskyblue", "yellowgreen"])
    observation_colormap = ListedColormap (["red","blue","green"])
    plt.figure(figsize =(10,10))
    plt.pcolormesh(xx, yy, prediction_grid, cmap = background_colormap, alpha = 0.5)
    plt.scatter(predictors[:,0], predictors [:,1], c = outcomes, cmap = observation_colormap, s = 50)
    plt.xlabel('Variable 1'); plt.ylabel('Variable 2')
    plt.xticks(()); plt.yticks(())
    plt.xlim (np.min(xx), np.max(xx))
    plt.ylim (np.min(yy), np.max(yy))
    #plt.show()
    plt.savefig(filename)
#--------------------------------------------------------


from sklearn import datasets
iris = datasets.load_iris()
predictors = iris.data[:, 0:2]
outcomes = iris.target
plt.plot(predictors[outcomes==0][:, 0], predictors[outcomes==0][:, 1], "ro")
plt.plot(predictors[outcomes==1][:, 0], predictors[outcomes==1][:, 1], "go")
plt.plot(predictors[outcomes==2][:, 0], predictors[outcomes==2][:, 1], "bo")
#plt.show()
plt.savefig("iris.pdf")

predictors = iris.data[:, 0:2]
outcomes = iris.target
k=5; filename="iris_grid.pdf"; limits = (4, 8, 1.5, 4.5); h = 0.1;
(xx, yy, prediction_grid) = make_predict_grid(predictors, outcomes, limits, h, k)
plot_prediction_grid(xx, yy, prediction_grid, filename)


#-----------COMBINING SKLEARN & OWN ALGORITHM-------------
'''sklearn'''
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(predictors, outcomes)
sk_predictions = knn.predict(predictors)
'''My predictions'''
iris = datasets.load_iris()
predictors = iris.data[:, 0:2]
outcomes = iris.target
my_predictions = np.array([knn_predict(i, predictors, outcomes) for i in predictors])
result = 100 * np.mean(sk_predictions == my_predictions)
print("sk and my predictions are equal: ", result, "%")
result = 100 * np.mean(sk_predictions == outcomes)
print("sk predictions are equal to outcomes: ", str(result), "%")
result = 100 * np.mean(my_predictions == outcomes)
print("my predictions are equal to outcomes: " + str(result) + "%")



