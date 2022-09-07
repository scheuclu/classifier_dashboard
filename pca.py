import dataprep
import plotly as py
import plotly.graph_objects as go

data = dataprep.read_data()
X_train = data['X_train']
X_test = data['X_test'].numpy()
y_train = data['y_train']
y_test = data['y_test']


# import numpy as np
# from sklearn.decomposition import PCA
# #X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# pca = PCA(n_components=2)
# X_embedded = pca.fit_transform(X_test)


from sklearn.manifold import TSNE
#X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
X_embedded = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=4).fit_transform(X_test)

print(X_embedded)

trace= go.Scatter(
    x=X_embedded[:,0],
    y=X_embedded[:,1],
    mode='markers',
    marker=dict(color=y_test)
)
fig=go.FigureWidget(trace)
fig.show()