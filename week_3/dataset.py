import numpy as np
import plotly.express as px

def dataset_Circles(m=1000, radius=0.7, noise=0.0):
    X = np.zeros((m, 2, 1))
    Y = np.zeros((m, 1, 1))

    for currentN in range(m):
        i, j = 2 * np.random.rand(2) - 1

        r = np.sqrt(i ** 2 + j ** 2)
        if (noise > 0.0):
            r += np.random.rand() * noise

        if (r < radius):
            l = 0
        else:
            l = 1

        X[currentN, 0] = [i]
        X[currentN, 1] = [j]
        Y[currentN] = [[float(l)]]

    return np.asarray(X), np.asarray(Y)

def draw_dataset(x, y):
    if x.shape[0] == 2:
        fig = px.scatter(x=x[0], y=x[1], color=y[0], width=700, height=700)
    elif x.shape[1] == 2:
        xx = x.reshape(-1,2).T
        yy = y.reshape(1,-1)
        fig = px.scatter(x=xx[0], y=xx[1], color=yy[0], width=700, height=700)
    else:
        return
    fig.show()


if __name__ == '__main__':
    x,y = dataset_Circles(128)
    print(x.shape)
    print(y.shape)
    draw_dataset(x, y)