import torch
import mat73
import plotly.graph_objects as go
import numpy as np
from scipy.ndimage import zoom


def PlotVolume(data = None, threshold = 0.1):
    '''Plot 3d volume
    Parameters
    ------------
    data : float
        3d desity matrix
    threshold : float (0.~1.)
        threshold of plot (larger -> less points)
    '''
    if data is None:
        data = mat73.loadmat(r'D:\LPP\NLOSFeatureEmbeddings\lct_180min.mat')
        data = np.array(data['lct'])

    X, Y, Z = np.mgrid[0:100:100j, 0:128:128j, 0:128:128j]
    # values = np.sin(X*Y*Z) / (X*Y*Z)
    # print(data)
    values = zoom(np.array(data) , (100. / data.shape[0], 128. / data.shape[1], 128. / data.shape[2]))

    # values *= 1e8
    print(values)
    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=values.flatten(),
        isomin=values.min() + (values.max() - values.min()) * threshold,
        isomax=values.max(),
        opacity=0.9, # needs to be small to see through all surfaces
        surface_count=20, # needs to be a large number for good volume rendering
    ))
    fig.show()


if __name__ == '__main__':

    PlotVolume()
    # data = mat73.loadmat(r'D:\LPP\NLOSFeatureEmbeddings\lct_180min.mat')
    # data = np.array(data['lct'])
    # a = (32. / data.shape[0], 32. / data.shape[1], 32. / data.shape[2])
    # print(a)