import matplotlib.pyplot as plt

def displayArray(data, height=5):
    """
    Display a 3D numpy array containing [r,g,b] values per pixel
    -
    Input /
    data = 3D numpy array
    height = maximum height to keep display ratio
    """
    plt.figure(figsize=(height * (data.shape[1] / data.shape[0]), height))
    plt.imshow(data, cmap = 'copper')
    plt.tight_layout()
    plt.axis('off')
    plt.show()
