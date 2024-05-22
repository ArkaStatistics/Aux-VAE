import h5py
import numpy as np
import matplotlib.pyplot as plt

def read_hdf5_outputs(file_path):
    """
    Reads an HDF5 file and returns a list of numpy arrays representing galaxies.
    Each dataset is treated as a collection of galaxies.

    Parameters:
    file_path (str): Path to the HDF5 file.

    Returns:
    list: List of numpy arrays (images).
    """
    images = []
    with h5py.File(file_path, 'r') as file:
        for key in file.keys():
            dataset = np.array(file[key])
            for img in dataset:
                images.append(img)
    return images


def read_input_params(file_path):
    """
    Reads the data from a text file using NumPy.

    Parameters:
    file_path (str): Path to the text file.

    Returns:
    numpy.ndarray: Data read from the file.
    """
    return np.loadtxt(file_path)



def plot_scatter_and_histogram_labeled(data, labels):
    """
    Creates a grid of scatter plots for each pair of parameters, and histograms on the diagonal using NumPy and Matplotlib.
    Adds labels for each axis.

    Parameters:
    data (numpy.ndarray): Data to be plotted.
    labels (list): List of labels for each column.
    """
    num_params = data.shape[1]
    fig, axes = plt.subplots(num_params, num_params, figsize=(15, 15))

    # Iterate over each subplot position
    for i in range(num_params):
        for j in range(num_params):
            ax = axes[i, j]
            # ax.set_aspect('equal')

            # Plotting histogram on the diagonal
            if i == j:
                ax.hist(data[:, i], bins=30, color='blue', edgecolor='black')
                ax.set_xlabel(labels[i])
                ax.set_ylabel(labels[i])
            # Plotting scatter plot for off-diagonal
            else:
                ax.scatter(data[:, j], data[:, i], alpha=0.6)
                ax.set_xlabel(labels[j])
                ax.set_ylabel(labels[i])

    plt.tight_layout()
    plt.show()




def plot_images(images, rows, cols):
    """
    Plots a list of images on a grid.

    Parameters:
    images (list): List of numpy arrays (images).
    rows (int): Number of rows in the grid.
    cols (int): Number of columns in the grid.
    """
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    for img, ax in zip(images, axes):
        ax.imshow(img, cmap='gray')  # Assuming grayscale images
        ax.axis('off')

    plt.tight_layout()
    plt.show()
