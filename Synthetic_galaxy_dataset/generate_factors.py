import numpy as np
from matplotlib import pyplot as plt
import pyDOE2 as pyDOE

def rescale01(xmin, xmax, f):
    """Rescale function to range [0, 1]."""
    return (f - xmin) / (xmax - xmin)

def create_plot_lhc(dir_in='./data/', dataset_mode='training'):
    """Create Latin Hypercube sampling and save the results to a file."""
    num_lhc = 1
    num_evals = [16384]
    num_params = 5
    verbose = False
    np.random.seed(7)

    for n in range(num_lhc):
        nevals = int(num_evals[n])

        # Define parameter ranges
        para1 = np.linspace(1e4, 1e5, nevals)
        para2 = np.linspace(0.1, 1.0, nevals)
        para3 = np.linspace(-0.5, 0.5, nevals)
        para4 = np.linspace(-0.5, 0.5, nevals)
        para5 = np.linspace(0.2, 0.4, nevals)

        if num_params == 7:
            para6 = np.linspace(0.5, 1.5, nevals)
            para7 = np.linspace(0.05, 0.5, nevals)
            all_para = np.vstack([para1, para2, para3, para4, para5, para6, para7])
            all_labels = [
                r'Flux', r'Radius', r'Shear g1', r'Shear g2', r'PSF fwhm', 
                r'z_m', r'FWHM'
            ]
        elif num_params == 5:
            all_para = np.vstack([para1, para2, para3, para4, para5])
            all_labels = [
                r'Flux', r'Radius', r'Shear g1', r'Shear g2', r'PSF fwhm'
            ]
        else:
            raise ValueError("Unknown parameter option")

        # Generate Latin Hypercube sampling
        lhd = pyDOE.lhs(all_para.shape[0], samples=nevals, criterion=None)
        idx = (lhd * nevals).astype(int)
        all_combinations = np.zeros((nevals, all_para.shape[0]))

        for i in range(all_para.shape[0]):
            all_combinations[:, i] = all_para[i][idx[:, i]]

        # Delete rows where g1^2 + g2^2 > 1
        del_rows = np.where(all_combinations[:, 2]**2 + all_combinations[:, 3]**2 > 1.0)[0]
        all_combinations = np.delete(all_combinations, del_rows, axis=0)

        # Save the results
        filename = f"{dir_in}lhc_{nevals}_{num_params}_{dataset_mode}.txt"
        np.savetxt(filename, all_combinations)
        print(f'Saved at: {filename}')

        if verbose:
            f, a = plt.subplots(all_para.shape[0], all_para.shape[0], sharex=True, sharey=True)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
            plt.rcParams.update({'font.size': 4})

            for i in range(all_para.shape[0]):
                for j in range(i + 1):
                    if i != j:
                        a[i, j].scatter(lhd[:, i], lhd[:, j], s=1, alpha=0.7)
                        a[i, j].grid(True)
                        a[j, i].set_visible(False)
                    else:
                        a[i, i].text(0.4, 0.4, all_labels[i], size='x-large')
                        hist, bin_edges = np.histogram(lhd[:, i], density=True, bins=12)
                        a[i, i].bar(bin_edges[:-1], hist / hist.max(), width=0.09, alpha=0.5)
                        plt.xlim(0, 1)
                        plt.ylim(0, 1)
            plt.tight_layout()

# Call the function
create_plot_lhc(dir_in='./data/')
