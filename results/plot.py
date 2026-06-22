import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

import numpy as np
import pandas as pd
import sys

# plot list of csv files
def plot_csv(csv_files, savefile='plot'):
    plt.figure(figsize=(10, 6))
    plt.xlabel('N')
    plt.ylabel('time[s]')
    plt.xscale('log')
    markers = ['o', '+', 's', '*', 'v']

    for filename in csv_files:
        # Use the filename (without extension) as the label and remove _ if present
        label = filename.split('.')[0].replace('_', ' ')   
        data = pd.read_csv(filename)
        # print(data.info())
        # print(data['stream_triad_size'])
        plt.plot(data['N'], data['h2d'], label=label, marker=markers[csv_files.index(filename) % len(markers)], markerfacecolor='none')
    
    # plotname = savefile.replace('_', ' ')
    plotname = "SPMV memory transfer time"
    plt.title(plotname)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(f'{savefile}.png', dpi=300)




if __name__ == "__main__":

    #take argument from command line
    #list of csv files
    if len(sys.argv) > 1:
        plot_csv(sys.argv[2:], sys.argv[1])
    else:   
        print("Usage: python plot.py <output_file> <csv_file1> <csv_file2> ...")
        print("Example: python plot.py simd.csv simd-0.csv")
        sys.exit(1)
# plot_csv('simd.csv', 'simd-1')
# plot_csv('stream_triad02.csv', 'test02')
