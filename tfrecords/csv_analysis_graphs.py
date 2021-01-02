from pandas import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# path al file csv da cui estrapolare i grafi
# CSV_FILE = "train.age_detected.csv"
CSV_FILE = "valid.reduced(random).csv"

def main():
    print("Inizio lettura csv")
    data = pd.read_csv(CSV_FILE)
    print(data.columns)

    age_frequency_graph(data)
    # identity_frequency_graph(data)


def identity_frequency_graph(data):
    data.set_index('percorso')
    plt.figure(figsize=(10, 5))

    y, x, _ = plt.hist(data['percorso'].str.slice(start=3, stop=7), bins=9279)   # n009279/0583_03.jpg -> 9279
    plt.title("Identities")
    plt.xlabel("Identity [ID]")
    plt.ylabel("Images [N]")
    # plt.xlim(0, 9279)
    # plt.ylim(0, 0.055)
    # annotate_max(x, y)

    plt.savefig(CSV_FILE.replace('.csv', '-ids.png'), dpi=300, bbox_inches='tight')
    plt.show()


def age_frequency_graph(data):
    data.set_index('età')
    y, x, _ = plt.hist(data['età'], density=600, bins=100)
    plt.title("Distribuzione di età nel training set ridotto (random)")
    plt.xlabel("Età [n]")
    plt.ylabel("Frequenza di campioni (normalizzata)")
    plt.xlim(0, 100)
    plt.ylim(0, 0.055)
    plt.grid(True)
    annotate_max(x, y)
    plt.draw()

    plt.savefig(CSV_FILE.replace('.csv', '-ages.png'), dpi=300, bbox_inches='tight')
    plt.show()


def annotate_max(x, y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94, 0.96), **kw)


if __name__ == '__main__':
    main()

