import numpy as np
import pandas as pd


def w_adj(w, n, down, up):
    w2 = pd.DataFrame(np.array(w).T, columns=['w']).sort_values(by='w', ascending=0)
    w2.iloc[0:n, 0] = w2.iloc[0:n, 0] * 1 / sum(w2.iloc[0:n, 0])
    w2.iloc[n:, 0] = 0

    j = n - 1
    while w2.iloc[j, 0] < down:
        w2.iloc[0, 0] += w2.iloc[j, 0]
        w2.iloc[j, 0] = 0
        j -= 1

    i = 0
    while w2.iloc[i, 0] > up:
        w2.iloc[i + 1, 0] += w2.iloc[i, 0] - up
        w2.iloc[i, 0] = up
        i += 1
    w3 = np.array(w2.sort_index().iloc[:, 0]).round(4)

    if sum(w3) > 1:
        w3[w3.argmax()] -= sum(w3) - 1
    elif sum(w3) < 1:
        w3[np.where(w3 != 0, w3, w3 + 1).argmin()] += 1 - sum(w3)
    return w3.round(4)
