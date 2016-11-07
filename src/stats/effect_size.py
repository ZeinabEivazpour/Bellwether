from __future__ import print_function, division
from pdb import set_trace
import numpy as np
import pandas as pd


def pooled_sd(dist1, dist2):
    s1 = np.std(dist1)
    s2 = np.std(dist2)
    n1 = len(dist1)
    n2 = len(dist2)

    s = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))

    return s


def cohens_d(dist1, dist2, cutoffs=None):
    if cutoffs is None:
        cutoffs = {"VS": 0.01,
                   "S": 0.20,
                   "M": 0.50,
                   "L": 0.80,
                   "VL": 1.20,
                   "H": 2.00,
                   }

    s = pooled_sd(dist1, dist2)
    m1 = np.mean(dist1)
    m2 = np.mean(dist2)
    d = np.abs(m1 - m2) / (s + 1e-32)
    return round(d, 2), d < cutoffs["S"]


def hedges_g(dist1, dist2, small=0.38):
    g, _ = cohens_d(dist1, dist2)
    n1 = len(dist1)
    n2 = len(dist2)
    correct_bias = (1 - 3 / (4 * (n1 + n2) - 9))
    """ Hedges's G measure with lowered bias.
        See: https://en.wikipedia.org/wiki/Effect_size#Hedges.27_g
    """
    g_star = correct_bias * g
    return round(g_star, 2), g_star < small


def hedges_g_2(dist, small=0.38):
    """
    Hedges's G measure with lowered bias.
        See: https://en.wikipedia.org/wiki/Effect_size#Hedges.27_g
    """

    n1, n2 = 4, 4
    correct_bias = (1 - 3 / (4 * (n1 + n2) - 9))

    def hg(m1, m2, s1, s2):
        s = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
        g_score = correct_bias * (m1 - m2) / (s + 1e-32)  # Adding 1e-32 handles 0 std
        return str(round(-g_score, 2)) if abs(g_score) >= small else " "

    all = []

    for id in dist.index:
        stats = dist.loc[id]

        pd_hg = hg(stats.values[0], stats.values[6], stats.values[1], stats.values[7])
        pf_hg = hg(stats.values[2], stats.values[8], stats.values[3], stats.values[9])
        g_hg = hg(stats.values[4], stats.values[10], stats.values[5], stats.values[11])

        all.append([id,  # Name
                    # TCA+
                    round(stats.values[0], 2),  # Pd (mean), Pd(std)
                    round(stats.values[2], 2),  # Pf (mean), Pf(std)
                    round(stats.values[4], 2),  # G (mean), G(std)
                    # Bellwether
                    round(stats.values[6], 2),  # Pd (mean), Pd(std)
                    round(stats.values[8], 2),  # Pf (mean), Pf(std)
                    round(stats.values[11], 2),  # G (mean), G(std)
                    # Hedges' G (Pd),  Hedges' G (Pf),  Hedges' G (G)
                    pd_hg, pf_hg, g_hg])

    return pd.DataFrame(all, columns=["Name",
                                      # TCA+
                                      "Pd (mean)",
                                      "Pf (mean)",
                                      "G (mean)",
                                      # Bellwether
                                      "Pd (mean)",
                                      "Pf (mean)",
                                      "G (mean)",
                                      # Hedge's G
                                      "Pd (Hedges's G)",
                                      "Pf (Hedges's G)",
                                      "G (Hedges's G)"])


def _test_effect_size():
    dist1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    dist2 = [1.5, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]

    g, g_eff = hedges_g(dist1, dist2)
    print("Hedge's G: {}| Small Effect: {}".format(g, g_eff))
    d, d_eff = cohens_d(dist1, dist2)
    print("Cohen's d: {}| Small Effect: {}".format(d, d_eff))


if __name__ == "__main__":
    _test_effect_size()
