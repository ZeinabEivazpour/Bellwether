from __future__ import division, print_function
import os
import sys
from pdb import set_trace
import networkx as nx
from KSAnalyzer import KSAnalyzer, get_data

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)


def weightedBipartite(matches):
    """
    Weighted Bipartite Matching Filter.
    Adapted from @WeiFoo's HDP codebase. See https://goo.gl/3AN2Qd.
    :param matches: Dictionary. Key is a pair (source, target). Value is p-value.
                    Obtained from KSAnalyzer
    :return: Best bipartite pair from the matches.
    """

    # Tags for source and target metrics.
    s_tag = "_s"
    t_tag = "_t"
    G = nx.Graph()
    for key, val in matches.iteritems():
        G.add_edge(key[0] + s_tag, key[1] + t_tag, weight=val)  # add suffix to make it unique
    result = nx.max_weight_matching(G)

    """
    We only want matched pairs with (S->T) and (T<-S). Remove singletons;
    i.e., only (S->T) or only (T->S)
    """

    # for attr_1, attr_2 in result.iteritems():
    return result


def _test__weightedBipartite():
    """
    Test Weighted Bipartite Matching
    :return:
    """
    print("Testing weightedBipartite\n")
    data0, data1 = get_data()
    matches = KSAnalyzer(source=data0, target=data1, cutoff=0.05)

    try:
        matched = weightedBipartite(matches)
        import pprint
        pretty = pprint.PrettyPrinter(indent=2)
        pretty.pprint(matched)
        print("Test Succeeded.")
    except Exception as e:
        print("Test Failed")
        # ----- Debug -----
        set_trace()


if __name__ == "__main__":
    _test__weightedBipartite()
