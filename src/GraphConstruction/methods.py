import numpy as np
import pandas as pd

def vectors2distance(list_vectors, metric='cosine', normalised=True):
    """
    This method takes a list of fixed dimensional vectors of same size, then returns pairwise distances between them using Scipy's pdist method in square form. 
    Input:

    list_vectors: list of fixed dimensional vectors in numpy array or native Python list format
    metric: string name of the distance metric to be used. Alternatives canbe found here: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
    normalised: boolean parameter to return distance normalised so that the largest distance equals to 1

    Output:

    distance: a square matrix of Numpy ndarray type.
    """
    from scipy.spatial.distance import pdist, squareform
    distance = squareform(pdist(np.array(list_vectors), metric=metric), checks=True)
    if normalised:
        distance = np.nan_to_num(distance)
        return distance / np.amax(distance)
    else:
        return distance


def distance2similarity(distance):
    """
    This method converts the distance into normalised similarity measure (between 0 and 1) assuming that similarity is 1 - max-normalised distance

    Input: distance matrix from vectors2distance
    Output: similarity matrix
    """
    return -(distance-np.amax(distance))


def distance_closure(D, b_type):
    """
    Distance closure is based on the following paper:
    "T. Simas and L.M. Rocha [2015]."[Distance Closures on Complex Networks](http://www.informatics.indiana.edu/rocha/publications/NWS14.php)". Network Science, 3(2):227-268. doi:10.1017/nws.2015.11"

    I forked it to make it support Python 3: pip install --upgrade git+https://github.com/tarikaltuncu/distanceclosure
    """
    from distanceclosure.closure import transitive_closure
    from distanceclosure.backbone import backbone
    bbone = backbone(D, transitive_closure(D, b_type))
    bbone[bbone != 1] = 0
    return bbone

def check_symmetric(A, tol=1e-8):
    """"
    Checks whether a given Adjacency matrix A is symmetric or not with a tolerance parameter
    """
    return np.allclose(A, A.T, atol=tol)

def c_knn(D, k=5):
    """
    :param D: Distance matrix in squareform
    :param k: Number of neighbours
    :return: Matrix in squareform with ones for the edges to keep, zeros to prune
    Citation: Python implementation of Continuous k-Nearest Neighbors(CkNN) proposed in the paper 'Consistent Manifold Representation for Topological Data Analysis' (https://arxiv.org/pdf/1606.02353.pdf)
    Source: https://github.com/chlorochrule/cknn/blob/master/cknn/cknn.py
    """
    import cknn
    from scipy.spatial.distance import squareform
    adjacency = cknn.cknneighbors_graph(squareform(D), n_neighbors=k,
                                   delta=1.0, metric='precomputed',
                                   t='inf', include_self=True,
                                   is_sparse=False)
    np.fill_diagonal(adjacency, 0)
    if check_symmetric(adjacency):
        return adjacency
    else:
        print(f"c_knn for {k} was not symmetric. Returning the average.")
        return (adjacency + adjacency.T) / 2


def rmst(D, p=2.0):
    """
    Citation: Beguerisse-Díaz, Mariano, Borislav Vangelov, and Mauricio Barahona. "Finding role communities in directed networks using role-based similarity, markov stability and the relaxed minimum spanning tree." 2013 IEEE Global Conference on Signal and Information Processing. IEEE, 2013.
    Original implementation of the RMST algorithm can be found here: https://github.com/barahona-research-group/RMST
    """

    N = D.shape[0]
    #     Emst, LLink = prim2(D)
    Emst, LLink = mst_sym(D)

    '''Find distance to nearest neighbors'''
    Dtemp = D + np.eye(N) * np.max(D, axis=0)
    mD = np.min(Dtemp, axis=0) / p

    '''Check condition'''
    mD = np.matlib.repmat(mD, N, 1) + np.matlib.repmat(np.matrix(mD).T, 1, N)
    E = (D - mD < LLink).astype(int)
    E = E - np.diag(np.diag(E))

    # E = np.sign(E)
    '''Overlay MST with RMST'''
    E = np.sign(E + Emst)

    return E

def knn_mst(D, k=13):
    """
    Citation: Veenstra, P., C. Cooper, and S. Phelps. “Spectral Clustering Using the KNN-MST Similarity Graph.” In 2016 8th Computer Science and Electronic Engineering (CEEC), 222–27, 2016. https://doi.org/10.1109/CEEC.2016.7835917.
    Implemented in Python by Tarik Altuncu
    """
    n = D.shape[0]
    assert (D.shape[0] == D.shape[1])

    np.fill_diagonal(D, 0)
    A = np.zeros((n, n))
    for i in range(n):
        ix = np.squeeze(np.asarray(np.argsort(D[i, :])))
        A[i, ix[1]] = 1  # Connect to the nearest node after itself
        A[ix[1], i] = 1  # The same on other direction

        for j in range(k - 1):
            j += 2
            A[i, ix[j]] = 1
            A[ix[j], i] = 1

    mst_remained_edges = mst_sym(D, False)
    remained_edges = np.maximum(A, mst_remained_edges)

    return remained_edges

def mst_sym(A, return_LongestLinks=True):
    """scipy mst (kruskal) return triangular matrix as mst"""
    from scipy.sparse.csgraph import minimum_spanning_tree
    dim = A.shape[0]
    mst = minimum_spanning_tree(A).todense()
    mst[mst > 0] = 1
    remained_edges = np.maximum(mst, mst.T)

    if return_LongestLinks:
        LongestLinks = findMlink(np.multiply(A, remained_edges))
        return remained_edges, LongestLinks
    else:
        return remained_edges

def one_node(graph, source):
    from networkx import single_source_shortest_path
    paths = single_source_shortest_path(graph, source)
    one_line = dict()
    one_line[source] = dict()
    for target, path in paths.items():
        LongestLink = 0
        for i in range(len(path) - 1):
            weight = graph[path[i]][path[i + 1]]['weight']
            LongestLink = np.maximum(weight, LongestLink)
        one_line[source][target] = LongestLink
    return one_line

def findMlink(adj):
    from itertools import repeat
    from multiprocessing import cpu_count, Pool
    from networkx import from_numpy_matrix
    dim = adj.shape[0]
    graph = from_numpy_matrix(adj)

    cpu_cores = min(8, cpu_count(), len(graph.nodes()))
    pool = Pool(cpu_cores)
    one_lines = pool.starmap(one_node, list(zip(repeat(graph), range(dim))))
    pool.close()

    LongestLinks = dict()
    for one_line in one_lines:
        LongestLinks[list(one_line.keys())[0]] = list(one_line.values())[0]

    LongestLinks = pd.DataFrame(LongestLinks).as_matrix().T

    return LongestLinks


def prim2(D):
    """This algorithm tries to return MST but it return nonaccurate results"""
    ''' A vector (T) with the shortest distance to all nodes.
    % After an addition of a node to the network, the vector is updated'''

    LLink = np.zeros(D.shape)
    '''Number of nodes in the network'''
    N = D.shape[0]
    assert (D.shape[0] == D.shape[1])

    '''Allocate a matrix for the edge list'''
    E = np.zeros([N, N])

    allidx = np.arange(N)

    '''Start with a node'''
    mstidx = np.array([0])

    otheridx = np.setdiff1d(allidx, mstidx)

    T = D[0, otheridx]
    P = np.zeros(otheridx.size, dtype=np.int)

    while (T.size > 0):
        i = np.argmin(T)
        m = T[i]
        idx = otheridx[i]

        '''Update the adjancency matrix'''
        E[idx, P[i]] = 1
        E[P[i], idx] = 1

        '''Update the longest links'''
        ''' 1) indexes of the nodes without the parent'''
        idxremove = np.nonzero(mstidx == P[i])
        tempmstidx = mstidx
        tempmstidx = np.delete(tempmstidx, idxremove)

        ''' 2) update the link to the parent'''
        LLink[idx, P[i]] = D[idx, P[i]]
        LLink[P[i], idx] = D[idx, P[i]]
        ''' 3) find the maximal'''
        tempLLink = np.maximum(LLink[P[i], tempmstidx], D[idx, P[i]])
        LLink[idx, tempmstidx] = tempLLink
        LLink[tempmstidx, idx] = tempLLink

        '''As the node is added clear his entries'''
        P = np.delete(P, i)
        T = np.delete(T, i)

        '''Add the node to the list '''
        mstidx = np.append(mstidx, idx)

        '''Remove the node from the list of the free nodes'''
        otheridx = np.delete(otheridx, i)

        '''Updata the distance matrix'''
        Ttemp = D[idx, otheridx]

        if (T.size > 0):
            idxless = np.nonzero(Ttemp < T)
            T[idxless] = Ttemp[idxless]
            P[idxless] = idx

    return E, LLink