import CTL.funcs.xplib as xplib
from CTL.funcs.graph import UndirectedGraph
import CTL.funcs.funcs as funcs
from CTL.tensor.tensor import TensorLike
from CTL.tensor.contract.link import makeLink
# import numpy as np

import warnings

class TensorGraph(UndirectedGraph):
    """
    An undirected graph generated from a set of tensors. Inheriting from CTL.funcs.graph.UndirectedGraph

    The vertices are tensors, while edges are just the bonds between tensors. Free edges are for outgoing bonds.

    With the tensor graph we can obtain the best or near-best sequence for contraction.

    Parameters
    ----------
    n : int 
        Number of tensors(vertices).
    diagonalFlags : None or list of bool
        Whether the input tensors are diagonal: if some tensor is diagonal, then use a different method to calculate the cost for contraction.

    Attributes
    ----------
    indexed : bool
        Whether the indices have been added to edges.
    diagonalFlags : list of bool
        Whether the tensors are considered as diagonal or not.
    contractRes : list of tuples of ([int], (int, ), bool)
        ContractRes[mask] is the result of contraction of tensors according to the bit-mask. Result contains (edge-list, shape, is-diagonal).
    optimalCost : list of int
        OptimalCost[mask] is the cost for contraction of tensors accoring to the bit-mask.
    optimalSeq : list of list of length-2 tuple of ints
        OptimalSeq[mask] is the sequence for contraction of tensors according to the bit-mask.
    greedyCost : int
        The cost of the greedy order.
    greedySeq : list of length-2 tuple of ints
        The sequence of the greedy order.
    """
    def __init__(self, n, diagonalFlags = None):
        super().__init__(n)
        self.indexed = False
        if (diagonalFlags is None):
            self.diagonalFlags = [False] * n
        else:
            self.diagonalFlags = diagonalFlags

    def addFreeEdge(self, idx, weight = None):
        """
        Add a free edge to a vertex.

        Parameters
        ----------
        idx : int
            The index of the vertex.
        weight : None or int
            If int, then added to the added edge as weight, used for cost calculation.
        """
        self.v[idx].addEdge(None, weight = weight)

    def addEdgeIndex(self):
        """
        Add an index to each edge.
        """
        if (self.indexed):
            return
        edges = self.getEdges()
        m = len(edges)
        for i in range(m):
            edges[i].index = i 
        self.indexed = True

    def tensorNetworkGraph(self):
        try:
            import networkx as nx
        except:
            print(funcs.errorMessage(err = 'Optional dependency networkX has not been installed, skip plotting tensor network graph.', location = 'tensorGraph.tensorNetworkGraph'))
        
        G = nx.Graph()
        edges = self.getEdges()
        G.add_nodes_from([v.index for v in self.v])
        for i in range(len(self.v)):
            G.nodes[i]['name'] = self.v[i].name
        for edge in edges:
            # print(edge)
            u, v = edge.vertices
            if (u is not None) and (v is not None):
                uIdx = u.index
                vIdx = v.index
                G.add_edge(uIdx, vIdx, weight = edge.weight)
            else:
                # deal with free bond
                pass

        return G

    def optimalCostResult(self):
        """
        Answer the query of cost for optimal contraction.

        Returns
        -------
        int
            The exact cost for contraction of all tensors in the graph.
        """
        if (self.optimalCost is None):
            self.optimalContractSequence()
        return self.optimalCost[(1 << self.n) - 1]
    
    def optimalContractSequence(self, bf = False, greedy = False, typicalDim = 10):
        """
        Generate an order for contraction of tensors in the graph.

        Parameters
        ----------
        bf : bool, default False
            Whether to calculate the order by brute-force.
        greedy : bool, default False
            Whether to calculate the order by greedy algorithm, so the order may not be optimal. If both bf and greedy is False, then calculate with ncon technique. Priority is greedy > bf.
        typicalDim : int or None
            The typical bond dimension for cost calculation. If None, then calculate with exact dimensions.
        
        Returns
        -------
        list of length-2 tuple of ints
            Length of the returned list should be $n - 1$, where $n$ is the length of tensorList.
            Every element contains two integers a < b, means we contract a-th and b-th tensors in tensorList, and save the new tensor into a-th location.

        """

        if (greedy and (typicalDim is not None)):
            warnings.warn(funcs.warningMessage(warn = 'greedy search of contract sequence, typicalDim {} has been ignored.'.format(typicalDim), location = 'TensorGraph.optimalContractSequence'))

        def lowbitI(x):
            return (x & (-x)).bit_length() - 1
        def lowbit(x):
            return (x & (-x))

        def subsetIterator(x):
            lb = lowbit(x)
            while (lb * 2 < x):
                yield lb 
                nlb = lowbit(x - lb)
                lb = (lb & (~(nlb - 1))) + nlb
            return 

        self.addEdgeIndex()

        edges = [[edge.index for edge in v.edges] for v in self.v]
        shapes = [[edge.weight for edge in v.edges] for v in self.v]

        n = len(self.v)

        if (not greedy):
            self.optimalCost = [None] * (1 << n)
            self.optimalSeq = [None] * (1 << n)
            self.contractRes = [None] * (1 << n)

            for i in range(n):
                self.optimalCost[(1 << i)] = 0
                self.optimalSeq[(1 << i)] = []
                self.contractRes[(1 << i)] = (edges[i], shapes[i], self.diagonalFlags[i])

        else:
            self.greedyCost = 0
            self.greedySeq = [] 
            self.contractRes = [None] * n
            for i in range(n):
                self.contractRes[i] = (edges[i], shapes[i], self.diagonalFlags[i])

        def getCost(tsA, tsB):
            # to deal with diagonal tensors:
            # what if we contract two diagonal tensors?
            # then it will be a new diagonal tensor with new index set
            # this means, we need to add a "diagonalFlag" to contractRes
            edgesA, shapeA, diagonalA = self.contractRes[tsA]
            edgesB, shapeB, diagonalB = self.contractRes[tsB]

            if (diagonalA and diagonalB):
                # both diagonal, only need to product
                return shapeA[0]
            
            diagonal = diagonalA or diagonalB

            clb = set(funcs.commonElements(edgesA, edgesB))
            res = 1
            for l, s in zip(edgesA + edgesB, shapeA + shapeB):
                if (l in clb):
                    if (not diagonal):
                        res *= xplib.xp.sqrt(s)
                    # if single diagonal: then the cost should be output shape
                else:
                    res *= s 

            return int(res + 0.5)

        def getCostTypical(tsA, tsB):
            edgesA, _, diagonalA = self.contractRes[tsA]
            edgesB, _, diagonalB = self.contractRes[tsB]

            if (diagonalA and diagonalB):
                costOrder = 1
            else:
                clb = set(funcs.commonElements(edgesA, edgesB))
                if (diagonalA or diagonalB):
                    costOrder = len(edgesA) + len(edgesB) - 2 * len(clb)
                else:
                    costOrder = len(edgesA) + len(edgesB) - len(clb)
            return typicalDim ** costOrder

        def isSharingBond(tsA, tsB):
            if (isinstance(tsA, int)):
                tsA = self.contractRes[tsA]
            if (isinstance(tsB, int)):
                tsB = self.contractRes[tsB]
            edgesA, _, _ = tsA
            edgesB, _, _ = tsB

            commonEdges = funcs.commonElements(edgesA, edgesB)
            return len(commonEdges) > 0
        
        def isDiagonalOuterProduct(tsA, tsB):
            return tsA[2] and tsB[2] and (not isSharingBond(tsA, tsB))

        def calculateContractRes(tsA, tsB):
            if (isinstance(tsA, int)):
                tsA = self.contractRes[tsA]
            if (isinstance(tsB, int)):
                tsB = self.contractRes[tsB]
            edgesA, shapeA, diagonalA = tsA
            edgesB, shapeB, diagonalB = tsB

            edges = funcs.listSymmetricDifference(edgesA, edgesB)
            shape = [None] * len(edges)

            edgeIndex = dict()
            for i in range(len(edges)):
                edgeIndex[edges[i]] = i

            for l, s in zip(edgesA + edgesB, shapeA + shapeB):
                if (l in edgeIndex):
                    shape[edgeIndex[l]] = s
            return (edges, shape, diagonalA and diagonalB)

        def getSize(tsA, tsB):
            '''
            contract self.contractRes[tsA] and self.contractRes[tsB]
            then give the size of the output tensor
            '''
            _, shape, _ = calculateContractRes(tsA, tsB)
            return funcs.tupleProduct(shape)

        costFunc = None
        if (greedy or (typicalDim is None)):
            costFunc = getCost
        else:
            costFunc = getCostTypical

        def solveSet(x):
            #print('solve_set({})'.format(x))
            if (self.optimalCost[x] is not None):
                return 
            minCost = None
            minTSA = None
            minTSB = None
            localCost = -1
            for tsA in subsetIterator(x):
                tsB = x - tsA
                #print('ss = {}, tt = {}'.format(ss, tt))
                if (self.optimalCost[tsA] is None):
                    solveSet(tsA)
                if (self.optimalCost[tsB] is None):
                    solveSet(tsB)
                if (self.contractRes[x] is None):
                    self.contractRes[x] = calculateContractRes(tsA, tsB)
                
                localCost = costFunc(tsA, tsB)
                currCost = self.optimalCost[tsA] + self.optimalCost[tsB] + localCost
                if (minCost is None) or (currCost < minCost):
                    minCost = currCost
                    minTSA = tsA
                    minTSB = tsB

            self.optimalCost[x] = minCost
            self.optimalSeq[x] = self.optimalSeq[minTSA] + self.optimalSeq[minTSB] + [(lowbitI(minTSA), lowbitI(minTSB))]

        def bruteForce():

            fullSet = (1 << n) - 1

            solveSet(fullSet)
            #print('minimum cost = {}'.format(self.optimalCost[full_s]))
            #print('result = {}'.format(self.contractRes[full_s]))
            return self.optimalSeq[fullSet]

        def greedySearch():
            tensorSet = list(range(n))

            while (len(tensorSet) > 1):
                minSize = -1
                minTSA = -1
                minTSB = -1

                for tsA, tsB in funcs.pairIterator(tensorSet):
                    if (not isSharingBond(tsA, tsB)):
                        continue
                    newSize = getSize(tsA, tsB)
                    if (minSize == -1) or (newSize < minSize):
                        minSize = newSize 
                        minTSA, minTSB = tsA, tsB 

                tsA, tsB = minTSA, minTSB
                self.greedySeq.append((tsA, tsB))
                self.greedyCost += costFunc(tsA, tsB)
                self.contractRes[tsA] = calculateContractRes(tsA, tsB)
                tensorSet.remove(tsB)

            return self.greedySeq

        def capping():
            obj_n = (1 << n)
            new_flag = [True] * obj_n 
            if (typicalDim is None):
                chi_min = min([min(x) for x in shapes])
            else:
                chi_min = typicalDim
            # mu_cap = 1

            chi_min = max(chi_min, 2)
            mu_old = 0
            mu_new = 1

            obj_list = [[] for i in range(n + 1)]
            obj_list[1] = [(1 << x) for x in range(n)]
            full_s = (1 << n) - 1

            def obj_iterator(c1, c2):
                if (len(obj_list[c1]) <= 0) or (len(obj_list[c2]) <= 0):
                    return
                if (c1 == c2):
                    cur1 = 1
                    cur2 = 0
                    while (cur1 < len(obj_list[c1])):
                        yield (obj_list[c1][cur2], obj_list[c1][cur1])
                        cur2 += 1
                        if (cur2 >= cur1):
                            cur1 += 1
                            cur2 = 0
                    return 
                else:
                    cur1 = 0
                    cur2 = 0
                    while (cur1 < len(obj_list[c1])):
                        #print('c1 = {}, c2 = {}, cur1 = {}, cur2 = {}'.format(c1, c2, cur1, cur2))
                        yield (obj_list[c1][cur1], obj_list[c2][cur2])
                        cur2 += 1
                        if (cur2 >= len(obj_list[c2])):
                            cur1 += 1
                            cur2 = 0
                    return

            while (len(obj_list[-1]) == 0):
                #print('mu = {}'.format(mu_new))
                mu_next = mu_new
                # print('mu = {}'.format(mu_new))
                for c in range(2, n + 1):
                    for c1 in range(1, c // 2 + 1):
                        c2 = c - c1
                        for t1, t2 in obj_iterator(c1, c2):
                            #print('t1 = {}, t2 = {}'.format(t1, t2))
                            if ((t1 & t2) != 0):
                                continue 
                            if (isDiagonalOuterProduct(self.contractRes[t1], self.contractRes[t2])):
                                # diagonal outer product is banned
                                continue
                            tt = t1 | t2
                            if (self.contractRes[tt] is None):
                                self.contractRes[tt] = calculateContractRes(t1, t2)

                            if (new_flag[t1] or new_flag[t2]):
                                mu_0 = 0
                            else:
                                mu_0 = mu_old

                            mu_curr = self.optimalCost[t1] + self.optimalCost[t2] + costFunc(t1, t2)
                            if (mu_curr > mu_new):
                                if (mu_next is None) or (mu_next > mu_curr):
                                    mu_next = mu_curr
                                continue

                            if (mu_curr > mu_0) and (mu_curr <= mu_new):
                                if (self.optimalCost[tt] is None):
                                    obj_list[c].append(tt)
                                    #print('append {} to {}'.format(tt, c))
                                if (self.optimalCost[tt] is None) or (self.optimalCost[tt] > mu_curr):
                                    self.optimalCost[tt] = mu_curr
                                    self.optimalSeq[tt] = self.optimalSeq[t1] + self.optimalSeq[t2] + [(lowbitI(t1), lowbitI(t2))]
                                    new_flag[tt] = True

                mu_old = mu_new
                mu_new = max(mu_next, mu_new * chi_min)
                for c in range(n + 1):
                    for tt in obj_list[c]:
                        new_flag[tt] = False

                #print('cost of {} = {}'.format(full_s, self.optimalCost[full_s]))
                #print('length of obj_list = {}'.format([len(x) for x in obj_list]))
            # print('minimum cost = {}'.format(self.optimalCost[full_s]))
            #print('result = {}'.format(self.contractRes[full_s]))
            # print('optimal cost = {}'.format(self.optimalCost[full_s]))
            return self.optimalSeq[full_s]
        if (greedy):
            return greedySearch()
        elif (bf):
            return bruteForce()
        else:
            return capping()

def decodeSubText(t):
    currName = None
    currSize = None
    nameWaitFlag = False
    sizeWaitFlag = False

    labels = []
    shape = []

    for i in range(len(t)):
        if (t[i] == '{'):
            if (currName is None):
                nameWaitFlag = True
            else:
                sizeWaitFlag = True
        elif (t[i] == '}'):
            assert (nameWaitFlag or sizeWaitFlag), funcs.errorMessage(err = 'not maching parenthesis in text {}'.format(t), location = 'CTL.tensor.contract.tensorGraph.decodeSubText')
            if nameWaitFlag:
                nameWaitFlag = False
            else:
                sizeWaitFlag = False
                print(currName, currSize)
                labels.append(currName)
                shape.append(currSize)

                currName = None
                currSize = None
        
        else:
            if (currName is None) or (nameWaitFlag):
                if (currName is None):
                    currName = ''
                currName += t[i]
            elif (not t[i].isdigit()):
                labels.append(currName)
                shape.append(currSize)

                currName = None
                currSize = None

                currName = t[i]
                if (sizeWaitFlag):
                    nameWaitFlag = True
                    sizeWaitFlag = False
            else:
                # assert t[i].isdigit(), funcs.errorMessage('{} contains non-digit character for leg size'.format(t), location = 'CTL.tensor.contract.tensorGraph.decodeSubText')
                if (currSize is None):
                    currSize = 0
                currSize = currSize * 10 + int(t[i])
                if (not sizeWaitFlag):
                    labels.append(currName)
                    shape.append(currSize)

                    currName = None
                    currSize = None

    if (currName is not None):
        labels.append(currName)
        shape.append(currSize)
    print(t, labels, shape)
    return labels, shape

def createTensorFromText(t):
    assert (len(t) > 0) and (t[0].isupper()) and (t[1:].islower()), funcs.errorMessage(err = '{} is not a valid string for single tensor'.format(t), location = 'CTL.tensor.tensorGraph.createTensorFromText')
    name = t[0]
    labels, shape = decodeSubText(t[1:])
    # print(labels, shape)
    tensor = TensorLike(labels = labels, name = name, shape = shape)
    return tensor

def createTensorListFromText(t):
    l = len(t)
    last = 0
    links = dict()
    tensors = []
    for i in range(l + 1):
        if (i == l) or ((i > 0) and t[i].isupper()):
            tensor = createTensorFromText(t[last : i])
            last = i
            for leg in tensor.legs:
                if (leg.name in links):
                    links[leg.name].append(leg)
                else:
                    links[leg.name] = [leg]

            tensors.append(tensor)

    for name in links:
        assert len(links[name]) <= 2, funcs.errorMessage(err = 'leg name {} appeared more than twice.'.format(name), location = 'CTL.tensor.tensorGraph.createTensorListFromText')
        if len(links[name]) == 2:
            leg0 = links[name][0]
            leg1 = links[name][1]
            if (leg0.dim is None):
                leg0.dim = leg1.dim
            elif (leg1.dim is None):
                leg1.dim = leg0.dim
            makeLink(leg0, leg1)
            # print('make link between {} and {}'.format(links[name][0], links[name][1]))
    
    return tensors