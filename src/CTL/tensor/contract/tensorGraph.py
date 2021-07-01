from CTL.funcs.graph import UndirectedGraph
import CTL.funcs.funcs as funcs
import numpy as np

class TensorGraph(UndirectedGraph):
    def __init__(self, n, diagonalFlags = None):
        super().__init__(n)
        self.indexed = False
        if (diagonalFlags is None):
            self.diagonalFlags = [False] * n
        else:
            self.diagonalFlags = diagonalFlags

    def addFreeEdge(self, idx, weight = None):
        self.v[idx].addEdge(None, weight = weight)

    def addEdgeIndex(self):
        if (self.indexed):
            return
        edges = self.getEdges()
        m = len(edges)
        for i in range(m):
            edges[i].index = i 
        self.indexed = True

    def optimalCostResult(self):
        if (self.optimalCost is None):
            self.optimalContractSequence()
        return self.optimalCost[(1 << self.n) - 1]
    
    def optimalContractSequence(self, bf = False, typicalDim = 10):

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

        self.optimalCost = [None] * (1 << n)
        self.optimalSeq = [None] * (1 << n)
        self.contractRes = [None] * (1 << n)

        for i in range(n):
            self.optimalCost[(1 << i)] = 0
            self.optimalSeq[(1 << i)] = []
            self.contractRes[(1 << i)] = (edges[i], shapes[i], self.diagonalFlags[i])

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
                        res *= np.sqrt(s)
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

        def calculateContractRes(tsA, tsB):
            edgesA, shapeA, diagonalA = self.contractRes[tsA]
            edgesB, shapeB, diagonalB = self.contractRes[tsB]

            edges = funcs.listSymmetricDifference(edgesA, edgesB)
            shape = [None] * len(edges)

            edgeIndex = dict()
            for i in range(len(edges)):
                edgeIndex[edges[i]] = i

            for l, s in zip(edgesA + edgesB, shapeA + shapeB):
                if (l in edgeIndex):
                    shape[edgeIndex[l]] = s
            return (edges, shape, diagonalA and diagonalB)

        costFunc = None
        if (typicalDim is None):
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

        def capping():
            obj_n = (1 << n)
            new_flag = [True] * obj_n 
            if (typicalDim is None):
                chi_min = min([min(x) for x in shapes])
            else:
                chi_min = typicalDim
            # mu_cap = 1
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

        if (bf):
            return bruteForce()
        else:
            return capping()
            