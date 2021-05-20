from CTL.funcs.graph import UndirectedGraph
import numpy as np

class TensorGraph(UndirectedGraph):
    def __init__(self, n):
        super().__init__(n)

    def addFreeEdge(self, idx, weight = None):
        self.v[idx].addEdge(None, weight = weight)

    def optimalContractSequence(self, bf = False, typical_dim = 10):

        def lowbit_i(x):
            return (x & (-x)).bit_length() - 1
        def lowbit(x):
            return (x & (-x))

        def subset_iterator(x):
            lb = lowbit(x)
            while (lb * 2 < x):
                yield lb 
                nlb = lowbit(x - lb)
                lb = (lb & (~(nlb - 1))) + nlb
            return 

        labels = [t.labels for t in tensor_list]
        shapes = [list(t.shape) for t in tensor_list]
        n = len(labels)

        opt_cost = [None] * (1 << n)
        opt_seq = [None] * (1 << n)
        con_res = [None] * (1 << n)

        for i in range(n):
            opt_cost[(1 << i)] = 0
            opt_seq[(1 << i)] = []
            con_res[(1 << i)] = (labels[i], shapes[i])

        def get_cost(ss, tt):
            label_ss, shape_ss = con_res[ss]
            label_tt, shape_tt = con_res[tt]

            clb = set(funcs.commonElements(label_ss, label_tt))
            res = 1
            for l, s in zip(label_ss + label_tt, shape_ss + shape_tt):
                if (l in clb):
                    res *= np.sqrt(s)
                else:
                    res *= s 

            return np.int(res + 1e-10)

        def get_cost_typical(ss, tt):
            label_ss, shape_ss = con_res[ss]
            label_tt, shape_tt = con_res[tt]
            clb = funcs.commonElements(label_ss, label_tt)

            cost_order = len(label_ss) + len(label_tt) - len(clb)
            return typical_dim ** cost_order

        def calc_con_res(ss, tt):
            label_ss, shape_ss = con_res[ss]
            label_tt, shape_tt = con_res[tt]

            label_res = funcs.listSymmetricDifference(label_ss, label_tt)
            label_dict = dict()
            for i in range(len(label_res)):
                label_dict[label_res[i]] = i 

            shape_res = [None] * len(label_res)
            for l, s in zip(label_ss + label_tt, shape_ss + shape_tt):
                if (l in label_dict):
                    shape_res[label_dict[l]] = s 
            return (label_res, shape_res)

        cost_func = None
        if (typical_dim is None):
            cost_func = get_cost
        else:
            cost_func = get_cost_typical

        def solve_set(x):
            #print('solve_set({})'.format(x))
            if (x in opt_cost):
                return 
            min_cost = None
            min_ss = None
            min_tt = None
            local_cost = -1
            for ss in subset_iterator(x):
                tt = x - ss
                #print('ss = {}, tt = {}'.format(ss, tt))
                if (opt_cost[ss] is None):
                    solve_set(ss)
                if (opt_cost[tt] is None):
                    solve_set(tt)
                if (con_res[x] is None):
                    con_res[x] = calc_con_res(ss, tt)
                
                local_cost = cost_func(ss, tt)
                curr_cost = opt_cost[ss] + opt_cost[tt] + local_cost
                if (min_cost is None) or (curr_cost < min_cost):
                    min_cost = curr_cost
                    min_ss = ss 
                    min_tt = tt

            opt_cost[x] = min_cost 
            opt_seq[x] = opt_seq[min_ss] + opt_seq[min_tt] + [(lowbit_i(min_ss), lowbit_i(min_tt))]

        def brute_force():

            full_s = (1 << n) - 1

            solve_set(full_s)
            #print('minimum cost = {}'.format(opt_cost[full_s]))
            #print('result = {}'.format(con_res[full_s]))
            return opt_seq[full_s]

        def capping():
            obj_n = (1 << n)
            new_flag = [True] * obj_n 
            if (typical_dim is None):
                chi_min = min([min(x) for x in shapes])
            else:
                chi_min = typical_dim
            mu_cap = 1
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
                for c in range(2, n + 1):
                    for c1 in range(1, c // 2 + 1):
                        c2 = c - c1
                        for t1, t2 in obj_iterator(c1, c2):
                            #print('t1 = {}, t2 = {}'.format(t1, t2))
                            if ((t1 & t2) != 0):
                                continue 
                            tt = t1 | t2
                            if (con_res[tt] is None):
                                con_res[tt] = calc_con_res(t1, t2)

                            if (new_flag[t1] or new_flag[t2]):
                                mu_0 = 0
                            else:
                                mu_0 = mu_old

                            mu_curr = opt_cost[t1] + opt_cost[t2] + cost_func(t1, t2)
                            if (mu_curr > mu_new):
                                if (mu_next is None) or (mu_next > mu_curr):
                                    mu_next = mu_curr
                                continue

                            if (mu_curr > mu_0) and (mu_curr <= mu_new):
                                if (opt_cost[tt] is None):
                                    obj_list[c].append(tt)
                                    #print('append {} to {}'.format(tt, c))
                                if (opt_cost[tt] is None) or (opt_cost[tt] > mu_curr):
                                    opt_cost[tt] = mu_curr
                                    opt_seq[tt] = opt_seq[t1] + opt_seq[t2] + [(lowbit_i(t1), lowbit_i(t2))]
                                    new_flag[tt] = True

                mu_old = mu_new
                mu_new = max(mu_next, mu_new * chi_min)
                for c in range(n + 1):
                    for tt in obj_list[c]:
                        new_flag[tt] = False

                #print('cost of {} = {}'.format(full_s, opt_cost[full_s]))
                #print('length of obj_list = {}'.format([len(x) for x in obj_list]))
            print('minimum cost = {}'.format(opt_cost[full_s]))
            #print('result = {}'.format(con_res[full_s]))
            return opt_seq[full_s]

        if (bf):
            return brute_force()
        else:
            return capping()
