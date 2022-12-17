from typing import Union
from BayesNet import BayesNet
import copy
import pandas as pd
import itertools
import numpy as np
from itertools import combinations
import networkx as nx
import dwave_networkx as dnx


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net
        
        self.cpt = copy.deepcopy(self.bn.get_all_cpts())
        self.graph = self.bn.get_interaction_graph()
        self.factors = copy.deepcopy(self.bn.get_all_cpts())
        
    # Methods 
    def del_leaf_nodes(self, Q, e):
        '''Delete every leaf nodes of BN not in Q and e
        Args
        Q : list of query variables
        '''
        vars_q_e = Q + list(e.keys())
        all_variables = set(self.factors.keys()).difference(set(vars_q_e)) # vars not in e and Q
        
        # if no children, delete
        for variable in all_variables:
            successors = [c for c in self.bn.structure.successors(variable)]
            if len(successors) == 0 : 

                # Delete leaf node
                self.bn.del_var(variable)
                print(f'Leaf node {variable} deleted')
                
                # Delete factor in factors cpt
                del self.factors[variable]
                
                        
    def apply_pruning(self, Q, e):
        '''Deletes every leaf node 𝑊 ∉𝑋∪𝑌∪𝑍,
        Deletes all edges outgoing from nodes in 𝑍
        Performs both rules iteratively until they cannot be applied anymore'''    
        
        #e = list(e.keys())
        
        # 1. Delete all edges outgoing from evidence Z
        for evidence in e :
            successors = self.bn.structure.successors(evidence)
            for successor in [c for c in successors]:               
                self.bn.del_edge((evidence, successor))
                print('Edges deleted between', (evidence, successor)) 
                
        # update factors cpt table for edge pruning       
        # get factors containing evidence and sum-out w.r. e
        all_variables = list(self.factors.keys())
        
        for evidence in e:
            for variable in all_variables:           
                # get factors containing evidence
                factor_var = self.factors[variable]
                variables_in_factor = list(factor_var.columns.map(''.join))
                if evidence in variables_in_factor and variable != evidence: # if not evidence itself
                    self.factors[variable] = self.marginalization_factor(self.factors[variable], evidence)
                    self.graph.add_node(evidence) # should keep node in interaction graph (marginalization deletes it)
        
        #print(node for node in self.graph.nodes)
        # 2. Delete leaf nodes (no children) not in Q and e 
        self.del_leaf_nodes(Q, e)
        
    def find_ancestors(self, variables):
        """
        Find all nodes that have descendants.
        Args:
            variables : a list of strings, names of the nodes. 
        Returns:
            a list of strings for the nodes' names with ancestors
        """
        visit_nodes = copy.copy(variables) ## nodes to visit
        ancestors = set() ## observed nodes and their ancestors

        while len(visit_nodes) > 0:
            next_node = visit_nodes.pop() 
            ## add parents
            for parent in self.bn.structure.predecessors(next_node):
                ancestors.add(parent)
                
        return ancestors

    def check_one_dseparation(self, x, y, Z):
        """
        Check whether variable x and variable y are d-separated given Z (observed).
        Algorithm from the "Reachable" procedure in  Koller and Friedman (2009), 
        Probabilistic Graphical Models: Principles and Techniques, p 75.
        Args:
            X: a string, name of the first query node
            Y: a string, name of the second query node
            Z: a list of strings, names of the observed nodes. 
        """
        
        if type(Z) is not list:
            Z = [Z]
            
        obs_ancestors = self.find_ancestors(Z) ## get all nodes having observed descendants

        ## Try all paths from the node X (start) to node Y (end).
        ## If paths reach the node Y, 
        ## then X and Y are not d-separated.
        via_nodes = [(x, "up")]
        visited = set() ## keep track of visited nodes to avoid cyclic paths
        
        while len(via_nodes) > 0: 
            (variable, direction) = via_nodes.pop()
        
            node = variable

            ## skip visited nodes
            if (variable, direction) not in visited:
                visited.add((variable, direction)) 

                ## if reaches the node "end", then it is not d-separated
                if variable not in Z and variable == y:
                    #print(f'{x} and {y} are not d-separated, given {Z}')
                    return False
                
                if direction == "up" and variable not in Z:
                    for parent in self.bn.structure.predecessors(variable):
                        via_nodes.append((parent, "up"))
                    for child in self.bn.structure.successors(variable):
                        via_nodes.append((child, "down"))
                        
                ## if traversing from parents, then need to check v-structure
                elif direction == "down":
                    ## path to children is always active
                    if variable not in Z: 
                        for child in self.bn.structure.successors(variable):
                            via_nodes.append((child, "down"))
                    ## path to parent forms a v-structure
                    if variable in Z or variable in obs_ancestors: 
                        for parent in self.bn.structure.predecessors(variable):
                            via_nodes.append((parent, "up"))
        
        #print(f'{x} and {y} are d-separated, given {Z}') 
        return True
    
    def check_set_dseparation(self, X, Y, Z):
        """
        Check whether set X and set Y are d-separated given set Z (observed).
        Args:
            X: a list of strings, names of the first query node
            Y: a list of strings, name of the second query node
            Z: a list of strings, names of the observed nodes. 
        """
        for x in X:
            for y in Y:
                d_separated = self.check_one_dseparation(x, y, Z)  
                if not d_separated :
                    print(f'{X} and {Y} are not d-separated, given {Z}')
                    return False
        print(f'{X} and {Y} are d-separated, given {Z}')
        return True
                
    
    def check_independence(self, X, Y, Z):
        print(f'{X} and {Y} are independent given {Z}?')
        print('Based on implication d-separation -> independence')
        return self.check_set_dseparation(X, Y, Z) 

    def marginalization_factor(self, factor, x):
        """ This method marginalizes out a factor 
        by summing over all possible value combinations of x
        factor :  BN factor
        x : str variable to sum-out
        """
        factor.columns = factor.columns.map(''.join)

        if x not in list(factor.columns):
            #print(f'variable {x} not found in cpt')
            return factor
                
        cpt_ext_f = factor.drop([x, "p"], axis=1)
        marginalized_cpt = cpt_ext_f.drop_duplicates().reset_index(drop= True)
    
        n_assignments = len(marginalized_cpt)
        n_rows_cpt = len(factor)
        
        cps =[]
        
        # sum conditional probability for same assignments
        for i in range(n_assignments):
            ass = [val for val in marginalized_cpt.iloc[i]]
            
            p = 0
            for j in range(n_rows_cpt):
                row_to_check = [val for val in cpt_ext_f.iloc[j]]
                
                if row_to_check == ass:
                    p += factor.iloc[j]['p']
                    
            cps.append(p)
        # Add to dataframe
        marginalized_cpt['p'] = cps
        
        # add edge between neighbors of x if non existent
        neighbors = [n for n in self.graph.neighbors(x)]
        combination_neighbors = list(set(combinations(neighbors, 2)))
        
        for i, pair in enumerate(combination_neighbors):
            
            if (combination_neighbors!= []) and ((combination_neighbors[i][0], combination_neighbors[i][1]) 
            and (combination_neighbors[i][1], combination_neighbors[i][0]) 
            not in self.graph.edges):
                    
                # add non-existent edge
                self.graph.add_edge(pair[0], pair[1])
                
        # delete node from interaction graph
        self.graph.remove_node(x)
       
        return marginalized_cpt
    
    def factor_multiplication(self, factor_f, factor_g):
        
        '''Params
        factor_f : cpt factor f
        factor_g : cpt factor g
        '''
        
        factor_f.columns = factor_f.columns.map(''.join)
        factor_g.columns = factor_g.columns.map(''.join)
        
        #print('factor f', factor_f)
        #print('factor g', factor_g)
        
        variables_factor_f = list(factor_f.columns.values)
        variables_factor_g = list(factor_g.columns.values)
        variables_factor_f.remove('p')
        variables_factor_g.remove('p')
        
        variables_rm_f = [x for x in variables_factor_f if not x.startswith('instantiation_')]
        variables_rm_g = [x for x in variables_factor_g if not x.startswith('instantiation_')]
        
        variables_wt_f = [x for x in variables_factor_f if x.startswith('instantiation_')]
        variables_wt_g = [x for x in variables_factor_g if x.startswith('instantiation_')]

        assignment_variables = np.unique(variables_rm_f + variables_rm_g)
        all_variables = np.unique(variables_factor_f + variables_factor_g)
        
        table = list(itertools.product([False, True], repeat=len(assignment_variables))) 
        table_list = [list(ele) for ele in table]
        multiplication_factor = pd.DataFrame(table_list, columns = [assignment_variables])
                
        products = []
        assignments_logbook = []
        #inst_f_list = []
        
        for ass in table:
            assignments_log = []
            assignment = dict(zip(assignment_variables, ass))
            
            ## print(assignment)
            instantiation = pd.Series(assignment)
             
            # get compatible instanciation table for factor 1
            factor_f_compat = self.bn.get_compatible_instantiations_table(instantiation, factor_f)
            factor_g_compat = self.bn.get_compatible_instantiations_table(instantiation, factor_g)
            
            if len(factor_g_compat['p'].values) == 0 or len(factor_f_compat['p'].values) == 0:
                product = 0
     
            else :
                # first get product 
                product = factor_g_compat['p'].values[0] * factor_f_compat['p'].values[0]
                
                # keep track of instantiation
                indx_f_comp = factor_f_compat.index.values[0]
                indx_g_comp = factor_g_compat.index.values[0]
                #print('INDEX G COMPATIBLE', indx_g_comp)
                #print('INDEX F COMPATIBLE', indx_f_comp)
                
                # if has previously been maxed-out (i.e. there is an 'instantiation_x column) 
                if len(variables_wt_f) != 0 :
                    var_f = factor_f.loc[indx_f_comp]
                    inst_f = var_f.index[len(variables_rm_f)+1:][0]
                    var_f = var_f.values[len(variables_rm_f)+1:][0]
                    assignments_log.append((inst_f, var_f))
                    
                if len(variables_wt_g) !=0: 
                    var_g = factor_g.loc[indx_g_comp]
                    inst_g = var_g.index[len(variables_rm_g)+1:][0]
                    var_g = var_g.values[len(variables_rm_g)+1:][0]
                    #print(inst_g)
                    assignments_log.append((inst_g, var_g))
            
            products.append(product)  
            if len(variables_wt_g) !=0 or len(variables_wt_f) !=0:
                assignments_logbook.append(assignments_log)
            
        multiplication_factor['p'] = products
            
        if len(variables_wt_g) !=0 or len(variables_wt_f) !=0:
            multiplication_factor['instantiation_log'] = assignments_logbook
        
        return multiplication_factor
            
    def maxing_out(self, factor, x):
        # has to keep track of which instantiation of X led to the maximized value
        """ 
        Args
        factor : cpt of the BN factor
        x : str variable to max-out
        """
        # probs = factor["p"]
        cpt_ext_f = factor.drop([x, "p"], axis=1)
        #factor.loc[:, ~factor.columns.str.startswith('instantiation_')]
        cpt_ext_f = cpt_ext_f[cpt_ext_f.columns.drop(list(cpt_ext_f.filter(regex='instantiation_')))]
        #print('REMOVED P AND INST', cpt_ext_f)
        
        if len(cpt_ext_f.columns) == 0 :
            #print('Empty')
            indx_max = factor['p'].idxmax()
            maxed_out_factor = factor.iloc[indx_max]
            #maxed_out_cpt['p'] = cps
            #maxed_out_cpt[f'instantiation_{x}'] = X_instantiations
            #print(summed_out_factor)
            return maxed_out_factor
         
        #factor.columns[factor.columns.str.startswith('instantiation_')]
        maxed_out_cpt = cpt_ext_f.drop_duplicates().reset_index(drop= True)
        n_assignments = len(maxed_out_cpt)
        n_rows_cpt = len(factor)
        
        # get values of the first row
        cps =[]
        X_instantiations = []
        X_instantiations_map = []
        
        # find max conditional probability for same assignments
        for i in range(n_assignments):
            ass = [val for val in maxed_out_cpt.iloc[i]]
            p_max = -1
            
            for j in range(n_rows_cpt):
                row_to_check = [val for val in cpt_ext_f.iloc[j]]
                                
                if row_to_check == ass:
                    if p_max < factor.iloc[j]['p']:
                        p_max = factor.iloc[j]['p']
                        ins_x = factor.iloc[j][x]
             
            X_instantiations.append(ins_x)    
            cps.append(p_max)
            
            #ind_max = np.argmax(cps)
            #X_instantiations_map.append(X_instantiations[ind_max])
        
        #print(X_instantiations)
        # Add to dataframe
        maxed_out_cpt['p'] = cps
        maxed_out_cpt[f'instantiation_{x}'] = X_instantiations
        
        #print(maxed_out_cpt)
        return maxed_out_cpt #, X_instantiations
    
    def ordering_MinDegreeHeuristic(self, X):
        '''Minimimum Degree Heuristic
        Gets the number of edge (degree) for each node
        
        Args
        X : list of variables
        returns: str variable with minimum degree, to eliminate first
        
        '''
        # get degree for every variable
        degree = dict(self.graph.degree)
        
        # build new df with only keys in interaction graph
        x_degree = {node: degree[node] for node in X}
        
        # order dict 
        ordered_x_degree = sorted(x_degree.items(), key=lambda item: item[1])
        
        # get full order of variables
        full_order = [var[0] for var in ordered_x_degree]
        
        # get variable with min degree
        min_degree = min(x_degree, key=x_degree.get) # should this be able to deal with ties?
        
        return min_degree, full_order, ordered_x_degree
        
        
    def ordering_MinFillHeuristic(self, X):
        '''Minimimum Fill Heuristic
        For each current node of set X,  count the number of
        neighbors not already connected (no edges between neighbors)
        
        Args
        X : list of variables
        return : str variable with minimum count of neighbors not connected, to eliminate first
        
        '''
        dict = {}
        ## print(self.bn.get_interaction_graph().edges)
        ## print(dnx.min_fill_heuristic(self.graph))
        
        for var in X:
            dict[var] = 0
            
            # compute n edges between neighbors of X that are not connected already
            neighbors_x = [n for n in self.graph.neighbors(var)]
            #print(f'Neighbors of {var}', neighbors_x)
            
            # if not edges, add to count
            combination_neighbors = list(set(combinations(neighbors_x, 2)))
            #print(combination_neighbors)
            
            if combination_neighbors == []:
                dict[var] += 0
                                        
            else:
                for i, pair in enumerate(combination_neighbors):
                    #print(combination_neighbors[i][0])
                    #print(combination_neighbors[i][1])

                    if (combination_neighbors[i][0], combination_neighbors[i][1]) and (combination_neighbors[i][1], combination_neighbors[i][0]) not in self.graph.edges:
                        #print('Edge not found', (combination_neighbors[i][0], combination_neighbors[i][1]))
                        dict[var] += 1 
                    
                # else:
                    # print('Edge already exists', (neighbor, var))         
            
        # build new df with only keys in interaction graph
        x_dict = {node: dict[node] for node in X}

        # order dict
        ordered_x_dict = sorted(x_dict.items(), key=lambda item: item[1])
        
        # get full order of variables as list
        full_order = [var[0] for var in ordered_x_dict]
        
        # get minimum fill
        min_fill = min(x_dict, key=x_dict.get) 
        
        return min_fill, full_order, ordered_x_dict
        
    def one_variable_elimination(self, x_target):
        '''Sum out one target variable with variable elimination
        Args 
        x_target : string of target variable name
        
        returns : Factors after the target variable has been eliminated.
        '''        
        all_variables = list(self.factors.keys())
        
        # get all factors that contain the target variable
        variables_product = []
        for variable in all_variables:
            # get factor
            factor_var = self.factors[variable]
            variables_in_factor = list(factor_var.columns.map(''.join))
            
            #print('variable in factor', variables_in_factor)
            
            if x_target in variables_in_factor:
                variables_product.append(variable)
                                    
        # compute the product of all factors containing the target variable        
        if len(variables_product) > 1:
            factor_f = self.factors[variables_product[0]]
            factor_g = self.factors[variables_product[1]]
            product_factor = self.factor_multiplication(factor_f, factor_g)
            
        else:
            if variables_product == []:
                print('There factors left to reduce with respect to target variable')
                return self.factors
            
            else : # only one factor to consider
                product_factor = self.factors[variables_product[0]]
        
        if len(variables_product) > 2:
            for i in range(2, len(variables_product)):
                new_factor = self.factors[variables_product[i]]
                product_factor = self.factor_multiplication(product_factor, new_factor)
                
        # sum out the target variable from the product factor
        summed_out_factor = self.marginalization_factor(product_factor, x_target)
        
        # remove all factors containing this target variable from cpt
        #print('FACTOR BEFORE MARGINAL', self.factors)
        for variable in variables_product:
            #print(variable)
            del self.factors[variable]
        #    self.bn.del_var(variable)

        # add the reduced factor to the list of factors
        self.factors[x_target] = summed_out_factor
        
        #print('FACTOR AFTER MARGINALIZATION', self.factors)
        return summed_out_factor
    
    
    def set_variable_elimination(self, x_target_set):
        '''Sum out a set of variable with variable elimination
        Args 
        X : list of variables where index = order for elimination -> 0, first to be eliminated
        
        returns : List of factors after the variables have been eliminated.
        '''            
    
        # remove variables outside of set from list ordering
        for target in x_target_set:
            new_factor = self.one_variable_elimination(target)
         
        return new_factor
            
    def compute_marginal_distribution(self, Q, e, heuristic):    
        '''Compute marginal distribution
        Args
        Q: list of variable names, Query -> for which we want to know probability
        e: pandas series with evidence, ex. pd.Series({'A': True; 'B' : False})
        heuristic : min_degree(), min_fill(), or random()
        '''
        # get all variables in evidence
        variables_e = list(e.keys()) # evidence
        
        all_variables = list(self.bn.get_all_variables())
        
        # reduce all factors with respect to e
        for variable in all_variables:
            cpt_var = self.factors[variable]
            reduced_cpt = self.bn.get_compatible_instantiations_table(pd.Series(e), cpt_var)
            self.factors[variable] = reduced_cpt                  
        not_q_variables = list(set(all_variables).difference(set(Q)))
        
        while len(not_q_variables) > 0:
                        
            # pick variable ordering
            if heuristic == 'min_fill':
                min_var, _, _ = self.ordering_MinFillHeuristic(not_q_variables)
            
            elif heuristic == 'min_degree':
                min_var, _, _ = self.ordering_MinDegreeHeuristic(not_q_variables)
                
            #print('Marginalizing w.r.:', min_var)
            
            ### compute joint marginal pr(Q & e) ###
            # find all factors that contain var
            # compute product of the factors
            # sum-out variable
            self.one_variable_elimination(min_var)
            
            # update evidence left to sum-out
            not_q_variables.remove(min_var)
              
        nfactors = list(self.factors.keys()) 
        product_factor = self.factors[nfactors[0]]
        
        # compute the product of resulting factors    
        if len(nfactors) > 1:    
            for name in (nfactors[1:]): 
                product_factor = self.factor_multiplication(product_factor, self.factors[name])    
        
        # get pr(Q) corresponding to truth assignment for Q
        Q_true = dict(zip(Q, [True for i in range(len(Q))]))
        pr_e_Q = self.bn.get_compatible_instantiations_table(pd.Series(Q_true), product_factor)
              
        pr_e_Q = pr_e_Q.sum()['p']
        print('pr(Q & e)', pr_e_Q)
        
        # from previous, sum out Q to get p(e)
        pr_e = product_factor.sum()['p']
        print('pr(e)', pr_e)
        
        # compute pr(Q & e) / pr(e)
        pr_Q = pr_e_Q / pr_e
        print('pr(Q)', pr_Q)
        
        return pr_Q, product_factor
    
    def MAP(self, Q, e, heuristic):
        _, product_factor = self.compute_marginal_distribution(Q, e, heuristic)
        map_instantiation = {}
        
        #print(self.factors)
        #print(Q)
        
        for q in Q:
            #print('Maxing out w.r.t.', q)
            #print(self.factors)
        
            # find all factors that contain variable
            # compute product of the factors
            # max-out variable
            variables_product = []
            
            for variable in list(self.factors.keys()):
                # get factor
                factor_var = self.factors[variable]
                variables_in_factor = list(factor_var.columns.map(''.join))
                        
                if q in variables_in_factor:
                    variables_product.append(variable)
                    
            #print(variables_product)
            # compute the product of factors containing the target variable        
            if len(variables_product) > 1:
                factor_f = self.factors[variables_product[0]]
                factor_g = self.factors[variables_product[1]]
                product_factor = self.factor_multiplication(factor_f, factor_g)
                #print(product_factor)
            
            else:
                if variables_product == []:
                    print('Nothing to max-out with respect to target variable')
                    return self.factors
            
                else : # only one factor to consider
                    product_factor = self.factors[variables_product[0]]
        
            if len(variables_product) > 2:
                for i in range(2, len(variables_product)):
                    new_factor = self.factors[variables_product[i]]
                    product_factor = self.factor_multiplication(product_factor, new_factor)    
            
            #print('PRODUCT FACTOR', product_factor)
            maxed_out_factor = self.maxing_out(product_factor, q)
            
            #print('MAXED OUT FACTOR', maxed_out_factor)
            #map_instantiation[q]= instantiation_map
            
            # update factors
            for variable in variables_product:
                del self.factors[variable]

            # add the maxed-out factor to the list of factors
            self.factors[q] = maxed_out_factor
        
        #print(self.factors)
        #max = maxed_out_factor.loc[maxed_out_factor['p'].idxmax()]
        #map_instantiation = max.drop(columns = 'p') 
        return  maxed_out_factor #, map_instantiation
        
    def MEP(self, e, heuristic):
        '''Get instantiation for all variables given the evidence'''
        # network pruning 
        Q = list(self.factors.keys())
        self.apply_pruning(Q, e)
        
        # reduce all factors with respect to e
        #print('Factors prior to reducing \n', self.factors)
        
        for variable in (self.factors.keys()):
            ## print(variable)
            cpt_var = self.factors[variable]
            reduced_cpt = self.bn.get_compatible_instantiations_table(pd.Series(e), cpt_var)
            self.factors[variable] = reduced_cpt
        
        #print('Factors after reducing \n', self.factors)
        all_variables = list(self.factors.keys())
        
        # get all variables not e
        not_e_variables = list(set(all_variables).difference(set(e)))
        print('Not evidence variables', not_e_variables)
        
        while len(not_e_variables) > 0:

            #print('VARIABLES LEFT TO MAX OUT (NOT E)', not_e_variables)
            
            # pick variable ordering
            if heuristic == 'min_fill':
                min_var, _, _= self.ordering_MinFillHeuristic(not_e_variables)
            
            elif heuristic == 'min_degree':
                min_var, _, _= self.ordering_MinDegreeHeuristic(not_e_variables)
                
            print('Maxing out w.r.:', min_var)
            
            # find all factors that contain variable
            # compute product of the factors
            # max-out variable
            variables_product = []
            
            for variable in list(self.factors.keys()):
                # get factor
                factor_var = self.factors[variable]
                variables_in_factor = list(factor_var.columns.map(''.join))
            
                #print('variable in factor', variables_in_factor)
            
                if min_var in variables_in_factor:
                    variables_product.append(variable)
            
            # compute the product of all factors containing the target variable        
            if len(variables_product) > 1:
                factor_f = self.factors[variables_product[0]]
                factor_g = self.factors[variables_product[1]]
                product_factor = self.factor_multiplication(factor_f, factor_g)
            
            else:
                if variables_product == []:
                    print('Nothing to max-out with respect to target variable')
                    return self.factors
            
                else : # only one factor to consider
                    product_factor = self.factors[variables_product[0]]
        
            if len(variables_product) > 2:
                for i in range(2, len(variables_product)):
                    new_factor = self.factors[variables_product[i]]
                    product_factor = self.factor_multiplication(product_factor, new_factor)    
                        
            #reduced_factor = self.bn.get_compatible_instantiations_table(pd.Series(e), product_factor)
            #print('Product', product_factor)  
            
            maxed_out_factor = self.maxing_out(product_factor, min_var)            
            #print('MAXED-OUT', maxed_out_factor)
            #print('MEP INSTANTIATION', instantiation_with_map)
            
            # update factors
            for variable in variables_product:
                del self.factors[variable]

            # add the maxed-out factor to the list of factors
            self.factors[min_var] = maxed_out_factor
            
            # update evidence left to sum-out
            not_e_variables.remove(min_var)        
        
        #print(self.factors)
        return product_factor #, product_factor.loc[:, ['instantiation_log']] #maxed_out_factor.loc['instantiation_log']

#######################################
