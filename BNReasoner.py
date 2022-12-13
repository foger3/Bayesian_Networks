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
            self.heuristic = 'random'
        
        self.cpt = copy.deepcopy(self.bn.get_all_cpts())
        self.graph = self.bn.get_interaction_graph()
        self.factors = copy.deepcopy(self.bn.get_all_cpts())
        
    # TODO: This is where your methods should go
    def del_leaf_nodes(self, variables):
        '''Delete leaf nodes of BN'''
        all_variables = set(self.bn.get_all_variables()) - set(variables)
        for variable in all_variables:
            successors = [c for c in self.bn.structure.successors(variable)]
            if len(successors) == 0 :
                # Delete leaf node
                self.bn.del_var(variable)
        
    def apply_pruning(self, X, Y, Z):
        '''Deletes every leaf node 𝑊 ∉𝑋∪𝑌∪𝑍,
        Deletes all edges outgoing from nodes in 𝑍
        Performs both rules iteratively until they cannot be applied anymore'''
                
        if type(X) is not list:
            X = [X]
        if type(Y) is not list:
            Y = [Y]
        if type(Z) is not list:
            Z = [Z]
            
        # reduce factors?
        #reduce_factor(instantiation: pd.Series, cpt: pd.DataFrame)
        
        variables = X + Y + Z
        ## print(variables)
        
        # 1. Delete every leaf nodes : that don't have children        
        self.del_leaf_nodes(variables)
        
        # 2. Delete all edges outgoing from Z
        ## find children
        for evidence in Z :
            ## print('evidence', evidence)
            
            successors = self.bn.structure.successors(evidence)
            for successor in [c for c in successors]:
                ## print('successor', successor)
                
                print(evidence, successor)
                self.bn.del_edge((evidence, successor))
            
        self.del_leaf_nodes(variables)
    
    
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
            ## print('visited nodes', visit_nodes)
            next_node = visit_nodes.pop() 
            ## print('next node', next_node)
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
        
        ## print(via_nodes)

        while len(via_nodes) > 0: 
            (variable, direction) = via_nodes.pop()
            
            #print(via_nodes)
            #Sprint(nname)
            node = variable

            ## skip visited nodes
            if (variable, direction) not in visited:
                visited.add((variable, direction)) 

                ## if reaches the node "end", then it is not d-separated
                if variable not in Z and variable == y:
                    #print(f'{X} and {Y} are not d-separated, given {Z}')
                    return False
                
                ## if traversing from children, then it won't be a v-structure
                ## the path is active as long as the current node is unobserved
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
        
        ## print(f'{X} and {Y} are d-separated, given {Z}') 
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
                    return False
        
        return True
                
    
    def check_independance(self, X, Y, Z):
        return self.check_set_dseparation(X, Y, Z) 

    def marginalization_factor(self, factor, x):
        """ This method marginalizes out a factor 
        by summing over all possible value combinations of x
        factor :  BN factor
        x : str variable to sum-out
        """
        #print(self.bn.get_all_variables())
        #factor = self.bn.get_cpt(nfactor)
        
        #if x not in list(factor.columns):
        #    return factor
        
        cpt_ext_f = factor.drop([x, "p"], axis=1)
        marginalized_cpt = cpt_ext_f.drop_duplicates().reset_index(drop= True)
    
        n_assignments = len(marginalized_cpt)
        n_rows_cpt = len(factor)
        
        #print(marginalized_cpt)
        # get values of the first row
        cps =[]
        
        # sum conditional probability for same assignments
        for i in range(n_assignments):
            ass = [val for val in marginalized_cpt.iloc[i]]
            #print('Assignment', ass)
            
            p = 0
            for j in range(n_rows_cpt):
                row_to_check = [val for val in cpt_ext_f.iloc[j]]
                #print('cpt row', row_to_check)
                
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
                
                print('An edge was added between {}'.format(pair))
    
                #print('BEFORE', self.graph.edges)
                # add non-existent edge
                self.graph.add_edge(pair[0], pair[1])
                #Sprint('AFTER', self.graph.edges)
                
        # delete node from interaction graph
        self.graph.remove_node(x)

        return marginalized_cpt
    
    def factor_multiplication(self, factor_f, factor_g):
        
        '''Params
        factor_f : cpt factor f
        factor_g : cpt factor g
        '''
        #factor_f_cpt = self.factors[factor_f]
        #factor_g_cpt = self.factors[factor_g]
        
        factor_f.columns = factor_f.columns.map(''.join)
        factor_g.columns = factor_g.columns.map(''.join)
        
        print(factor_f)
        print(factor_g)
        
        variables_factor_f = list(factor_f.columns.values)[:-1]
        variables_factor_g = list(factor_g.columns.values)[:-1]
       
        # print('factors f', variables_factor_f)
        # print('factors g', variables_factor_g)

        all_variables = np.unique(variables_factor_f + variables_factor_g)
        ## print(all_variables)
                
        table = list(itertools.product([False, True], repeat=len(all_variables))) 
        table_list = [list(ele) for ele in table]
        
        ## print(table_list)
        multiplication_factor = pd.DataFrame(table_list, columns = [all_variables])
        
        ## print(multiplication_factor)
        products = []
        
        for ass in table:
            assignment = dict(zip(all_variables, ass))
            
            ## print(assignment)
            instantiation = pd.Series(assignment)
            
            print('INSTANCIATION', instantiation)
            
            # get compatible instanciation table for factor 1
            factor_f_compat = self.bn.get_compatible_instantiations_table(instantiation, factor_f)
            factor_g_compat = self.bn.get_compatible_instantiations_table(instantiation, factor_g)
            
            print('factor compatible', factor_g_compat['p'].values)
            print('factor compatible', factor_f_compat['p'].values)
            
            if len(factor_g_compat['p'].values) == 0 or len(factor_f_compat['p'].values) == 0:
                product = 0
            
            else :
                product = factor_g_compat['p'].values[0] * factor_f_compat['p'].values[0]
            
            #print('product', product) 
            products.append(product)
               
        multiplication_factor['p'] = products
        
        # add an edge in the interaction graph between neighbors if non-existent         
        #self.graph.add_edge(factor_f, factor_g)
        
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
        maxed_out_cpt = cpt_ext_f.drop_duplicates().reset_index(drop= True)
    
        n_assignments = len(maxed_out_cpt)
        n_rows_cpt = len(factor)
        
        print(maxed_out_cpt)
        # get values of the first row
        cps =[]
        X_instantiations = []
        
        # find max conditional probability for same assignments
        for i in range(n_assignments):
            ass = [val for val in maxed_out_cpt.iloc[i]]
            #print('Assignment', ass)
           
            p_max = -1
            
            for j in range(n_rows_cpt):
                row_to_check = [val for val in cpt_ext_f.iloc[j]]
                #print('cpt row', row_to_check)
                                
                if row_to_check == ass:
                    if p_max < factor.iloc[j]['p']:
                        p_max = factor.iloc[j]['p']
                        ins_x = factor.iloc[j][x]
             
            X_instantiations.append(ins_x)    
            cps.append(p_max)
            ind_max = np.argmax(cps)
            
            instantiation_with_map = X_instantiations[ind_max]
            
        # Add to dataframe
        maxed_out_cpt['p'] = cps
        maxed_out_cpt['instantiation_x'] = X_instantiations
        
        ## print(marginalized_cpt)
        return maxed_out_cpt, instantiation_with_map
    
    def ordering_MinDegreeHeuristic(self, X):
        '''Minimimum Degree Heuristic
        Gets the number of edge (degree) for each node
        
        Args
        X : list of variables
        returns: str variable with minimum degree, to eliminate first
        
        '''
        # get degree for every variable
        degree = dict(self.graph.degree)
        print(degree)
        
        # build new df with only keys in interaction graphS
        x_degree = {node: degree[node] for node in X}
        print(x_degree)
        
        # order dict 
        ordered_x_degree = sorted(x_degree.items(), key=lambda item: item[1])
                
        # get full order of variables
        full_order = [var[0] for var in ordered_x_degree]
        
        # get variable with min degree
        min_degree = min(x_degree, key=x_degree.get) # should this be able to deal with ties?
        
        return min_degree, full_order
        
        
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
        print(ordered_x_dict)
        
        # get full order of variables as list
        full_order = [var[0] for var in ordered_x_dict]
        
        # get minimum fill
        min_fill = min(x_dict, key=x_dict.get) # should this be able to deal with ties?
        
        return min_fill, full_order
        
    def one_variable_elimination(self, x_target):
        '''Sum out one target variable with variable elimination
        Args 
        x_target : string of target variable name
        
        returns : Factors after the target variable has been eliminated.
        '''
        ## print('CPT BEFORE MARGINALIZATION', self.bn.get_all_cpts())
        
        all_variables = list(self.factors.keys())
        print(all_variables)
        
        # get all factors that contain the target variable
        variables_product = []
        for variable in all_variables:
            # get factor
            factor_var = self.factors[variable]
            variables_in_factor = list(factor_var.columns.map(''.join))
            
            print('variable in factor', variables_in_factor)
            
            if x_target in variables_in_factor:
                variables_product.append(variable)
                            
        # print('FACTORS TO BE CONSIDERED', variables_product)       
        
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
        # print(summed_out_factor)
        
        # remove all factors containing this target variable from cpt
        #print('FACTOR BEFORE MARGINAL', self.factors)
        for variable in variables_product:
            print(variable)
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
            self.one_variable_elimination(target)
                    
            
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
        #print(all_variables)
        
        # reduce all factors with respect to e
        for variable in all_variables:
            ## print(variable)
            cpt_var = self.factors[variable]
            
            reduced_cpt = self.bn.get_compatible_instantiations_table(pd.Series(e), cpt_var)
            
            self.factors[variable] = reduced_cpt
        
        print('FACTORS REDUCED', self.factors)
                
        not_q_variables = list(set(all_variables).difference(set(Q)))
        print('Not query variables', not_q_variables)
        
        while len(not_q_variables) > 0:
            
            print('VARIABLES LEFT TO SUM-OUT (NOT Q)', not_q_variables)
            
            # pick variable ordering
            if heuristic == 'min_fill':
                min_var, _ = self.ordering_MinFillHeuristic(not_q_variables)
            
            elif heuristic == 'min_degree':
                min_var, _ = self.ordering_MinDegreeHeuristic(not_q_variables)
                
            print('Marginalizing w.r.:', min_var)
            
            ### compute joint marginal pr(Q & e) ###
            # find all factors that contain var
            # compute product of the factors
            # sum-out variable
            self.one_variable_elimination(min_var)
            
            # update evidence left to sum-out
            not_q_variables.remove(min_var)
            
        ## print('FACTORS AFTER ALL SUMMING OUT COMPLETED')
        ## print('N FACTORS' , len(self.factors))
        ## print('FACTORS', self.factors)
                          
        nfactors = list(self.factors.keys()) 
        ## print(nfactors)   
        product_factor = self.factors[nfactors[0]]
        
        # compute the product of resulting factors    
        if len(nfactors) > 1:    
            for name in (nfactors[1:]): 
                product_factor = self.factor_multiplication(product_factor, self.factors[name])    

        ## print('Product factor', product_factor)
        
        # get pr(Q) corresponding to truth assignment for Q
        # if assignment is true, sum 
        Q_true = dict(zip(Q, [True for i in range(len(Q))]))
        pr_e_Q = self.bn.get_compatible_instantiations_table(pd.Series(Q_true), product_factor)
              
        pr_e_Q = pr_e_Q.sum()['p']
        print('pr(Q & e)', pr_e_Q)
        
        # from previous, sum out Q to get p(e)
        pr_e = product_factor.sum()['p']
        print('pr(e)', pr_e)
        
        # compute pr(Q & e) / pr(e)
        pr_Q = pr_e_Q / pr_e
                
        return pr_Q, product_factor
    
    def MAP(self, Q, e, heuristic):
        
        _, product_factor = self.compute_marginal_distribution(Q, e, heuristic)
        maxed_out = product_factor       
        map_instantiation = {}
        
        for q in Q:
            maxed_out, instantiation_with_map = self.maxing_out(maxed_out, q)
            map_instantiation[q] = instantiation_with_map
            
            ##print(maxed_out)
            ##print(map_instantiation)
        
        return maxed_out['p'], map_instantiation
        
    def MEP(self, e, heuristic):
        '''Get instantiation for all variables given the evidence'''
        
        # network pruning
    
        all_variables = list(self.bn.get_all_variables())
        
        # get all variables not e
        not_e_variables = list(set(all_variables).difference(set(e)))
        
        print('Not evidence variables', not_e_variables)
        
        while len(not_e_variables) > 0:
            
            print('VARIABLES LEFT TO MAX OUT (NOT E)', not_e_variables)
            
            # pick variable ordering
            if heuristic == 'min_fill':
                min_var, _ = self.ordering_MinFillHeuristic(not_e_variables)
            
            elif heuristic == 'min_degree':
                min_var, _ = self.ordering_MinDegreeHeuristic(not_e_variables)
                
            print('Maxing out w.r.:', min_var)
            
            # find all factors that contain variable
            # compute product of the factors
            # max-out variable
            variables_product = []
            
            for variable in all_variables:
                # get factor
                factor_var = self.factors[variable]
                
                print(factor_var)
                variables_in_factor = list(factor_var.columns.map(''.join))
            
                print('variable in factor', variables_in_factor)
            
                if min_var in variables_in_factor:
                    variables_product.append(variable)
            
            # compute the product of all factors containing the target variable        
            if len(variables_product) > 1:
                factor_f = self.factors[variables_product[0]]
                factor_g = self.factors[variables_product[1]]
                product_factor = self.factor_multiplication(factor_f, factor_g)
            
            else:
                if variables_product == []:
                    print('Nothing to reduce with respect to target variable')
                    return self.factors
            
                else : # only one factor to consider
                    product_factor = self.factors[variables_product[0]]
        
            if len(variables_product) > 2:
                for i in range(2, len(variables_product)):
                    new_factor = self.factors[variables_product[i]]
                    product_factor = self.factor_multiplication(product_factor, new_factor)    
            
            maxed_out_factor = self.maxing_out(product_factor, min_var)
            
            for variable in variables_product:
                del self.factors[variable]

            # add the maxed-out factor to the list of factors
            self.factors[min_var] = maxed_out_factor
            
            # update evidence left to sum-out
            not_e_variables.remove(min_var)
        
        print(self.factor)

###########################################################################
############################### TESTING ###################################
###########################################################################


############ Instantiate ############
bayesian = BNReasoner("dog.BIFXML") # heuristic = ordering_MinFillHeuristic(x)

#print(bayesian.bn.structure.predecessors('dog-out'))
#print(bayesian.bn.get_all_cpts())
#print(bayesian.bn.del_var('hear-bark'))
#print(bayesian.bn.get_all_cpts())


#print(bayesian.bn.get_all_variables())

#################### Network pruning #########################

#################### test d-separation #######################
#print(bayesian.check_one_dseparation('light-on', 'dog-bark', ['family-out', 'hear-bark']))
#print(bayesian.check_set_dseparation(['light-on'], ['dog-bark'], ['family-out', 'hear-bark']))

#################### test independence #######################
#print(bayesian.check_independance('light-on', 'dog-bark', ['family-out', 'hear-bark']))
#print(bayesian.bn.structure)
#print(bayesian.apply_pruning('family-out', 'dog-bark', ['light-on', 'bowel-problem']))
#print(bayesian.bn.structure)

################### test marginalization of factors ###################
cpt_factor1 = bayesian.bn.get_all_cpts()['hear-bark'] 
#cpt_factor2 = bayesian.bn.get_all_cpts()['bowel-problem']
#cpt_factor3 = bayesian.bn.get_all_cpts()['dog-out']

#print(bayesian.marginalization_factor(cpt_factor1, 'dog-out'))
#print('BEFORE', bayesian.bn.structure.edges
#bayesian.bn.add_edge(('light-on', 'bowel-problem'))
#print('AFTER', bayesian.bn.structure.edges)

################### test factor multiplication ###################
#print(bayesian.factor_multiplication('light-on', 'bowel-problem'))

################### test maxing-out ###################
#print(bayesian.maxing_out(cpt_factor3, 'dog-out'))

################## Test min heuristics ###################
#print('MINIMUM DEGREE HEURISTIC')
#print(bayesian.ordering_MinDegreeHeuristic(['hear-bark', 'light-on','family-out']))

#print('MINIMUM FILL HEURISTIC')
#print(bayesian.ordering_MinFillHeuristic(['hear-bark', 'light-on','family-out']))
#g = bayesian.bn.get_interaction_graph()
#print(treewidth_min_fill_in(g))

#print(bayesian.compute_marginal_distribution(['hear-bark', 'light-on'], {'family-out': False}, 'min_fill'))
#print(bayesian.MAP(['hear-bark', 'light-on'], {'family-out': False}, 'min_fill'))

#print('CPT COPY', bayesian.cpt)
################## Test variable elimination #####################
#print(bayesian.one_variable_elimination('light-on'))
#print(bayesian.one_variable_elimination('dog-out'))
#print(bayesian.set_variable_elimination(['dog-out', 'hear-bark', 'light-on']))
#print(bayesian.bn.get_interaction_graph().adj)
#bayesian.find_observed_ancestors(['dog-out', 'family-out'])

#print(bayesian.bn.get_compatible_instantiations_table(pd.Series({'fam-out': False}),bayesian.bn.get_all_cpts()['fam-out']))
#print(bayesian.bn.get_all_cpts())

print(bayesian.MEP({'light-on': False}, 'min_fill'))


 # get min variable neighbors
'''neighbors_min_degree = [n for n in graph.neighbors(min_degree)]
        combination_neighbors_min_degree = list(set(combinations(neighbors_min_degree, 2)))
        
        for i, pair in enumerate(combination_neighbors_min_degree):
            
            # if not connected, add edge
            if (combination_neighbors_min_fill != []) and ((combination_neighbors_min_fill[i][0], combination_neighbors_min_fill[i][1]) 
            and (combination_neighbors_min_fill[i][1], combination_neighbors_min_fill[i][0]) 
            not in self.bn.get_interaction_graph().edges):
                
                print('EDGE WAS ADDED TO GRAPH')
                
                # add non-existent edge
                self.bn.add_edge(pair)
        
        # remove node from graph (sum-out)
        self.bn.del_var(min_degree)'''