from typing import Union
from BayesNet import BayesNet
import copy
import pandas as pd
import itertools
import numpy as np

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
        factor : cpt of the BN factor
        x : str variable to sum-out
        """
        probs = factor["p"]
        cpt_ext_f = factor.drop([x, "p"], axis=1)
        marginalized_cpt = cpt_ext_f.drop_duplicates().reset_index(drop= True)
    
        n_assignments = len(marginalized_cpt)
        n_rows_cpt = len(factor)
        
        print(marginalized_cpt)
        # get values of the first row
        cps =[]
        
        # sum conditional probability for same assignments
        for i in range(n_assignments):
            ass = [val for val in marginalized_cpt.iloc[i]]
            print('Assignment', ass)
            
            p = 0
            for j in range(n_rows_cpt):
                row_to_check = [val for val in cpt_ext_f.iloc[j]]
                print('cpt row', row_to_check)
                
                if row_to_check == ass:
                    p += factor.iloc[j]['p']
                    
            cps.append(p)
        # Add to dataframe
        marginalized_cpt['p'] = cps
        ## print(marginalized_cpt)
        return marginalized_cpt
    
    def factor_multiplication(self, factor_f, factor_g):
        
        ## print(factor_f)
        ## print(factor_g)
        
        variables_factor_f = factor_f.columns.tolist()[:-1]
        variables_factor_g = factor_g.columns.tolist()[:-1]

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
            
            # get compatible instanciation table from factor 1
            factor_f_compat = self.bn.get_compatible_instantiations_table(instantiation, factor_f)
            factor_g_compat = self.bn.get_compatible_instantiations_table(instantiation, factor_g)
            
            ## print(factor_g_compat['p'].values)
            ## print(factor_f_compat['p'].values)
            
            product = factor_g_compat['p'].values[0] * factor_f_compat['p'].values[0]
            
            #print('product', product) 
            products.append(product)
               
        multiplication_factor['p'] = products
        return multiplication_factor
            
    def maxing_out(self, factor, x):
        # has to keep track of which instantiation of X led to the maximized value
        """ 
        Args
        factor : cpt of the BN factor
        x : str variable to max-out
        """
        probs = factor["p"]
        cpt_ext_f = factor.drop([x, "p"], axis=1)
        summed_out_cpt = cpt_ext_f.drop_duplicates().reset_index(drop= True)
    
        n_assignments = len(summed_out_cpt)
        n_rows_cpt = len(cpt)
        
        print(summed_out_cpt)
        # get values of the first row
        cps =[]
        X_instantiations = []
        
        # find max conditional probability for same assignments
        for i in range(n_assignments):
            ass = [val for val in summed_out_cpt.iloc[i]]
            print('Assignment', ass)
           
            p_max = -1
            
            for j in range(n_rows_cpt):
                row_to_check = [val for val in cpt_ext_f.iloc[j]]
                print('cpt row', row_to_check)
                                
                if row_to_check == ass:
                    if p_max < cpt.iloc[j]['p']:
                        p_max = cpt.iloc[j]['p']
                        ins_x = cpt.iloc[j][X]
          
            X_instantiations.append(ins_x)    
            cps.append(p_max)
            
        # Add to dataframe
        summed_out_cpt['p'] = cps
        summed_out_cpt['instantiation_x'] = X_instantiations
        
        ## print(marginalized_cpt)
        return summed_out_cpt
    
    def ordering_MinDegreeHeuristic(self, X):
        '''Minimimum Degree Heuristic
        Gets the number of edge (degree) for each node
        
        Args
        X : list of variables
        returns: str variable with minimum degree, to eliminate
        
        '''
        # get degree for every variable
        degree = dict(self.bn.get_interaction_graph().degree)
        
        # build new df with only keys in set X
        x_degree = {node: degree[node] for node in X}
        
        # order dict 
        ordered_x_degree = sorted(x_degree.items(), key=lambda item: item[1])
                
        # get full order of variables
        full_order = [var[0] for var in ordered_x_degree]
        
        # get variable with min degree
        min_degree = min(x_degree, key=x_degree.get) # should this be able to deal with ties?
        
        # remove from graph
        
        # add an edge between neighbors
        return min_degree, full_order
        
        
    def ordering_MinFillHeuristic(self, X):
        '''Minimimum Fill Heuristic
        For each node of set X, finds neighbors, and add to the count 
        if neighbors not already connected (no edges between neighbors)
        
        Args
        X : list of variables
        return : str variable with minimum count of neighbors not connected, to eliminate
        
        '''
        dict = {}
        graph = self.bn.get_interaction_graph()
        ## print(self.bn.get_interaction_graph().edges)
    
        for var in self.bn.get_interaction_graph().nodes:
            dict[var] = 0
            
            # compute n edges between neighbors of X that are not connected already
            neighbors_x = [n for n in graph.neighbors(var)]
            ## print(f'Neighbors of {var}', neighbors_x)
            
            # if not edges, add to count
            for neighbor in neighbors_x :
                ## print('Edge required between', (neighbor, var))
                
                if (var, neighbor) and (neighbor, var) not in self.bn.get_interaction_graph().edges:
                    ## print('Edge not found', (neighbor, var))
                    dict[var] +=1 
                    
                # else:
                    # print('Edge already exists', (neighbor, var))         
            
        ## print(dict)  
        # build new df with only keys in set
        x_dict = {node: dict[node] for node in X}
        
        # order 
        ordered_x_dict = sorted(x_dict.items(), key=lambda item: item[1])
        
        # get full order of variables
        full_order = [var[0] for var in ordered_x_dict]
        
        # get minimum fill
        min_fill = min(dict, key=dict.get) # should this be able to deal with ties?
    
        # add interactions
        
        # remove node
        
        return min_fill, full_order
        
    def one_variable_elimination(self, x_target):
        '''Sum out one target variable with variable elimination
        Args 
        X : list of variables where index = order for elimination -> 0, first to be eliminated
        
        returns : List of factors after the target variable has been eliminated.
        '''
        # get random variable order
        # get variable order according to heuristic
        # first, full_order = self.ordering_MinFillHeuristic(X)
        
        #all_factors = self.bn.get_all_cpts()
        
        all_variables = self.bn.get_all_variables()
        
        # get all factors that contain the target variable
        variables_product = []
        
        for variable in all_variables:
            # get factor
            factor_var = self.bn.get_cpt(variable)
            variables_in_factor = list(factor_var.columns)
            
            if x_target in variables_in_factor:
                variables_product.append(variable)
                            
                
        # get the product of all factors containing the target variable        
        if len(variables_product) > 1:
            factor_f = self.bn.get_cpt(variables_product[0])
            factor_g = self.bn.get_cpt(variables_product[1])
            
            product_factor = self.factor_multiplication(factor_f, factor_g)
            
        else:
            return 
            
        for i in range(2, len(variables_product)):
            new_factor = self.bn.get_cpt(variables_product[i])
            product_factor = self.factor_multiplication(product_factor, new_factor)
        
        # sum out the target variable from the product factor
        summed_out_factor = self.marginalization_factor(product_factor, x_target)
        
        # remove all factors containing this target variable from list
        for variable in variables_product:
            self.bn.update_cpt(variable, None)
        
        
        # add the reduced factor to the list of factors
        self.bn.update_cpt(x_target, summed_out_factor)
    
        return self.bn.get_all_cpts()
    
    
        # remove variables outside of set from list ordering
        
        # for each variable in elimination ordering
        
            # find factors/fonctions with variable
            
            # multiply those functions
            
        #
        
        # regroup factors 
        
        # compute joint prob 
        #for factor in factors :
            
        #self.factor_multiplication(factor_f, factor_g)
        
        # which we eliminate first
        #var_eliminate = self.heuristic() 
        #if var_eliminate == None:
        #    break

        #self.marginalization_factor(var_eliminate)
            
        # factor multiplication over factors
        #if len(self.list_of_factors) > 1:
        #    self.factor_multiplication(factor_f, factor_g)
            
    

# reduce factors
# 

###########################################################################
############################### TESTING ###################################
###########################################################################


############ Instantiate ############
bayesian = BNReasoner("dog.BIFXML")

#print(bayesian.bn.structure.predecessors('dog-out'))
#print(bayesian.bn.get_all_cpts()['light-on'])
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
cpt_factor1 = bayesian.bn.get_all_cpts()['light-on'] 
cpt_factor2 = bayesian.bn.get_all_cpts()['bowel-problem']
cpt_factor3 = bayesian.bn.get_all_cpts()['dog-out']

#print(bayesian.marginalization_factor(cpt_factor3, 'bowel-problem'))

################### test factor multiplication ###################
#print(bayesian.factor_multiplication(cpt_factor1, cpt_factor2))

################### test maxing-out ###################
#print(bayesian.maxing_out(cpt_factor3, 'dog-out'))

################## Test min heuristics ###################
#print(bayesian.ordering_MinDegreeHeuristic(['light-on', 'bowel-problem', 'dog-out']))
# print(bayesian.ordering_MinFillHeuristic(['light-on', 'bowel-problem', 'dog-out']))

################## Test variable elimination #####################
print(bayesian.one_variable_elimination('dog-out'))

#print(bayesian.bn.get_interaction_graph().adj)
#bayesian.find_observed_ancestors(['dog-out', 'family-out'])
