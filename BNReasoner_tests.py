import BNReasoner
from BNReasoner import BNReasoner

###########################################################################
############################### TESTING ###################################
###########################################################################


############ Instantiate ############
bayesian = BNReasoner("lecture_example.BIFXML") 

#################### Network pruning #########################
print('-------------------- Testing - Network pruning -------------------- \n\n')
print('Q : Wet Grass?, evidence -> Winter? : True, Rain? : False')
print(bayesian.apply_pruning(['Wet Grass?'], {'Winter?': True, 'Rain?' : False}))
print('\n\n')

#################### D-separation #######################
bayesian = BNReasoner("lecture_example.BIFXML")
print('Testing - D-separation - \n\n')
print(bayesian.check_set_dseparation(['Wet Grass?'], ['Winter?'], ['Rain?', 'Sprinkler?']))
print('\n\n')

#################### Test independence ####################
print('-------------------- Testing - Independence -------------------- \n\n')
print(bayesian.check_independence(['Wet Grass?'], ['Winter?'], ['Rain?', 'Sprinkler?']))
print('\n\n')

################### Marginalization ###################
to_sum_out = 'Rain?'
factor = bayesian.bn.get_all_cpts()[to_sum_out] 
print('-------------------- Testing - Summing-out -------------------- \n\n')
print('Factor prior to summing-out - \n')
print(factor)
print('\n')
print('Factor after summing-out w.r.t. {}- \n'.format(to_sum_out))
print(bayesian.marginalization_factor(factor, to_sum_out))
print('\n\n')

################### Factor multiplication ###################
print('-------------------- Testing - Summing-out -------------------- \n\n')
factor_f ='Wet Grass?'
factor_g = 'Rain?'
cpt_factor_f = bayesian.bn.get_all_cpts()[factor_f]
cpt_factor_g = bayesian.bn.get_all_cpts()[factor_g]
print('Factors considered for multiplication - \n')
print(cpt_factor_f)
print('\n')
print(cpt_factor_g)
print('\n')
print('Product resulting factor multiplication - \n')
print(bayesian.factor_multiplication(cpt_factor_f, cpt_factor_g))
print('\n\n')

################### Maxing-out ###################
print('-------------------- Testing - Maxing-out -------------------- \n\n')
to_max_out = 'Wet Grass?'
factor = bayesian.bn.get_all_cpts()[to_max_out] 
print('Factor prior to maxing-out - \n')
print(factor)
print('\n')
print('Factor after maxing-out w.r.t. {}- \n'.format(to_sum_out))
print(bayesian.maxing_out(factor, to_max_out))
print('\n \n')

################## Heuristics - Min Degree, Min Fill ###################
bayesian = BNReasoner("lecture_example.BIFXML")
print('-------------------- Testing - Min Degree Heuristic -------------------- \n\n')
min_variable, ordering, dict_count =  bayesian.ordering_MinDegreeHeuristic(['Wet Grass?', 'Rain?', 'Winter?'])
print('Variable with minimum degree based on current interaction graph \n')
print(min_variable)
print(dict_count)
print('\n')
print('Ordering of provided set of X based on current interaction graph \n')
print(ordering)
print('\n \n')

print('-------------------- Testing - Min Fill Heuristic -------------------- \n\n')
min_variable, ordering, dict_count =  bayesian.ordering_MinFillHeuristic(['Wet Grass?', 'Rain?', 'Winter?'])
print('Variable with minimum fill based on current interaction graph \n')
print(dict_count)
print(min_variable)
print('\n')
print('Ordering of provided set of X based on current interaction graph \n')
print(ordering)
print('\n \n')

################## Variable elimination #####################
var_to_eliminate = 'Rain?'
print('-------------------- Testing - Variable elimination w.r.t. x -------------------- \n\n')
print('Cpt tables prior to variable elimination \n')
print(bayesian.bn.get_all_cpts())
print('\n')
print(f'New factor from variable elimination w.r.t. to {var_to_eliminate}\n')
print(bayesian.one_variable_elimination(var_to_eliminate))
print('\n \n')

vars_to_eliminate = ['Winter?', 'Wet Grass?']
print('-------------------- Testing - Variable elimination with set X -------------------- \n\n')
print('Cpt tables prior to variable elimination \n')
print(bayesian.bn.get_all_cpts())
print('\n')
print(f'New factor from variable elimination w.r.t. to {vars_to_eliminate}\n')
print(bayesian.set_variable_elimination(vars_to_eliminate))
print('\n \n')


################## Marginal distribution #####################
bayesian = BNReasoner("lecture_example.BIFXML")
print('-------------------- Testing - Marginal distribution w.r.t. Q, e -------------------- \n\n')
pr_Q, product_factor = bayesian.compute_marginal_distribution(['Wet Grass?'], {'Winter?': True, 'Rain?' : False}, 'min_fill')
print('\n \n')

# given set query X
bayesian = BNReasoner("lecture_example.BIFXML")
print('-------------------- Testing - Marginal distribution w.r.t. Q, e given set Q -------------------- \n\n')
pr_Q, product_factor = bayesian.compute_marginal_distribution(['Slippery Road?', 'Sprinkler?'], {'Winter?': True, 'Rain?' : False}, 'min_fill')
print('\n \n')

# empty evidence 
print('-------------------- Testing - Marginal distribution w.r.t. Q, e given empty evidence -------------------- \n\n')
bayesian = BNReasoner("lecture_example.BIFXML")
pr_Q, product_factor = bayesian.compute_marginal_distribution(['Wet Grass?'], {}, 'min_fill')
print('\n \n')

################## MAP #####################
bayesian = BNReasoner("lecture_example2.BIFXML")
print('-------------------- Testing - Maximum a-posteriory instantiation w.r.t. Q, e  -------------------- \n\n')
print(bayesian.MAP(['I', 'J'], {'O': True}, 'min_fill'))
print('\n \n')


################## MEP #####################
bayesian = BNReasoner("lecture_example2.BIFXML")
print('-------------------- Testing - Most probable explanation w.r.t. e  -------------------- \n\n')
print(bayesian.MEP({'O': False, 'J': True}, 'min_fill'))
print('\n \n')