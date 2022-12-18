import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statistics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import data
# hypothesis 1
methodVE_10 = [0.53125, 5.3125, 2.40625, 9.421875, 4.53125, 6.421875, 7.984375, 2.265625, 2.796875, 1.40625, 0.765625, 2.25, 1.03125, 3.5, 1.359375, 2.3125, 1.359375, 0.53125, 2.828125, 1.203125, 1.515625, 2.609375, 2.96875, 2.953125, 0.5, 2.59375, 1.203125, 1.90625, 2.1875, 10.34375, 1.4375, 4.140625, 0.78125, 1.53125, 2.015625, 24.453125, 2.71875, 0.765625, 15.6875, 1.265625, 4.125, 0.546875, 1.640625, 1.015625, 1.5625, 3.484375, 3.1875, 6.46875, 1.25, 3.046875, 2.84375, 1.03125, 1.109375, 0.6875, 0.890625, 7.546875, 3.828125, 6.296875, 0.96875, 2.1875, 0.640625, 2.953125, 0.625, 0.609375, 2.859375, 1.890625, 3, 2.15625, 25.703125, 0.5625, 10.84375, 1.078125, 4.203125, 0.40625, 0.625, 1.234375, 1.28125, 3.484375, 3.546875, 0.890625, 3.3125, 1.078125, 1.875, 1, 1.0625, 0.6875, 0.921875, 1.90625, 1.359375, 1.25, 0.65625, 1.453125, 1.328125, 1.5, 1.109375, 3.21875, 3.421875, 4, 1.765625, 5.046875]
methodNa_10 = [27.578125, 29.703125, 30.984375, 34.140625, 30.6875, 32.984375, 30.53125, 30.453125, 30.4375, 28.609375, 28.953125, 31.09375, 29.375, 28.796875, 30.0, 28.65625, 29.3125, 28.71875, 34.5625, 28.796875, 28.3125, 28.4375, 30.453125, 29.421875, 27.984375, 28.703125, 29.921875, 32.171875, 28.4375, 33.3125, 29.671875, 30.640625, 28.1875, 30.828125, 28.3125, 36.25, 32.71875, 28.171875, 32.96875, 30.21875, 33.171875, 28.046875, 27.90625, 27.78125, 30.375, 28.390625, 30.9375, 31.9375, 29.21875, 30.3125, 29.421875, 28.28125, 29.125, 28.515625, 28.140625, 32.421875, 30.21875, 31.5, 29.875, 29.609375, 28.65625, 30.015625, 29.921875, 28.5, 30.5625, 28.84375, 29.671875, 33.8125, 31.796875, 28.4375, 33.171875, 29.40625, 29.515625, 27.46875, 29.796875, 28.515625, 28.265625, 30.75, 37.578125, 28.359375, 29.359375, 28.4375, 30.46875, 29.578125, 28.453125, 30.78125, 29.46875, 34.203125, 31.46875, 31.828125, 31.296875, 33.0, 31.65625, 32.03125, 30.859375, 32.0, 31.640625, 31.796875, 31.46875, 33.859375]
methodVE_5 = [0.16588758, 0.23275358, 0.16805708, 0.17087234, 0.16931476, 0.12889411, 0.10464868, 0.19689434, 0.13224005, 0.1887007, 0.24402137, 0.09728658, 0.25384317, 0.13657, 0.20464666, 0.12049134, 0.17912391, 0.33824963, 0.09521976, 0.1364536, 0.19939653, 0.12550455, 0.16082199, 0.08823199, 0.22801545, 0.11351404, 0.184558, 0.11414248, 0.11036884, 0.11562634, 0.18089552, 0.12727563, 0.19334769, 0.15243503, 0.35248708, 0.12908963, 0.11492413, 0.11030615, 0.1065375, 0.13036695, 0.17585047, 0.22378563, 0.20963202, 0.13513463, 0.18170796, 0.1945297, 0.13504754, 0.18498409, 0.17080037, 0.16739002, 0.12813819, 0.0986626, 0.09212918, 0.20814298, 0.19695142, 0.13878151, 0.25596685, 0.09875508, 0.14138399, 0.13811078, 0.14822157, 0.1855357, 0.17570724, 0.11080237, 0.13276261, 0.1542449, 0.16418904, 0.18495383, 0.18822465, 0.16966614, 0.15474235, 0.10598139, 0.0884557, 0.16759631, 0.08327834, 0.10467056, 0.18684362, 0.13896397, 0.18275821, 0.32226359, 0.0954379, 0.29331808, 0.12474517, 0.097134, 0.23302539, 0.34496593, 0.18128912, 0.12240228, 0.08985711, 0.09819967, 0.11872488, 0.10151503, 0.14643197, 0.23328256, 0.09111834, 0.32708417, 0.14235935, 0.10395476, 0.12313283, 0.14127584]
methodNa_5 = [0.36863617, 0.37836116, 0.38023306, 0.3166863, 0.3977999, 0.31182054, 0.31352879, 0.36418433, 0.32420846, 0.33072024, 0.37679747, 0.31070376, 0.32960477, 0.27912421, 0.29715118, 0.31615453, 0.30773728, 0.3727683, 0.33253715, 0.32468662, 0.35206795, 0.35361024, 0.29283154, 0.30998477, 0.39726773, 0.29990997, 0.31748649, 0.33622516, 0.30453259, 0.34549957, 0.37984483, 0.32976455, 0.31795349, 0.34932702, 0.3749342, 0.33178567, 0.32179498, 0.31098278, 0.33752454, 0.33038238, 0.30648162, 0.32347123, 0.37346658, 0.32997178, 0.34208225, 0.39716827, 0.31863793, 0.3216098, 0.34723841, 0.3344953, 0.31881073, 0.31925732, 0.29543478, 0.31182921, 0.36018695, 0.30846278, 0.37607098, 0.33788583, 0.31312318, 0.30574798, 0.29763281, 0.28902261, 0.31871902, 0.32217274, 0.32275312, 0.3247101, 0.35606766, 0.3043943, 0.36041053, 0.3230462, 0.30507325, 0.31483924, 0.34331697, 0.34716741, 0.30957676, 0.32986363, 0.31486852, 0.33630139, 0.35362919, 0.34362108, 0.32533172, 0.44082347, 0.3241908, 0.3533406, 0.3456727, 0.38922142, 0.35035272, 0.30763113, 0.29520625, 0.31140761, 0.32232613, 0.32271587, 0.35443567, 0.33932275, 0.32215155, 0.38040802, 0.3197855, 0.28498469, 0.28895119, 0.30158704]
methodVE_11 = [119.4905376, 19.51770698, 9.40195761, 9.36914277, 5.42422872, 5.4836775, 29.88348445, 16.17747026, 27.64971002, 12.43314718, 2.91485454, 7.19103891, 4.85974043, 26.42102278, 14.84561824, 5.09271722, 12.58871702, 13.29487343, 8.30793645, 16.81439691, 5.52951567, 69.34030603, 18.84688016, 4.69193183, 4.39225455, 6.124255, 16.61451795, 8.11295147, 62.52357979, 3.51304119, 5.96885964, 3.93047373, 3.00776649, 28.23680413, 4.91249049, 33.84694691, 1.37546819, 3.50293134, 46.90058162, 1.41082837, 15.49118125, 30.03565056, 41.6285916, 6.64146851, 3.72042964, 7.47345, 4.20059242, 25.44033882, 3.0555914, 24.23252282, 1.7248301, 15.7850084, 8.79233806, 6.39757808, 52.5786283, 10.02704488, 17.98466232, 2.85476832, 6.15045911, 3.02871042, 6.94264097, 7.64368872, 15.91487311, 4.75610266, 4.6182125, 79.51034972, 31.7943002, 26.08310282, 21.37774355, 1.10802705, 6.08802159, 10.23987196, 15.83579675, 210.4330069, 10.48522198, 1.65903314, 1.71338667, 20.27506261, 6.72380011, 9.83281623, 13.03803395, 2.02234539, 7.18673007, 31.83310848, 9.17967588, 12.41296784, 8.32509328, 2.47149456, 14.19980602, 2.97317679, 29.65542554, 2.37902467, 1.45837807, 4.38537023, 10.89946917, 3.51860579, 1.8432597, 11.61930118, 7.46902496, 1.59668073]
methodNa_11 = [776.234375, 277.359375, 177.09375, 187.40625, 175.71875, 183.15625, 320.15625, 518.671875, 398.578125, 512.5, 127.4375, 137.921875, 377.4375, 315.625, 377.953125, 808.265625, 791.0625, 264.375, 103.15625, 357.546875, 382.390625, 525.828125, 647.6875, 144.25, 115.578125, 112.375, 259.078125, 111.140625, 200.984375, 443.59375, 114.09375, 146.671875, 101.46875, 267.609375, 128.8491745, 332.9153744, 126.3469269, 390.8576898, 221.4639004, 225.208578, 310.7422401, 339.3322156, 199.0717996, 114.373719, 231.6340093, 323.4487, 338.0219229, 314.3493161, 302.5088893, 154.8079157, 662.6906758, 145.4580407, 308.2599709, 426.3564428, 466.4433885, 328.5313231, 158.8575585, 454.2141814, 425.0581447, 128.70065, 119.8102905, 128.4502843, 292.8933308, 155.424247, 228.5345395, 354.0025585, 182.2522991, 234.3612608, 119.6110171, 128.4502843, 292.8933308, 155.424247, 228.5345395, 354.0025585, 182.2522991, 234.3612608, 119.6110171, 339.0358963, 879.7713186, 190.8686875, 761.6889988, 128.8156364, 160.9197377, 154.3454592, 1050.341395, 122.4907762, 461.3397347, 335.9361949, 360.6460315, 367.8452298, 375.0408491, 372.1520804, 341.3237331, 384.3739143, 320.6084545, 372.6156822, 308.2472363, 333.9182148, 320.6557746, 376.7170622]

# hypothesis 2
MFO_5nodes = [0.1875, 0.203125, 0.203125, 0.125, 0.21875, 0.09375, 0.078125, 0.109375, 0.078125, 0.140625, 0.109375, 0.078125, 0.15625, 0.09375, 0.109375, 0.09375, 0.078125, 0.1875, 0.078125, 0.09375, 0.140625, 0.078125, 0.15625, 0.0625, 0.171875, 0.0625, 0.109375, 0.125, 0.09375, 0.09375, 0.109375, 0.109375, 0.09375, 0.171875, 0.21875, 0.09375, 0.078125, 0.078125, 0.09375, 0.078125, 0.15625, 0.1875, 0.125, 0.109375, 0.125, 0.109375, 0.0625, 0.15625, 0.171875, 0.09375, 0.0625, 0.0625, 0.0625, 0.09375, 0.125, 0.09375, 0.171875, 0.09375, 0.0625, 0.078125, 0.078125, 0.109375, 0.109375, 0.0625, 0.078125, 0.078125, 0.140625, 0.140625, 0.109375, 0.09375, 0.078125, 0.109375, 0.046875, 0.109375, 0.0625, 0.0625, 0.09375, 0.109375, 0.171875, 0.1875, 0.09375, 0.203125, 0.109375, 0.09375, 0.078125, 0.234375, 0.140625, 0.09375, 0.078125, 0.078125, 0.09375, 0.078125, 0.078125, 0.109375, 0.09375, 0.25, 0.15625, 0.09375, 0.09375, 0.0625]
MFO_10nodes = [0.6875, 6.5, 3.25, 8.921875, 4.0625, 6.8125, 10.59375, 2.40625, 2.46875, 1.421875, 0.5625, 1.984375, 0.90625, 2.84375, 1.734375, 1.734375, 1.703125, 0.515625, 5.921875, 1.1875, 1.4375, 2.46875, 4.015625, 3.28125, 0.546875, 1.6875, 1.515625, 2.296875, 2.484375, 8.375, 1.375, 4.046875, 0.828125, 1.71875, 2.546875, 14.671875, 7.0625, 0.65625, 13.5625, 2.484375, 4, 0.4375, 1.90625, 1.09375, 2.203125, 3.390625, 3.25, 5.640625, 1.078125, 4.265625, 3.8125, 1.109375, 1.203125, 0.75, 0.875, 5.65625, 4.953125, 7.953125, 1.015625, 3.078125, 0.734375, 3.375, 1.03125, 0.625, 3.609375, 2.421875, 2.625, 5.234375, 18.1875, 0.6875, 6.78125, 0.921875, 2.625, 0.3125, 1.046875, 1.0625, 1.1875, 2.921875, 7.359375, 0.859375, 3.109375, 1.203125, 1.59375, 1.015625, 1.171875, 0.78125, 0.609375, 2, 1.703125, 1.71875, 0.609375, 2.078125, 1.328125, 2.34375, 1.265625, 3.90625, 3.578125, 3.296875, 2.078125, 5.28125]
MFO_15nodes = [158.515625, 233.109375, 264.59375, 52.140625, 141.796875, 140.8125, 1082.46875, 648.375, 657.375, 151.71875, 139.484375, 291.296875, 162.671875, 248.90625, 192.25]
MDO_5nodes = [0.140625, 0.171875, 0.125, 0.09375, 0.171875, 0.0625, 0.078125, 0.09375, 0.078125, 0.125, 0.09375, 0.0625, 0.09375, 0.078125, 0.109375, 0.078125, 0.078125, 0.109375, 0.09375, 0.109375, 0.140625, 0.09375, 0.15625, 0.0625, 0.15625, 0.0625, 0.125, 0.140625, 0.078125, 0.09375, 0.09375, 0.109375, 0.078125, 0.109375, 0.140625, 0.09375, 0.09375, 0.078125, 0.0625, 0.09375, 0.125, 0.140625, 0.109375, 0.109375, 0.125, 0.09375, 0.0625, 0.140625, 0.15625, 0.09375, 0.0625, 0.078125, 0.0625, 0.09375, 0.109375, 0.09375, 0.171875, 0.09375, 0.078125, 0.09375, 0.0625, 0.109375, 0.109375, 0.0625, 0.078125, 0.078125, 0.140625, 0.15625, 0.109375, 0.125, 0.09375, 0.09375, 0.046875, 0.09375, 0.0625, 0.0625, 0.125, 0.109375, 0.171875, 0.15625, 0.078125, 0.171875, 0.109375, 0.09375, 0.078125, 0.15625, 0.140625, 0.109375, 0.09375, 0.109375, 0.109375, 0.078125, 0.078125, 0.109375, 0.09375, 0.1875, 0.1875, 0.078125, 0.09375, 0.078125]
MDO_10nodes = [0.59375, 10.78125, 2.671875, 11.640625, 4.296875, 9.59375, 10.265625, 3.15625, 2.328125, 1.625, 0.65625, 2.171875, 0.9375, 2.734375, 1.546875, 1.078125, 1.28125, 0.59375, 9.234375, 1.015625, 2.046875, 2.453125, 4.140625, 3.734375, 0.453125, 2.40625, 1.6875, 2.703125, 2.640625, 10.65625, 1.21875, 2.546875, 0.6875, 1.578125, 1.9375, 15.40625, 8.84375, 0.625, 16.109375, 3.046875, 5.234375, 0.65625, 1.5625, 0.859375, 2.03125, 2.546875, 3.328125, 5.140625, 1.078125, 4.296875, 3.703125, 1.125, 1.265625, 0.578125, 0.859375, 5.109375, 3.4375, 8.421875, 1.203125, 2.203125, 0.765625, 3.015625, 1.09375, 0.703125, 4.546875, 2.515625, 2.515625, 3.90625, 14.875, 0.71875, 5.921875, 1.140625, 3.03125, 0.328125, 1.09375, 1.171875, 1.203125, 2.734375, 6.390625, 0.859375, 3.171875, 1.046875, 1.5, 1.046875, 1.1875, 0.78125, 0.609375, 1.65625, 1.40625, 1.375, 0.5625, 2.03125, 1.203125, 1.96875, 1.25, 3.15625, 2.53125, 3.40625, 1.890625, 4.96875]
MDO_15nodes = [164.5625, 391.09375, 433.390625, 53.0625, 241.515625, 210.375, 1154.234375, 427.84375, 445.171875, 139.0625, 107.640625, 442.5, 210.390625, 233.953125, 184.484375]

# Tests
print("---These should all be 100:---")
print(len(methodVE_5))
print(len(methodVE_10))
print(len(methodVE_11))
print(len(methodNa_5))
print(len(methodNa_10))
print(len(methodNa_11))

print("---The first 4 should be 100, the last 2 should be 15:---")
print(len(MFO_5nodes))
print(len(MFO_10nodes))
print(len(MFO_15nodes))
print(len(MDO_5nodes))
print(len(MDO_10nodes))
print(len(MDO_15nodes))
print("------------------------------------------ \n")
# run ANOVA
# hypothesis 1:
fvalue, pvalue = stats.f_oneway(methodVE_5, methodNa_5, methodVE_10, methodNa_10, methodVE_11, methodNa_11)

# hypothesis 2:
fvalue2, pvalue2 = stats.f_oneway(MFO_5nodes, MDO_5nodes, MFO_10nodes, MDO_10nodes, MFO_15nodes, MDO_15nodes)
#fvalue3, pvalue3 = stats.f_oneway(MFO_5nodes, MDO_5nodes, MFO_10nodes, MDO_10nodes)

# run Paired t-tests
# hypothesis 1:
# Paired t-tests
t_value0, p_value0 = stats.ttest_rel(methodVE_5, methodNa_5)
t_value01, p_value01 = stats.ttest_rel(methodVE_10, methodNa_10)
t_value02, p_value02 = stats.ttest_rel(methodVE_11, methodNa_11)

# hypothesis 2:
t_value, p_value = stats.ttest_rel(MFO_5nodes, MDO_5nodes)
t_value1, p_value1 = stats.ttest_rel(MFO_10nodes, MDO_10nodes)
t_value2, p_value2 = stats.ttest_rel(MFO_15nodes, MDO_15nodes)

# print results
print("Hypothesis 1: Variable Elimination vs Naive Summing-Out")
print('Two-Way ANOVA Results Hypothesis 1: F={}, p={}'.format(fvalue, pvalue))
if pvalue < 0.05:
    print('There is a significant difference between Variable Elimination and Naive Summing-Out \n')

print('Paired t-test for MFO_5nodes and MDO_5nodes: t={}, p={}'.format(t_value0, p_value0))
if p_value0 < 0.05:
    print('There is a significant difference between methodVE_5 and methodNa_5 \n')

print('Paired t-test for MFO_10nodes and MDO_10nodes: t={}, p={}'.format(t_value01, p_value01))
if p_value01 < 0.05:
    print('There is a significant difference between methodVE_10 and methodNa_10 \n')

print('Paired t-test for MFO_15nodes and MDO_15nodes: t={}, p={}'.format(t_value02, p_value02))
if p_value02 < 0.05:
    print('There is a significant difference between methodVE_11 and methodNa_11 \n')

print("----------------------------------- \n")
print("Hypothesis 2: MFO vs MDO")

print('Two-Way ANOVA Results Hypothesis 2: F={}, p={}'.format(fvalue2, pvalue2))
if pvalue2 < 0.05:
    print('There is a significant difference between MFO and MDO \n')

print('Paired t-test for MFO_5nodes and MDO_5nodes: t={}, p={}'.format(t_value, p_value))
if p_value < 0.05:
    print('There is a significant difference between MFO_5nodes and MDO_5nodes \n')

print('Paired t-test for MFO_10nodes and MDO_10nodes: t={}, p={}'.format(t_value1, p_value1))
if p_value1 < 0.05:
    print('There is a significant difference between MFO_5nodes and MDO_5nodes \n')

print('Paired t-test for MFO_15nodes and MDO_15nodes: t={}, p={}'.format(t_value2, p_value2))
if p_value2 < 0.05:
    print('There is a significant difference between MFO_5nodes and MDO_5nodes \n')

# Perform Tukey's Honestly Significant Difference (HSD) Test
#print("Hypothesis 1: \n")
#tukey_results = pairwise_tukeyhsd(np.concatenate([methodVE_5, methodNa_5, methodVE_10, methodNa_10, methodVE_11, methodNa_11]),
    #np.concatenate([['VE_5'] * len(methodVE_5), ['Na_5'] * len(methodNa_5), ['VE_10'] * len(methodVE_10), ['Na_10'] * len(methodNa_10), ['VE_11'] * len(methodVE_11), ['Na_11'] * len(methodNa_11)]))

#print(tukey_results)

#print("Hypothesis 2: \n")
#tukey_results2 = pairwise_tukeyhsd(np.concatenate([MFO_5nodes, MDO_5nodes, MFO_10nodes, MDO_10nodes, MFO_15nodes, MDO_15nodes]),
    #np.concatenate([['MFO_5'] * len(MFO_5nodes), ['MDO_5'] * len(MDO_5nodes), ['MFO_10'] * len(MFO_10nodes), ['MDO_10'] * len(MDO_10nodes), ['MFO_15'] * len(MFO_15nodes), ['MDO_15'] * len(MDO_15nodes)]))

#print(tukey_results2)

# Means
## Hypothesis 1 ##
mean_5_nodes_VE = 0.161370162
mean_10_nodes_VE = 3.040625
mean_11_nodes_VE = 16.5869024
mean_5_nodes_Na = 0.332656209
mean_10_nodes_Na = 30.31390625
mean_11_nodes_Na = 306.4947646

## Hypothesis 2 ##
mean_5_nodes_MFO = 0.1128125
mean_10_nodes_MFO = 3.05234375
mean_15_nodes_MFO = 304.3677083
mean_5_nodes_MDO = 0.10484375
mean_10_nodes_MDO = 3.13046875
mean_15_nodes_MDO = 322.61875

# Means Dataframes
## Hypothesis 1 ##
df1 = pd.DataFrame({'Network_Size':['5_Nodes', '10_Nodes', '11_Nodes'],
                   'Variable Elimination':[mean_5_nodes_VE, mean_10_nodes_VE, mean_11_nodes_VE],
                   'Naïve Summing-Out':[mean_5_nodes_Na, mean_10_nodes_Na, mean_11_nodes_Na]})

## Hypothesis 2 ##
df2 = pd.DataFrame({'Network_Size':['5_Nodes', '10_Nodes', '15_Nodes'],
                   'Minimum_Fill_Ordering':[mean_5_nodes_MFO, mean_10_nodes_MFO, mean_15_nodes_MFO],
                   'Minimum_Degree_Ordering':[mean_5_nodes_MDO, mean_10_nodes_MDO, mean_15_nodes_MDO]})

## Hypothesis 1 ##
# Add data to the graph
plt.plot(df1['Network_Size'], df1['Variable Elimination'], marker='o', label='Variable Elimination')
plt.plot(df1['Network_Size'], df1['Naïve Summing-Out'], marker='o', label='Naïve Summing-Out')

# Create labels and title
plt.title('Mean Times by Network Size')
plt.grid(axis='both', linestyle='--', linewidth=0.5)
plt.xlabel('Network Size')
plt.ylabel('Time (in seconds)')

# Add labels to the nodes in the graph
for x, y1, y2 in zip(df1['Network_Size'], df1['Naïve Summing-Out'], df1['Variable Elimination']):
    plt.text(x, y1 + 2, round(y1, 2), ha='center', va='bottom')  # positioning the labels
    plt.text(x, y2 - 4.5, round(y2, 2), ha='center', va='top')

# Show the legend
plt.legend()

# Show the graph
plt.show()


## Hypothesis 2 ##
# Add data to the graph
plt.plot(df2['Network_Size'], df2['Minimum_Fill_Ordering'], marker='o', label='Minimum Fill Ordering')
plt.plot(df2['Network_Size'], df2['Minimum_Degree_Ordering'], marker='o', label='Minimum Degree Ordering')

# Create labels and title
plt.title('Mean Times by Network Size')
plt.grid(axis='both', linestyle='--', linewidth=0.5)
plt.xlabel('Network Size')
plt.ylabel('Time (in seconds)')

# Add labels to the nodes in the graph
for x, y1, y2 in zip(df2['Network_Size'], df2['Minimum_Degree_Ordering'], df2['Minimum_Fill_Ordering']):
    plt.text(x, y1 + 2, round(y1, 2), ha='center', va='bottom')  # positioning the labels
    plt.text(x, y2 - 4.5, round(y2, 2), ha='center', va='top')

# Show the legend
plt.legend()

# Show the graph
plt.show()

print("-----------------------------\n")
print("Means and std Hypothesis 1:")
VE_5nodes_mean = round(sum(methodVE_5)/len(methodVE_5), 2)
VE_5nodes_std = round(statistics.stdev(methodVE_5), 2)

VE_10nodes_mean = round(sum(methodVE_10)/len(methodVE_10), 2)
VE_10nodes_std = round(statistics.stdev(methodVE_10), 2)

VE_11nodes_mean = round(sum(methodVE_11)/len(methodVE_11), 2)
VE_11nodes_std = round(statistics.stdev(methodVE_11), 2)

Na_5nodes_mean = round(sum(methodNa_5)/len(methodNa_5), 2)
Na_5nodes_std = round(statistics.stdev(methodNa_5), 2)

Na_10nodes_mean = round(sum(methodNa_10)/len(methodNa_10), 2)
Na_10nodes_std = round(statistics.stdev(methodNa_10), 2)

Na_11nodes_mean = round(sum(methodNa_11)/len(methodNa_11), 2)
Na_11nodes_std = round(statistics.stdev(methodNa_11), 2)

print('VE_5nodes_mean:', VE_5nodes_mean, '\nVE_5nodes_std:', VE_5nodes_std)
print('VE_10nodes_mean:', VE_10nodes_mean, '\nVE_10nodes_std:', VE_10nodes_std)
print('VE_11nodes_mean:', VE_11nodes_mean, '\nVE_11nodes_std:', VE_11nodes_std)
print('Na_5nodes_mean:', Na_5nodes_mean, '\nNa_5nodes_std:', Na_5nodes_std)
print('Na_10nodes_mean:', Na_10nodes_mean, '\nNa_10nodes_std:', Na_10nodes_std)
print('Na_11nodes_mean:', Na_11nodes_mean, '\nNa_11nodes_std:', Na_11nodes_std)

print("\n Means and std Hypothesis 2:")
MFO_5nodes_mean = round(sum(MFO_5nodes)/len(MFO_5nodes), 2)
MFO_5nodes_std = round(statistics.stdev(MFO_5nodes), 2)

MFO_10nodes_mean = round(sum(MFO_10nodes)/len(MFO_10nodes), 2)
MFO_10nodes_std = round(statistics.stdev(MFO_10nodes), 2)

MFO_15nodes_mean = round(sum(MFO_15nodes)/len(MFO_15nodes), 2)
MFO_15nodes_std = round(statistics.stdev(MFO_15nodes), 2)

MDO_5nodes_mean = round(sum(MDO_5nodes)/len(MDO_5nodes), 2)
MDO_5nodes_std = round(statistics.stdev(MDO_5nodes), 2)

MDO_10nodes_mean = round(sum(MDO_10nodes)/len(MDO_10nodes), 2)
MDO_10nodes_std = round(statistics.stdev(MDO_10nodes), 2)

MDO_15nodes_mean = round(sum(MDO_15nodes)/len(MDO_15nodes), 2)
MDO_15nodes_std = round(statistics.stdev(MDO_15nodes), 2)

print('MFO_5nodes mean:', MFO_5nodes_mean, '\nMFO_5nodes standard deviation:', MFO_5nodes_std)
print('MFO_10nodes mean:', MFO_10nodes_mean, '\nMFO_10nodes standard deviation:', MFO_10nodes_std)
print('MFO_15nodes mean:', MFO_15nodes_mean, '\nMFO_15nodes standard deviation:', MFO_15nodes_std)
print('MDO_5nodes mean:', MDO_5nodes_mean, '\nMDO_5nodes standard deviation:', MDO_5nodes_std)
print('MDO_10nodes mean:', MDO_10nodes_mean, '\nMDO_10nodes standard deviation:', MDO_10nodes_std)
print('MDO_15nodes mean:', MDO_15nodes_mean, '\nMDO_15nodes standard deviation:', MDO_15nodes_std)