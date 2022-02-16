# python3 DC_Prob_1.py

import numpy as np
import pandas as pd
import os
import argparse
import pickle
import matplotlib.pyplot as plt
from operator import itemgetter, attrgetter
import sklearn
import csv

# ****************************************** Functions ******************************************
	# To make histograms:
def Hist(X, Col, Tit, x_nam, y_nam, fig_nam):
		# Calculo del número de bins optimo para los histogramas vía Freedman-Diaconis number:
	q25, q75 = np.percentile(X, [25, 75])
	bin_width = 2*(q75 - q25)*len(X)**(-1/3)
	bs = round((X.max() - X.min())/bin_width)
		# Makes a histogram for data:
	plt.figure(figsize=(8, 8))
	plt.hist(X, density=True, bins=bs, color=Col)
	plt.title(Tit, fontsize=20)
	plt.xlabel(x_nam, fontsize=18)
	plt.ylabel(y_nam, fontsize=18)	
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.savefig(fig_nam)
# ****************************************** Functions ******************************************

	# Read data:
data = pd.read_csv('Data.csv', delimiter=' ', engine='python')

	# Definir variables de datos:
Prom = data['PROM_ACUMULADO']
Salario = data['SALARIO']
N = len(Prom)
mean_Sal = np.mean(Salario)
std_Sal = np.std(Salario)
mean_Prom = np.mean(Prom)
std_Prom = np.std(Prom)

print(mean_Sal, std_Sal)

	# Makes an array with no repeated values from Salario and takes into acount the number of repetitions 
	# of its values:
no_rep_sal, counts_sal = np.unique(Salario, return_counts=True)

ave_Prom = np.zeros(len(no_rep_sal))
std_Prom = np.zeros(len(no_rep_sal))
for i in range(len(no_rep_sal)):
	aux = np.zeros(counts_sal[i], dtype=float)
	value = no_rep_sal[i]
	a = 0
	for j in range(N):
		if Salario[j]==value:
			aux[a] = Prom[j]
			a += 1
	ave_Prom[i] = np.mean(aux)
	std_Prom[i] = np.std(aux)
	np.delete(aux, 0)

i = 0
for val in no_rep_sal:
	print(val, ave_Prom[i], std_Prom[i])
	i += 1

mean_ave_Prom = np.mean(ave_Prom)
std_ave_Prom = np.std(ave_Prom)

print(mean_ave_Prom, std_ave_Prom)

	# Separte the total sample into two groups, the first group with salarios greather than mean salario and
	# the second with salarios less than mean salario:
great_less_Sal = np.zeros(N)
a = 0
great_Sal = 0
	# Takes 1 value if salario is greater than mean salario, 0 if not:
for values in Salario:
	if values>=mean_Sal:
		great_less_Sal[a]=1
		great_Sal += 1
	a += 1
less_Sal = N - great_Sal

print(great_Sal, less_Sal)

	# Creates two groups of notes according the sample of great-less salarios:
less_Notes = np.zeros(less_Sal)
great_Notes = np.zeros(great_Sal)
a = 0
b = 0
i = 0
for vals in great_less_Sal:
	if vals==1:
		great_Notes[a] = Prom[i]
		a += 1
	else:
		less_Notes[b] = Prom[i]
		b += 1
	i += 1
mean_great_Notes = np.mean(great_Notes)
std_great_Notes = np.mean(great_Notes)
mean_less_Notes = np.mean(less_Notes)
std_less_Notes = np.mean(less_Notes)
real_diff = mean_great_Notes - mean_less_Notes
real_diff_std = std_great_Notes - std_less_Notes

print(real_diff, real_diff_std)

	# Shuffle method:
cumulativa = np.linspace(1.0/N, 1.0, N)
n_iteraciones = 1000
diffs = np.zeros(n_iteraciones)
diffs_std = np.zeros(n_iteraciones)

for i in range(n_iteraciones):
		# Shuffle: Reordena los datos aleatoreamente:
	Prom_Shuff = sklearn.utils.shuffle(Prom)
		# Hace un array tomando los primeros great_Sal datos:
	notas_fake_f = Prom_Shuff[:great_Sal]
		# Hace un array después del dato great_Sal en adelante:
	notas_fake_m = Prom_Shuff[great_Sal:]
		# Define la nueva variable z (vía promedios):
	diffs[i] = np.mean(notas_fake_f) - np.mean(notas_fake_m)
		# Define la nueva variable z (vía std)
	diffs_std[i] = np.std(notas_fake_f) - np.std(notas_fake_m)

	# Define el p-value (para promedios) con el valor frac_altos:
frac_altos = np.count_nonzero(diffs>real_diff)/len(diffs)

	# Define el p-value (para std) con el valor frac_altos_std:
frac_altos_std = np.count_nonzero(diffs_std<real_diff_std)/len(diffs_std)

print(frac_altos, frac_altos_std)
print(np.max(diffs), np.max(diffs_std))

	# ****************************************** To make all plots ******************************************

#Hist(data['PROM_2019-2'], 'blue', 'Promedio 2019-2 Histograma', 'Promedios 2019-2', 'Probability', 'Hist_2019-2.png')
#Hist(data['PROM_2020-1'], 'blue', 'Promedio 2020-1 Histograma', 'Promedios 2020-1', 'Probability', 'Hist_2020-1.png')
#Hist(data['PROM_2020-2'], 'blue', 'Promedio 2020-2 Histograma', 'Promedios 2020-2', 'Probability', 'Hist_2020-2.png')
#Hist(data['PROM_2021-1'], 'blue', 'Promedio 2021-1 Histograma', 'Promedios 2021-1', 'Probability', 'Hist_2021-1.png')
#Hist(Prom, 'blue', 'Promedio Acumulado Histograma', 'Promedio Acumulado', 'Probability', 'Hist_Prom_Cumm.png')
#Hist(Salario, 'blue', 'Salario Histograma', 'Salario (en millones de $)', 'Probability', 'Salario.png')


'''plt.figure(figsize=(8, 8))
plt.scatter(Salario, Prom, alpha=0.5, s=75)
plt.title('Salario - Prom. Acumulado', fontsize=20)
plt.xlabel('Salario (en millones de $)', fontsize=18)
plt.ylabel('Prom. Acumulado', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('Salario_Prom.png')'''

'''plt.figure(figsize=(8, 8))
plt.errorbar(no_rep_sal, ave_Prom, yerr=std_Prom, fmt='o', color='orange', ecolor='lightgreen', capsize=5)
plt.scatter(no_rep_sal, ave_Prom, alpha=0.5, s=75)
plt.vlines(mean_Sal, 4, 5, color='red')
plt.vlines(mean_Sal-std_Sal, 4, 5, color='red')
plt.vlines(mean_Sal+std_Sal, 4, 5, color='red')
plt.vlines(mean_Sal, 4, 5, color='red')
plt.fill_between([mean_Sal-std_Sal, mean_Sal+std_Sal], [5, 5], [4, 4], facecolor='red', alpha=0.1)
plt.hlines(mean_Prom, 2e6,7.5e6, color='blue')
plt.hlines(mean_ave_Prom+std_ave_Prom, 2e6,7.5e6, color='blue')
plt.hlines(mean_ave_Prom-std_ave_Prom, 2e6,7.5e6, color='blue')
plt.fill_between([2e6,7.5e6], [mean_ave_Prom-std_ave_Prom,mean_ave_Prom-std_ave_Prom], [mean_ave_Prom+std_ave_Prom,mean_ave_Prom+std_ave_Prom], facecolor='blue', alpha=0.1)
plt.title('Promedio notas agrupadas por nivel salarial', fontsize=20)
plt.xlabel('Salario (en millones de $)', fontsize=18)
plt.ylabel('Promedio Agrupado', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('Salario_ave_Prom.png')'''



























