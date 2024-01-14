###################################################################################################
# ANALISI DATI 
#
# Per visualizzare i plot togliere il commento alla riga 402
###################################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import statsmodels.tsa.stattools as stools
import statsmodels.api as sm

from scipy import stats
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen as coint_j
from statsmodels.graphics.tsaplots import plot_acf

plt.style.use('seaborn-v0_8-darkgrid')



# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# importo i dati

chn = pd.read_csv('cina.csv', sep='\t')             # 1960-2021
ger = pd.read_csv('germania.csv', sep='\t')         # 1970-2021
ind = pd.read_csv('india.csv', sep='\t')            # 1960-2021
ita = pd.read_csv('italia.csv', sep='\t')           # 1970-2021
usa = pd.read_csv('usa.csv', sep='\t')              # 1966-2021
eng = pd.read_csv('uk.csv', sep='\t')               # 1960-2021
fra = pd.read_csv('francia.csv', sep='\t')          # 1960-2021
saf = pd.read_csv('sudafrica.csv', sep='\t')        # 1960-2021
bra = pd.read_csv('brasile.csv', sep='\t')          # 1960-2021



# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# creo una lista con tutti i dataframes e una con i rispettivi 'titoli'

data = [chn, ger, ind, ita, usa, eng, fra, saf, bra]
titles = ['Cina', 'Germania', 'India', 'Italia', 'Stati Uniti', 'Regno Unito', 'Francia', 'Sudafrica', 'Brasile']



# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# plotto i dati

plt.figure(figsize=(22,4))
for i in range(len(data)) :
    plt.subplot(2,9,i+1)
    plt.plot(data[i]['Year'], data[i]['PIL'])
    plt.title('PIL ' + titles[i])

for i in range(len(data)) :
    plt.subplot(2,9,i+10)
    plt.plot(data[i]['Year'], data[i]['CO2'])
    plt.title('CO2 '+ titles[i])
    plt.tight_layout()
plt.savefig('dati.pdf')



# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# calcolo gli incrementi percentuali di emissioni, ho già quelli di PIL nei dati
for df in data :
    df['DCO2'] = np.divide(np.array(df['Diff CO2']), np.array(df['CO2'])) * 100



# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# plotto gli incrementi

plt.figure(figsize=(22,4))
for i in range(len(data)) :
    plt.subplot(2,9,i+1)
    plt.plot(data[i]['Year'], data[i]['DPIL'])
    plt.title('DPIL ' + titles[i])

for i in range(len(data)) :
    plt.subplot(2,9,i+10)
    plt.plot(data[i]['Year'], data[i]['DCO2'])
    plt.title('DCO2 '+ titles[i])
    plt.tight_layout()
plt.savefig('incrementi.pdf')



# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# definisco il test di stazionarietà di Dickey-Fuller (ADF)

def ADF(timeseries, titolo=''):
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    #print (dfoutput)
    if dfoutput[1] > 0.05:
        print('La serie \"' + titolo + '\" non è stazionaria')
    else:
        print('La serie \"' + titolo + '\" è stazionaria')



# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# eseguo il test di stazionarietà ADF sui dati
print('\n–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')
print('TEST DI STAZIONARIETÀ DICKEY-FULLER SUI DATI')

for i in range(len(data)) :
    print('\n' + titles[i] + ':')
    ADF(data[i]['PIL'], 'PIL')
    ADF(data[i]['CO2'], 'CO2')



# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# eseguo il test di stazionarietà ADF dugli incrementi
print('\n–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')
print('TEST DI STAZIONARIETÀ DICKEY-FULLER SUGLI INCREMENTI')

for i in range(len(data)) :
    print('\n' + titles[i] + ':')
    ADF(data[i]['DPIL'], 'DPIL')
    ADF(data[i]['DCO2'], 'DCO2')



# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# definisco l'analisi di cross-correlation

def CCA(data, maxlag, dt=1, titles=[]):

    cross_corr_max = pd.DataFrame(titles, columns=['Stato'])
    corr_max = []
    lag_max = []
    plt.figure(figsize=(8,7))

    for i in range(len(data)) :

        X = np.array(data[i]['DPIL'])
        Y = np.array(data[i]['DCO2'])

        result_X, result_Y = adfuller(X), adfuller(Y)
        if (result_X[1] > 0.05 or result_Y[1]>0.05):
            print('Le serie temporali non sono stazionarie!')
            return
        else:
            dim = len(X)
            lags = np.arange(-maxlag, maxlag+1, 1)
            lagged_cross_corr = []
            for lag in lags:
                XY = [(X[t+lag], Y[t]) for t in range(dim-np.abs(lag))]
                X_lagged = [el[0] for el in XY]
                Y_lagged = [el[1] for el in XY]
                lagged_cross_corr.append(pearsonr(X_lagged, Y_lagged)[0])
            
            corr_max.append(max(lagged_cross_corr))
            lag_max.append(lagged_cross_corr.index(corr_max[i]) - maxlag)
                
            #plt.figure(figsize=(2, 2), dpi=180)
            plt.subplot(3,3,i+1)
            lags_min = [l for l in lags if l<=0]
            lags_maj = [l for l in lags if l>=0]
            lcc_min = [lagged_cross_corr[i] for i in range(len(lags)) if lags[i]<=0]
            lcc_maj = [lagged_cross_corr[i] for i in range(len(lags)) if lags[i]>=0]
            plt.plot(lags_min, lcc_min, lw=3, color='b')
            plt.plot(lags_maj, lcc_maj, lw=3, color='r')
            plt.vlines(0, -1, 1, lw=2, ls='--', color='k')
            plt.xlabel('time lags'), plt.ylabel('lagged-cross-correlation')
            plt.text(-maxlag, 0.8, 'PIL precede CO2', color='b', fontsize = 7)
            plt.text(maxlag-9, 0.8, 'CO2 precede PIL', color='r', fontsize = 7)
            plt.tight_layout()
            plt.title(titles[i])
    plt.savefig('cross.pdf')
    
    cross_corr_max['Lag'] = lag_max
    cross_corr_max['Corr'] = corr_max
    return cross_corr_max

    #plt.show()





    corr_mean = []
    plt.figure(figsize=(8,7))

    for i in range(len(data)) :

        X = np.array(data[i]['DPIL'])
        Y = np.array(data[i]['DCO2'])

        plt.subplot(3,3,i+1)
        for j in range(N) :
            random.shuffle(X)
            random.shuffle(Y)

            result_X, result_Y = adfuller(X), adfuller(Y)
            while result_X[1] > 0.05 :
                random.shuffle(X)
                result_X = adfuller(X)
            while result_Y[1] > 0.05 :
                random.shuffle(Y)
                result_Y = adfuller(Y)

            dim = len(X)
            lags = np.arange(-maxlag, maxlag+1, 1)
            lagged_cross_corr = []
            for lag in lags:
                XY = [(X[t+lag], Y[t]) for t in range(dim-np.abs(lag))]
                X_lagged = [el[0] for el in XY]
                Y_lagged = [el[1] for el in XY]
                lagged_cross_corr.append(pearsonr(X_lagged, Y_lagged)[0])

        plt.hist(lagged_cross_corr, bins=20)
        corr_mean.append(np.mean(np.array(lagged_cross_corr)))
        plt.axvline(corr_mean[i], color='tab:orange', linestyle='--', label='Media')
        plt.axvline(cross_corr_max['Corr'][i], color='tab:green', linestyle='--', label='Originale massima')
        plt.legend()
        plt.tight_layout()
        plt.title(titles[i])



# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# eseguo l'analisi di cross-correlation
print('\n–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')
print('MASSIMA CROSS-CORRELATION CON LAG')

maxlag = 10
dt=1
cross_corr_max = CCA(data, maxlag, dt, titles)

print(cross_corr_max)
print('\nSe \"Lag\" è negativo => PIL precede CO2')
print('Se \"Lag\" è positivo => CO2 precede PIL')



# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# definisco l'analisi di cross-correlation in cui riordino casualmente le serie temporali

def CCA_shuffle(data, maxlag, dt=1, titles=[], N=10000):

    plt.figure(figsize=(8,7))
    for i in range(len(data)) :

        X = np.array(data[i]['DPIL'])
        Y = np.array(data[i]['DCO2'])
        corr_shuff_max = []

        plt.subplot(3,3,i+1)
        for j in range(N) :
            random.shuffle(X)
            random.shuffle(Y)

            result_X, result_Y = adfuller(X), adfuller(Y)
            while result_X[1] > 0.05 :
                random.shuffle(X)
                result_X = adfuller(X)
            while result_Y[1] > 0.05 :
                random.shuffle(Y)
                result_Y = adfuller(Y)

            dim = len(X)
            lags = np.arange(-maxlag, maxlag+1, 1)
            lagged_cross_corr = []
            for lag in lags:
                XY = [(X[t+lag], Y[t]) for t in range(dim-np.abs(lag))]
                X_lagged = [el[0] for el in XY]
                Y_lagged = [el[1] for el in XY]
                lagged_cross_corr.append(pearsonr(X_lagged, Y_lagged)[0])
            corr_shuff_max.append(max(lagged_cross_corr))

        plt.hist(corr_shuff_max, bins=20, color='salmon')
        mean = np.mean(corr_shuff_max)
        sigma = np.std(corr_shuff_max)

        plt.axvline(mean, color='navy', linestyle='--')
        plt.axvline(mean + 3*sigma, color='navy', linestyle='--')
        plt.axvline(mean - 3*sigma, color='navy', linestyle='--')
        plt.axvline(cross_corr_max['Corr'][i], color='red', linestyle='--')

        plt.tight_layout()
        plt.title(titles[i])
    plt.savefig('shuffle.pdf')



# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# eseguo l'analisi di cross-correlation riordinata

maxlag = 10
dt=1
n = 10000
CCA_shuffle(data, maxlag, dt, titles, N=n)



# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# definisco un meotodo per la scelta del miglior ordine del modello (lag)

def SceltaLag(df):
    aic, bic, fpe, hqic = [], [], [], []
    model = VAR(df) 
    p = np.arange(1,40)
    for i in p:
        result = model.fit(i)
        aic.append(result.aic)
        bic.append(result.bic)
        fpe.append(result.fpe)
        hqic.append(result.hqic)
    lags_metrics_df = pd.DataFrame({'AIC': aic, 
                                    'BIC': bic, 
                                    'HQIC': hqic,
                                    'FPE': fpe}, 
                                   index=p)
    print(lags_metrics_df.idxmin(axis=0))
    print()
    return lags_metrics_df['AIC'].idxmin(axis=0)



# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# scelgo i vari ordini dei VAR utilizzando AIC
# devo eliminare i primi punti dei datasets (1 o 3) perchè altrimenti non posso applicare AIC
print('\n–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')
print('SCELTA DELL\'ORDINE DEI MODELLI VAR PER IL TEST DI CAUSALITÀ')

lags = []
for i in range(len(data)) : 
    if i == 1 or i == 3 :
        start = 3
    else :
        start = 1
    print(titles[i])
    lags.append(SceltaLag(data[i][['DPIL','DCO2']][start:]))



# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# eseguo il test per la Granger causality costruendo i VAR models
# ipotesi nulla è assenza di Granger-causalità
print('\n–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')
print('TEST DI CAUSALITÀ DI GRANGER SUGLI INCREMENTI (STAZIONARI)')

granger = pd.DataFrame(titles, columns=['Stato'])
granger['Lag'] = lags
models = []
fits = []
p1 = []
p2 = []

for i in range(len(data)) :
    models.append(VAR(data[i][['DPIL','DCO2']]))
    fits.append(models[i].fit(lags[i]))
    p1.append(fits[i].test_causality('DCO2', 'DPIL', kind='f').summary()[1][2])
    p2.append(fits[i].test_causality('DPIL', 'DCO2', kind='f').summary()[1][2])

granger['p value (DPIL => DCO2)'] = p1
granger['p value (DCO2 => DPIL)'] = p2

print(granger)



# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# risultati Engle-Granger test
# ipotesi nulla è assenza di cointegrazione
print('\n–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')
print('TEST DI COINTEGRAZIONE DI ENGLE-GRANGER')

englegranger = pd.DataFrame(titles, columns=['Stato'])
p = []
stat = []
crit1 = []
crit5 = []
crit10 = []

for i in range(len(data)) :
    egtest = stools.coint(data[i]['CO2'], data[i]['PIL'], method='aeg', trend='ct', autolag='AIC')

    p.append(egtest[1])
    stat.append(egtest[0])
    crit1.append(egtest[2][0])
    crit5.append(egtest[2][1])
    crit10.append(egtest[2][2])

englegranger['p value'] = p
englegranger['Statistic'] = stat
englegranger['Critical 1%'] = crit1
englegranger['Critical 5%'] = crit5
englegranger['Critical 10%'] = crit10

print(englegranger)


#plt.show()
print('\n–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')