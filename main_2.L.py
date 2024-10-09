# Import standard

# Sandro
from collections import Counter
import sys
import io

# Cattura l'output
# output_capture = io.StringIO()
# sys.stdout = output_capture
# sys.stderr = output_capture
#Sandro

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import exponweib, weibull_min
# from clustering_module import multi_sensor_cc, single_dimensional_fcluster  # Fonte: GEIRI, commentto poiche
# non serve per i primi 3 punti

# Import per clustering
from kneed import KneeLocator, DataGenerator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


# from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer #NUOVO ERRORE DI IMPORTAZIONE (senza di questo, gli altri errori scompaiono)
# sembra che faccia problemi con versione sklearn >0.24


from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster


#import per clustering divisivo gerarchico top-down
from HiPart.clustering import BisectingKmeans
from HiPart.clustering import DePDDP    #principale
from HiPart.clustering import IPDDP
from HiPart.clustering import KMPDDP
from HiPart.clustering import PDDP
from HiPart.clustering import MDH
from sklearn.datasets import make_blobs #principale
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi


from HiPart import visualizations as viz
# import matplotlib.pyplot as plt
import time


#import for iterative confusion matrix
from sklearn import metrics
from itertools import permutations
from scipy.spatial import distance

#import x input data da piattaforma TP (momentaneamente disabilitato ma funzionante)
from suds.client import Client
import zipfile
import pandas as pd
from io import BytesIO, TextIOWrapper
import base64
import shutil
import os

#import per output data (file labels_df.cvs) su piattaforma TP (momentaneamente disabilitato ma funzionante)
import io
import suds
from suds.client import Client
import pandas as pd
from io import BytesIO, TextIOWrapper
import base64


###### MEMO PER IMPORTAZIONE PACCHETTI CON ANACONDA (usato per creazione ambiente main)
# conda install -c anaconda numpy
# conda install -c anaconda pandas
# conda install -c conda-forge matplotlib
# conda install -c anaconda scipy
# conda install -c anaconda scikit-learn
# conda install -c conda-forge kneed
# conda install -c districtdatalabs yellowbrick
# conda install -c conda-forge pickle5 (non compatibile con python 3.12 nè 3.11; funziona solo fino a python 3.8)=>usare pacchetto interno
# ##### fine MEMO


################################################
# FUNZIONI DI PREPROCESSING
def constant_detrending(array):  # N.B. Detrend vuole array in input
    signal_array_d1 = np.array([detrend(x, type='constant') for x in array])
    return signal_array_d1


def linear_detrending(array):  # N.B. Detrend vuole array in input
    signal_array_d2 = np.array([detrend(x, type='linear') for x in array])
    return signal_array_d2


def normalization_2D_manual(array):
    # Agisce su array 2D che rappresenta il dataframe iniziale
    max_value = np.max(np.max(abs(array), axis=1))
    # Indicare axis=1 per specificare di procedere prima per righe e poi per colonne
    array_norm = abs(array) / max_value
    # Verifica funzionamento normalizzazione: il massimo assoluto vale 1
    # max_verifica_norm = np.max(np.max(array_norm))
    return array_norm


def normalization_2D_scikit(array):  # Formula da documentazione scikit learn, ma implementata a mano per input array 2D
    # Agisce su array 2D che rappresenta il dataframe iniziale
    max_value = np.max(np.max(abs(array), axis=1))
    min_value = np.min(np.min(abs(array), axis=1))
    # Indicare axis=1 per specificare di procedere prima per righe e poi per colonne
    array_norm = (abs(array) - min_value) / (max_value - min_value)
    # Verifica funzionamento normalizzazione: il massimo assoluto vale 1
    # max_verifica_norm = np.max(np.max(array_norm))
    return array_norm


def preprocessing(signal_array): #array 2D in input #I SEGNALI PER ORA NON SONO STATI ESTRATTI: RICORDARSI DI CONVERTIRLI IN FLOAT!!! (usare la funzione già fatta per le altre features e stratte direttamente)
    # Constant detrending - MOMENTANEAMENTE DISATTIVATO
        #signal_array_d1 = constant_detrending(signal_array)
    # Linear detrending - MOMENTANEAMENTE DISATTIVATO
        #signal_array_d2 = linear_detrending(signal_array_d1)
        #signal_array_d2_df = pd.DataFrame(signal_array_d2)
    # Normalizzazione- MOMENTANEAMENTE DISATTIVATO
        #signal_norm = normalization_2D_manual(signal_array_d2).- MOMENTANEAMENTE DISATTIVATO
    # signal_norm = normalization_2D_scikit(signal_array_d2) # Alternativa alla normalizzazione manuale
    # Conversione in dataframe dell'array in output
        # signal_array_norm = pd.DataFrame(signal_norm)
    signal_array_d2_df=signal_array # - DA RIMUOVERE DOPO ATTIVAZIONE
    signal_array_norm=signal_array # - DA RIMUOVERE DOPO ATTIVAZIONE
    return signal_array_d2_df, signal_array_norm

# FUNZIONI DI PREPROCESSING - FINE
################################################

################################################
# FUNZIONI DI ESTRAZIONE SINGOLE FEATURES

def feature_extraction_AF1(df):
    AF1 = df['AF1']
    return AF1
def feature_extraction_AF2(df):
    AF2 = df['AF2']
    return AF2
def feature_extraction_AF3(df):
    AF3 = df['AF3']
    return AF3
def feature_extraction_A01(df): #automatizzare features 1-25
    A01 = df['A01']
    return A01
def feature_extraction_A02(df):
    A02 = df['A02']
    return A02
def feature_extraction_A03(df):
    A03 = df['A03']
    return A03
def feature_extraction_A04(df):
    A04 = df['A04']
    return A04
def feature_extraction_A05(df):
    A05 = df['A05']
    return A05
def feature_extraction_A06(df):
    A06 = df['A06']
    return A06
def feature_extraction_A07(df):
    A07 = df['A07']
    return A07
def feature_extraction_A08(df):
    A08 = df['A08']
    return A08
def feature_extraction_A09(df):
    A09 = df['A09']
    return A09
def feature_extraction_A10(df):
    A10 = df['A10']
    return A10
def feature_extraction_A11(df):
    A11 = df['A11']
    return A11
def feature_extraction_A12(df):
    A12 = df['A12']
    return A12
def feature_extraction_A13(df):
    A13 = df['A13']
    return A13
def feature_extraction_A14(df):
    A14 = df['A14']
    return A14
def feature_extraction_A15(df):
    A15 = df['A15']
    return A15
def feature_extraction_A16(df):
    A16 = df['A16']
    return A16
def feature_extraction_A17(df):
    A17 = df['A17']
    return A17
def feature_extraction_A18(df):
    A18 = df['A18']
    return A18
def feature_extraction_A19(df):
    A19 = df['A19']
    return A19
def feature_extraction_A20(df):
    A20 = df['A20']
    return A20
def feature_extraction_A21(df):
    A21 = df['A21']
    return A21
def feature_extraction_A22(df):
    A22 = df['A22']
    return A22
def feature_extraction_A23(df):
    A23 = df['A23']
    return A23
def feature_extraction_A24(df):
    A24 = df['A24']
    return A24
# def feature_extraction_A25(df):
#     A25 = df['A25']
#     return A25
if(0): #disabilitata poiché non funzionante (non è numerica) e non è una feature opportuna (serve solo per stratificare il campionamento per area geografica GME, non per clusterizzare)
    def feature_extraction_Area_GME(df):
        Area_GME = df['Area_GME']
        return Area_GME

# FUNZIONI DI ESTRAZIONE SINGOLE FEATURES - FINE
################################################

################################################################################################
# LANCIO ESTRAZIONE FEATURES COME SELEZIONATO DA PANNELLO DI CONTROLLO FEATURES (DIZIONARIO FEATURES)

def feature_extraction(dataframe, dataframe_norm, df, sampling_rate, opt): #indipendente dai dataframes: variabili non usate ma lasciate per eventuali estensioni
    opt_true = {key: value for key, value in opt.items() if value[0]}
    features = {}
    for key, val in opt_true.items():
        features[key] = eval(val[1] + '(' + val[2] + ')')
    features = pd.DataFrame.from_dict(features)

    print('features richieste:')
    print(features)
    features.to_csv('./output/features_extraction_function_output.csv')
    print(features.dtypes)

    #SOSTITUZIONE DELLE VIRGOLE CON DEI PUNTI PER SEPARARE I DECIMALI AL FINE DELLA CONVERSIONE IN FLOAT
    for key, val in opt_true.items():
        features[key] = features[key].replace(',', '.', regex=True)
        features[key] = features[key].astype(float)
    # features.types
    print('features richieste float:')
    print(features)
    print(features.dtypes)
    return features

# LANCIO ESTRAZIONE FEATURES COME SELEZIONATO DA PANNELLO DI CONTROLLO FEATURES - FINE
################################################################################################


#######################################################################
# BLOCCO 1 - ESTRAZIONE FEATURES (Importazione dati; Preprocessing: detrend e normalizzazione; Features selection: tramite pannello di controllo con dizionario variabile di features)

def main_features(opt_features,feature_set_selection):
    print_test('PyCharm-test_AE_TP_RZ-start function main_features')

    #############################################################
    # Load data FROM PICKLE FILE TO DATAFRAME - DATI SERIALIZZATI (disabilitato ma funzionante - ma solo fino a Python 3.8 !!!!!)
    #############################################################
    if(0):
        # CAMPIONAMENTO DATI PER OGNI FONTE
        with open('./output/TEST_DATA_experiment_all_sources_data', 'rb') as pickle_file_1:
            data_1 = pickle.load(pickle_file_1)
            signals = pd.DataFrame(data_1[3])
        # DATI GENERALI DI OGNI FONTE (poi sarà il singolo POD)
        with open('./output/TEST_DATA_experiment_all_sources_df', 'rb') as pickle_file_2:
            df = pickle.load(pickle_file_2)
            df = df.sort_values('row_id')
        # Tutti i segnali sono acquisiti a cadenza oraria; (sampling rate = 3600, settato a mano)
        sampling_rate = 3600  # in base al sistema di acquisizione e alla frequenza di campionamento
        # N.B.: unità di misura: s; spazio temporale tra campioni consecutivi

        test_df=df.head(100)
        test_signals=signals.head(100)
        test_df.to_csv('./output/TEST_DATA_FRAME_input_PICKLE.csv')
        test_signals.to_csv('./output/TEST_DATA_SIGNALS_input_PICKLE.csv')

    # Load data FROM PICKLE FILE TO DATAFRAME - DATI SERIALIZZATI - FINE
    ####################################################################


    #########################################
    # Load data FROM CSV FILE TO DATAFRAME
    #########################################
    if(1): #attivare

        # Ottieni il percorso della directory dello script corrente
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Combina la directory dello script con il nome del file
        file_path_gen = os.path.join(script_dir, './input/curve-gen-24.csv')
        file_path_feb = os.path.join(script_dir, './input/curve-feb-24.csv')
                
        # df1 = pd.read_csv('curve-gen-24.csv', sep=';')  # USANO IL PUNTO E VIRGOLA COME SEPARATORE DEL CSV !!
        # USANO IL PUNTO E VIRGOLA COME SEPARATORE DEL CSV !!
        df1 = pd.read_csv(file_path_gen, sep=';')
        # df2 = pd.read_csv('curve-feb-24.csv', sep=';')
        df2 = pd.read_csv(file_path_feb, sep=';')

        dfTOTinputAE = df1._append(df2)  #concatena (append) df1 e df2

        # DROP COLONNE INUTILIZZATE
        dfTOTinputAE.drop(['APK','AOP','ATO','R01','R02','R03','R04','R05','R06','R07','R08','R09','R10','R11','R12','R13','R14','R15','R16','R17','R18','R19','R20','R21','R22','R23','R24','R25','RF1','RF2','RF3','RPK','ROP','RTO','PF1','PF2','PF3','PTO','RCF1','RCF2','RCF3','RCPK','RCOP','RCTO','RIF1','RIF2','RIF3','RIPK','RIOP','RITO','IDCategoriaUso','CategoriaUso'], axis=1, inplace=True)

        dfTOTinputAE.to_csv('./output/curve_GEN-FEB-24_AE_input_CSV.csv')
        print("ver2.L - AHC_POD (campione stratificato e classification dataset completo per POD con FEATURE SET SELECTION")
        print('curve_GEN-FEB_2024:')
        print(dfTOTinputAE.head(10))



    ########################################

    ###################################
    # LOAD DATA FROM URL TO DATAFRAME (disabilitato ma funzionante)
    ###################################
    if (0):
        wcf_url = "http://sdm.demat.develop.technoplants.lan/EngineAssets.svc?wsdl"
        client = Client(wcf_url)


        # carica le info delle track padri delle raw (i due files)
        trackInfo_file_01 = client.service.GetTrackRawDataInfoByTrack(5100)  # file 1 -> id track 5100
        trackInfo_file_02 = client.service.GetTrackRawDataInfoByTrack(5101)  # file 2 -> id track 5101

        raw_id = trackInfo_file_01.RawId

        raws_traks = [trackInfo_file_01.RawId]  # ,trackInfo_file_02.RawId


        for raw_id in raws_traks:
            byte_array = base64.b64decode(client.service.DownloadTrack(raw_id))
            byte_io = BytesIO(byte_array)


            # decompatta il file e lo copia nel dataframe
            with zipfile.ZipFile(byte_io, 'r') as zip_file:
                csv_file_name = zip_file.namelist()[0]
                with zip_file.open(csv_file_name) as csv_file:
                    csv_text_wrapper = TextIOWrapper(csv_file, encoding='iso-8859-1')

                    df1 = pd.read_csv(csv_text_wrapper, delimiter=';', decimal=',')



            print(df1)

        raw_id = trackInfo_file_02.RawId

        raws_traks = [trackInfo_file_02.RawId]  # trackInfo_file_01.RawId,

        for raw_id in raws_traks:
            byte_array = base64.b64decode(client.service.DownloadTrack(raw_id))
            byte_io = BytesIO(byte_array)

            # decompatta il file e lo copia nel dataframe
            with zipfile.ZipFile(byte_io, 'r') as zip_file:
                csv_file_name = zip_file.namelist()[0]
                with zip_file.open(csv_file_name) as csv_file:
                    csv_text_wrapper = TextIOWrapper(csv_file, encoding='iso-8859-1')

                    df2 = pd.read_csv(csv_text_wrapper, delimiter=';', decimal=',')


            print(df2)

        # LOAD DATA FROM URL TO DATAFRAME - FINE
        ###################################

        dfTOTinputAE = df1._append(df2)  # concatena (append) df1 e df2


    ###############################################################
    #COMPLETAMENTO DB CON INSERIMENTO CLASSIFICAZIONE REGIONALE GME
    ###############################################################
    if(1): #attivare
        #CARICAMENTO DB PROVINCE da CSV
        file_path_Aree_GME_CSV = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), './input/Aree_GME_CSV.csv')
        df_PROV = pd.read_csv(file_path_Aree_GME_CSV, sep=';') # I dati 2023 erano separati da punto e virgola
        #df_PROV = pd.read_csv('Aree_GME_txt_tab.txt', sep='\t') # prova per dati 2024 separati da tab

        # JOIN DB - AGGIUNTA COLONNA Area_GME (uso merge perchè hanno colonna in comune)
        dfTOTinputAE= dfTOTinputAE.merge(df_PROV, how = 'left', on = 'siglaprovincia')
        print("**************AREE GME**************")
        print(dfTOTinputAE.head(10))
        dfTOTinputAE.to_csv(
            './output/curve_GEN-FEB_2024_AE_input_CSV_Aree_GME.csv')

        #####################
        #NB:  salvo copia del DB iniziale al fine di successivo riutilizzo - senza rifare un passaggio su disco (in fase di classificazione)...ricordarsi di riapplicare lo standardscaler
        Data_Frame_iniziale_GME=dfTOTinputAE
        print("INFO iniziali su 'Data_Frame_iniziale_GME':")
        Data_Frame_iniziale_GME.info()

        ########################

    # COMPLETAMENTO DB CON INSERIMENTO CLASSIFICAZIONE REGIONALE GME - FINE
    ###############################################################



    ###############################################################
    # AGGREGAZIONE PER POD
    ###############################################################
    if(1):
        ###SUBSET COLONNE UTILIZZATE PER L'AGGREGAZIONE

        #DF dati geografici per POD univoci
        dfPOD_geo= dfTOTinputAE[["pod_id", "cap", "comune",	"siglaprovincia","Area_GME"]]
        dfPOD_geo=dfPOD_geo.drop_duplicates('pod_id')
        dfPOD_geo.to_csv("./output/output_GEO_POD_univoci.csv")

        #df dati numerici mediati per POD

        dfPOD_rec = pd.DataFrame()


        dfPOD_rec['pod_id'] = dfTOTinputAE['pod_id']
        dfPOD_rec['AF1'] = dfTOTinputAE['AF1'].str.replace(',', '.').astype(float).round(3)
        dfPOD_rec['AF2'] = dfTOTinputAE['AF2'].str.replace(',', '.').astype(float).round(3)
        dfPOD_rec['AF3'] = dfTOTinputAE['AF3'].str.replace(',', '.').astype(float).round(3)
        dfPOD_rec['A01'] = dfTOTinputAE['A01'].str.replace(',', '.').astype(float).round(3)  # automatizzare
        dfPOD_rec['A02'] = dfTOTinputAE['A02'].str.replace(',', '.').astype(float).round(3)
        dfPOD_rec['A03'] = dfTOTinputAE['A03'].str.replace(',', '.').astype(float).round(3)
        dfPOD_rec['A04'] = dfTOTinputAE['A04'].str.replace(',', '.').astype(float).round(3)
        dfPOD_rec['A05'] = dfTOTinputAE['A05'].str.replace(',', '.').astype(float).round(3)
        dfPOD_rec['A06'] = dfTOTinputAE['A06'].str.replace(',', '.').astype(float).round(3)
        dfPOD_rec['A07'] = dfTOTinputAE['A07'].str.replace(',', '.').astype(float).round(3)
        dfPOD_rec['A08'] = dfTOTinputAE['A08'].str.replace(',', '.').astype(float).round(3)
        dfPOD_rec['A09'] = dfTOTinputAE['A09'].str.replace(',', '.').astype(float).round(3)
        dfPOD_rec['A10'] = dfTOTinputAE['A10'].str.replace(',', '.').astype(float).round(3)
        dfPOD_rec['A11'] = dfTOTinputAE['A11'].str.replace(',', '.').astype(float).round(3)  # automatizzare
        dfPOD_rec['A12'] = dfTOTinputAE['A12'].str.replace(',', '.').astype(float).round(3)
        dfPOD_rec['A13'] = dfTOTinputAE['A13'].str.replace(',', '.').astype(float).round(3)
        dfPOD_rec['A14'] = dfTOTinputAE['A14'].str.replace(',', '.').astype(float).round(3)
        dfPOD_rec['A15'] = dfTOTinputAE['A15'].str.replace(',', '.').astype(float).round(3)
        dfPOD_rec['A16'] = dfTOTinputAE['A16'].str.replace(',', '.').astype(float).round(3)
        dfPOD_rec['A17'] = dfTOTinputAE['A17'].str.replace(',', '.').astype(float).round(3)
        dfPOD_rec['A18'] = dfTOTinputAE['A18'].str.replace(',', '.').astype(float).round(3)
        dfPOD_rec['A19'] = dfTOTinputAE['A19'].str.replace(',', '.').astype(float).round(3)
        dfPOD_rec['A20'] = dfTOTinputAE['A20'].str.replace(',', '.').astype(float).round(3)
        dfPOD_rec['A21'] = dfTOTinputAE['A21'].str.replace(',', '.').astype(float).round(3)  # automatizzare
        dfPOD_rec['A22'] = dfTOTinputAE['A22'].str.replace(',', '.').astype(float).round(3)
        dfPOD_rec['A23'] = dfTOTinputAE['A23'].str.replace(',', '.').astype(float).round(3)
        dfPOD_rec['A24'] = dfTOTinputAE['A24'].str.replace(',', '.').astype(float).round(3)
        #dfPOD_rec['A25'] = dfTOTinputAE['A25'].str.replace(',', '.').astype(float).round(3)
        dfPOD_rec.info()

        dfPOD_rec_med = pd.DataFrame(dfPOD_rec.groupby('pod_id').aggregate('median').round(3))
        dfPOD_rec_mean = pd.DataFrame(dfPOD_rec.groupby('pod_id').aggregate('mean').round(3))
        dfPOD_rec_max = pd.DataFrame(dfPOD_rec.groupby('pod_id').aggregate('max').round(3))
        dfPOD_rec_min = pd.DataFrame(dfPOD_rec.groupby('pod_id').aggregate('min').round(3))
        dfPOD_rec_std = pd.DataFrame(dfPOD_rec.groupby('pod_id').aggregate('std').round(3))

        dfPOD_rec_med.to_csv("./output/mediane_orarie_POD_univoci.csv")
        dfPOD_rec_mean.to_csv("./output/medie_orarie_POD_univoci.csv")
        dfPOD_rec_max.to_csv("./output/massime_orarie_POD_univoci.csv")
        dfPOD_rec_min.to_csv("./output/minime_orarie_POD_univoci.csv")
        dfPOD_rec_std.to_csv("./output/std_orarie_POD_univoci.csv")

        #merge dei dati geografici univoci per pod e relative mediane delle registrazioni

        #dfTOTinputAE=merge geo_univ e rec_mediate (scegliere se usare medie, mediane max min std ecc : USO MEDIANE)
        dfTOTinputAE= dfPOD_geo.merge(dfPOD_rec_mean, how = 'left', on = 'pod_id')
        dfTOTinputAE.to_csv(
            './output/curveMEDIE_GEN-FEB_2024_AE_input_CSV_Aree_GME_POD_UNIVOCI.csv')
        dfTOTinputAE_POD=dfTOTinputAE
        print(dfTOTinputAE.head(10))
        ########################
        #AGGREGAZIONE PER POD  - FINE
        ###################################






    ###############################################################
    # CAMPIONAMENTO STRATIFICATO (MULTIPLO) IN BASE ALLA CLASSIFICAZIONE REGIONALE GME e dati geografici (sottoinsiemi di GME)
    ###############################################################
    if(1): #attivare per AHC+KNear e AHC+KNear_POD; disattivare per KMeans
        # TEST STRATIFIAZIONE MULTIPLA OK
        #possibile pre-stratificazione per "POD" (50%), in modo da essere sicuri di avere almeno un campione per POD (in versione 2.L sostituita con groupby POD)
        #possibile pre-stratificazione per "giorno della settimana"(1/7: 14,3%), campo da aggiungere al DB in fase di preprocessing
        print("INFO iniziali su 'df...stratified':")
        dfTOTinputAE_stratified0 = dfTOTinputAE.groupby('cap', group_keys=False).apply(
            lambda x: x.loc[:, dfTOTinputAE.columns].sample(frac=0.5))


        dfTOTinputAE_stratified1 = dfTOTinputAE_stratified0.groupby(
            'comune', group_keys=False).apply(lambda x: x.loc[:, dfTOTinputAE.columns].sample(frac=0.5))
        dfTOTinputAE_stratified2 = dfTOTinputAE_stratified1.groupby(
            'siglaprovincia', group_keys=False).apply(lambda x: x.loc[:, dfTOTinputAE.columns].sample(frac=0.5))
        dfTOTinputAE_stratified3 = dfTOTinputAE_stratified2.groupby(
            'Area_GME', group_keys=False).apply(lambda x: x.loc[:, dfTOTinputAE.columns].sample(frac=0.6))

        # dfTOTinputAE_stratified0 = dfTOTinputAE.groupby('cap', group_keys=False).apply(lambda x: x.sample(frac=0.5))
        dfTOTinputAE_stratified0.info()
        # dfTOTinputAE_stratified1 = dfTOTinputAE_stratified0.groupby('comune', group_keys=False).apply(lambda x: x.sample(frac=0.5))
        dfTOTinputAE_stratified1.info()
        # dfTOTinputAE_stratified2 = dfTOTinputAE_stratified1.groupby('siglaprovincia', group_keys=False).apply(lambda x: x.sample(frac=0.5))
        dfTOTinputAE_stratified2.info()
        # dfTOTinputAE_stratified3 = dfTOTinputAE_stratified2.groupby('Area_GME', group_keys=False).apply(lambda x: x.sample(frac=0.6))
        dfTOTinputAE_stratified3.info()
        dfTOTinputAE=dfTOTinputAE_stratified3
        print('dfTOTinputAE:')
        print("INFO iniziali su 'dfTOTinputAE':")
        dfTOTinputAE.info()
        print(dfTOTinputAE.head(10))
        ########################
        ########################
        ########################
        #NB: sovrascrittura del dataframe di input iniziale( per motivi di riutilizzo funzioni)... il DF iniziale è stato salvato con nome diverso (per riutilizzarlo, bisognerà riapplicare lo standardscaler)
        ########################
        ########################
        ########################
        dfTOTinputAE.to_csv(
            './output/curve_GEN-FEB_2024_AE_input_CSV_Aree_GME_4strat.csv')

        # dfTOTinputAE_stratified = dfTOTinputAE.groupby('Area_GME', group_keys=False).apply(lambda x: x.sample(frac=0.1))
        # dfTOTinputAE_stratified = dfTOTinputAE_stratified.groupby('siglaprovincia', group_keys=False).apply(
        #     lambda x: x.sample(frac=0.1))
        # dfTOTinputAE = dfTOTinputAE_stratified
        # dfTOTinputAE.to_csv(r'GEN-FEB-24_AE_input_CSV_Aree_GME_2strat.csv')


    ##############################################
    # ESTRAZIONE CAMPIONAMENTO DATI PER OGNI FONTE (esce un DF... da trasformare in array di float per il detrend e la normalizzazione (x ora nonnecessario)
    if(1): #attivare
        signals = pd.DataFrame(dfTOTinputAE.iloc[:,7:31])
        print('SIGNALS:')
        print(signals.head(10))
        #RINOMINA LABELS A01-24=>1-24  AUTOMATIZZARE!!! DA FARE DOPO SE NECESSARIO COSTITUIRE FEATURES 2D PRIMA DELLA CLUSTERIZZAZIONE (ALTRIMENTI E' SUFFICIENTE LA FEATURE SELECTION
        signals = signals.rename(columns={"A01": 0, "A02": 1, "A03": 2, "A04": 3, "A05": 4, "A06": 5, "A07": 6, "A08": 7, "A09": 8, "A10": 9, "A11": 10, "A12": 11, "A13": 12, "A14": 13, "A15": 14, "A16": 15, "A17": 16, "A18": 17, "A19": 18, "A20": 19, "A21": 20, "A22": 21, "A23": 22, "A24": 23}) #STATICO
        # for index in range(24):
        #     #print("A0"+str(index+1))
        #     signals2=signals.rename(columns={"A0"+str(index+1): index})
        print(signals.head(10))
        ###############################################

    ######################################################################################################
    # ESTRAZIONE FEATURES GENERALI ARBITRARIE DA OGNI FONTE (poi, eventualmente, saranno aggregate per singolo POD => fatto preliminarmente nella verione 2.L del giugno '24)
    if(1): #attivare
        df = pd.DataFrame(dfTOTinputAE.iloc[:,0:5])
        df['AF1'] = dfTOTinputAE['AF1']
        df['AF2'] = dfTOTinputAE['AF2']
        df['AF3'] = dfTOTinputAE['AF3']
        df['A01'] = dfTOTinputAE['A01'] #eventualmente automatizzare A01-25
        df['A02'] = dfTOTinputAE['A02']
        df['A03'] = dfTOTinputAE['A03']
        df['A04'] = dfTOTinputAE['A04']
        df['A05'] = dfTOTinputAE['A05']
        df['A06'] = dfTOTinputAE['A06']
        df['A07'] = dfTOTinputAE['A07']
        df['A08'] = dfTOTinputAE['A08']
        df['A09'] = dfTOTinputAE['A09']
        df['A10'] = dfTOTinputAE['A10']
        df['A11'] = dfTOTinputAE['A11']
        df['A12'] = dfTOTinputAE['A12']
        df['A13'] = dfTOTinputAE['A13']
        df['A14'] = dfTOTinputAE['A14']
        df['A15'] = dfTOTinputAE['A15']
        df['A16'] = dfTOTinputAE['A16']
        df['A17'] = dfTOTinputAE['A17']
        df['A18'] = dfTOTinputAE['A18']
        df['A19'] = dfTOTinputAE['A19']
        df['A20'] = dfTOTinputAE['A20']
        df['A21'] = dfTOTinputAE['A21']
        df['A22'] = dfTOTinputAE['A22']
        df['A23'] = dfTOTinputAE['A23']
        df['A24'] = dfTOTinputAE['A24']
        #df['A25'] = dfTOTinputAE['A25']
        print('DF:')
        print("INFO iniziali su 'df':")
        df.info()
        print(df.head(10))
        ######################################################################################################
    if(1):
        #ripetizione dell'estrazione features applicata al dataframe iniziale (prima della stratificazione)... appena possibile sviluppare funzione automatica
        df_iniziale = pd.DataFrame(Data_Frame_iniziale_GME.iloc[:, 0:5])
        df_iniziale['AF1'] = Data_Frame_iniziale_GME['AF1']
        df_iniziale['AF2'] = Data_Frame_iniziale_GME['AF2']
        df_iniziale['AF3'] = Data_Frame_iniziale_GME['AF3']
        df_iniziale['A01'] = Data_Frame_iniziale_GME['A01']  # eventualmenteautomatizzare A01-25
        df_iniziale['A02'] = Data_Frame_iniziale_GME['A02']
        df_iniziale['A03'] = Data_Frame_iniziale_GME['A03']
        df_iniziale['A04'] = Data_Frame_iniziale_GME['A04']
        df_iniziale['A05'] = Data_Frame_iniziale_GME['A05']
        df_iniziale['A06'] = Data_Frame_iniziale_GME['A06']
        df_iniziale['A07'] = Data_Frame_iniziale_GME['A07']
        df_iniziale['A08'] = Data_Frame_iniziale_GME['A08']
        df_iniziale['A09'] = Data_Frame_iniziale_GME['A09']
        df_iniziale['A10'] = Data_Frame_iniziale_GME['A10']
        df_iniziale['A11'] = Data_Frame_iniziale_GME['A11']
        df_iniziale['A12'] = Data_Frame_iniziale_GME['A12']
        df_iniziale['A13'] = Data_Frame_iniziale_GME['A13']
        df_iniziale['A14'] = Data_Frame_iniziale_GME['A14']
        df_iniziale['A15'] = Data_Frame_iniziale_GME['A15']
        df_iniziale['A16'] = Data_Frame_iniziale_GME['A16']
        df_iniziale['A17'] = Data_Frame_iniziale_GME['A17']
        df_iniziale['A18'] = Data_Frame_iniziale_GME['A18']
        df_iniziale['A19'] = Data_Frame_iniziale_GME['A19']
        df_iniziale['A20'] = Data_Frame_iniziale_GME['A20']
        df_iniziale['A21'] = Data_Frame_iniziale_GME['A21']
        df_iniziale['A22'] = Data_Frame_iniziale_GME['A22']
        df_iniziale['A23'] = Data_Frame_iniziale_GME['A23']
        df_iniziale['A24'] = Data_Frame_iniziale_GME['A24']
        #df_iniziale['A25'] = Data_Frame_iniziale_GME['A25']
        print("INFO iniziali su 'df_iniziale':")
        df_iniziale.info()

    if (1):
        # ripetizione dell'estrazione features applicata al dataframe iniziale (prima della stratificazione)... appena possibile sviluppare funzione automatica
        df_iniziale_pod = pd.DataFrame(dfTOTinputAE_POD.iloc[:, 0:5])
        df_iniziale_pod['AF1'] = dfTOTinputAE_POD['AF1']
        df_iniziale_pod['AF2'] = dfTOTinputAE_POD['AF2']
        df_iniziale_pod['AF3'] = dfTOTinputAE_POD['AF3']
        df_iniziale_pod['A01'] = dfTOTinputAE_POD['A01']  # eventualmenteautomatizzare A01-25
        df_iniziale_pod['A02'] = dfTOTinputAE_POD['A02']
        df_iniziale_pod['A03'] = dfTOTinputAE_POD['A03']
        df_iniziale_pod['A04'] = dfTOTinputAE_POD['A04']
        df_iniziale_pod['A05'] = dfTOTinputAE_POD['A05']
        df_iniziale_pod['A06'] = dfTOTinputAE_POD['A06']
        df_iniziale_pod['A07'] = dfTOTinputAE_POD['A07']
        df_iniziale_pod['A08'] = dfTOTinputAE_POD['A08']
        df_iniziale_pod['A09'] = dfTOTinputAE_POD['A09']
        df_iniziale_pod['A10'] = dfTOTinputAE_POD['A10']
        df_iniziale_pod['A11'] = dfTOTinputAE_POD['A11']
        df_iniziale_pod['A12'] = dfTOTinputAE_POD['A12']
        df_iniziale_pod['A13'] = dfTOTinputAE_POD['A13']
        df_iniziale_pod['A14'] = dfTOTinputAE_POD['A14']
        df_iniziale_pod['A15'] = dfTOTinputAE_POD['A15']
        df_iniziale_pod['A16'] = dfTOTinputAE_POD['A16']
        df_iniziale_pod['A17'] = dfTOTinputAE_POD['A17']
        df_iniziale_pod['A18'] = dfTOTinputAE_POD['A18']
        df_iniziale_pod['A19'] = dfTOTinputAE_POD['A19']
        df_iniziale_pod['A20'] = dfTOTinputAE_POD['A20']
        df_iniziale_pod['A21'] = dfTOTinputAE_POD['A21']
        df_iniziale_pod['A22'] = dfTOTinputAE_POD['A22']
        df_iniziale_pod['A23'] = dfTOTinputAE_POD['A23']
        df_iniziale_pod['A24'] = dfTOTinputAE_POD['A24']
        # df_iniziale_pod['A25'] = dfTOTinputAE_POD['A25']
        print("INFO iniziali su 'df_iniziale_pod':")
        df_iniziale_pod.info()

        # Tutti i segnali sono acquisiti a cadenza oraria (AF a cadenza 8h?); (sampling rate = 3600, settato a mano)
        sampling_rate = 3600  # in base al sistema di acquisizione e alla frequenza di campionamento
        # N.B.: unità di misura: s; spazio temporale tra campioni consecutivi

        ####################################################
        #TEST OUTPUT IN CSV
        test_df=df.head(100)
        test_signals=signals.head(100)
        test_df.to_csv('./output/TEST_AE_DF_output_CSV.csv')
        test_signals.to_csv('./output/TEST_AE_SIGNALS_output_CSV.csv')
        #####################################################


        # Load data FROM CSV FILE TO DATAFRAME - FINE
        ################################################



        print("INFO su signals")
        signals.info()
        df.info()
        signals.head(10)

        # Preprocessing dei segnali (detrending e normalizzazione)
        signals_detrended, signals_normalized = preprocessing(np.array(signals))
        #passaggio qui non necessario e non applicato nè per AHC nè per Classificazione; si tiene solo per mantenere la struttura generale del SW

        # Estrazione delle features
        features = feature_extraction(signals_detrended, signals_normalized, df, sampling_rate, opt_features)
        #features.to_csv(r'features.csv')
        print(features.head(10))
        # ripetizione dell'estrazione delle features applicata al dataframe iniziale
        features_iniziali = feature_extraction(signals_detrended, signals_normalized, df_iniziale, sampling_rate, opt_features)
        #features_iniziali.to_csv(r'features_iniziali.csv')
        # ripetizione dell'estrazione delle features applicata al dataframe iniziale
        features_iniziali_pod = feature_extraction(signals_detrended, signals_normalized, df_iniziale_pod, sampling_rate, opt_features)
        # features_iniziali.to_csv(r'features_iniziali_pod.csv')

        # Selezione delle features (in base all'opzione impostata in __main__
        if feature_set_selection == 1:
            features = pd.DataFrame(features,columns=['AF1', 'AF2', 'AF3'])
        if feature_set_selection == 2:
            features = pd.DataFrame(features,columns=['AF1', 'AF2', 'AF3', 'A01'])
        if feature_set_selection == 3:
            features = pd.DataFrame(features,columns=['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24'])
        if feature_set_selection == 4:
            features = pd.DataFrame(features,columns=['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'AF1', 'AF2', 'AF3'])
        print(features.head(10))
        #ripetizione della selezione feature al dataframe iniziale
        if feature_set_selection == 1:
            features_iniziali = pd.DataFrame(features_iniziali, columns=['AF1', 'AF2', 'AF3'])
        if feature_set_selection == 2:
            features_iniziali = pd.DataFrame(features_iniziali, columns=['AF1', 'AF2', 'AF3', 'A01'])
        if feature_set_selection == 3:
            features_iniziali = pd.DataFrame(features_iniziali,columns=['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10','A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20','A21', 'A22', 'A23', 'A24'])
        if feature_set_selection == 4:
            features_iniziali = pd.DataFrame(features_iniziali,columns=['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10','A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20','A21', 'A22', 'A23', 'A24', 'AF1', 'AF2', 'AF3'])
        # ripetizione della selezione feature al dataframe iniziale aggregato per POD
        if feature_set_selection == 1:
            features_iniziali_pod = pd.DataFrame(features_iniziali_pod, columns=['AF1', 'AF2', 'AF3'])
        if feature_set_selection == 2:
            features_iniziali_pod = pd.DataFrame(features_iniziali_pod, columns=['AF1', 'AF2', 'AF3', 'A01'])
        if feature_set_selection == 3:
            features_iniziali_pod = pd.DataFrame(features_iniziali_pod, columns=['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09','A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18','A19', 'A20', 'A21', 'A22', 'A23', 'A24'])
        if feature_set_selection == 4:
            features_iniziali_pod = pd.DataFrame(features_iniziali_pod, columns=['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09','A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18','A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'AF1', 'AF2','AF3'])

        #conversione DF features da string a float
        features[:] = features[:].astype('float64')
        #ripetizione conversione per DF iniziale
        features_iniziali[:] = features_iniziali[:].astype('float64')
        # ripetizione conversione per DF iniziale aggregato per POD
        features_iniziali_pod[:] = features_iniziali_pod[:].astype('float64')

        #ESPORTAZIONE FEATURES IN PICKLE (per utilizzo in "main_clustering_kmeans")
        with open('./output/features', 'wb') as file:
            pickle.dump(features, file, protocol=pickle.HIGHEST_PROTOCOL)

        #ripetizione ESPORTAZIONE FEATURES IN PICKLE (per utilizzo in fase classificazione K_nearest)
        with open('./output/features_iniziali', 'wb') as fileB:
            pickle.dump(features_iniziali, fileB, protocol=pickle.HIGHEST_PROTOCOL)

        # ripetizione ESPORTAZIONE FEATURES IN PICKLE (per utilizzo in fase classificazione K_nearest)
        with open('./output/features_iniziali_pod', 'wb') as fileC:
            pickle.dump(features_iniziali_pod, fileC, protocol=pickle.HIGHEST_PROTOCOL)

        # # Representation (single features)
        # plt.hist(features['AF1'], bins=10)
        # plt.show()
        # plt.hist(features['AF2'], bins=10)
        # plt.show()
        # plt.hist(features['AF3'], bins=10)
        # plt.show()

        a = 0

        # # #CREAZIONE DATAFRAME COMPLETO CON LABELS (non funziona se prima non è stato creato labels_df.csv... spostare blocco a fine calcoli; scorporare il numero POD) - fatto in AHC
        # data_df_input = pd.read_csv('gennaio2023_AE_input_CSV.csv', sep=';')
        # label_df_input = pd.read_csv('labels_df.csv', sep=';')
        # print(data_df_input.head(10))
        # output_labeled_df = pd.DataFrame(data_df_input)
        # output_labeled_df['LABEL'] = label_df_input['0']
        # print(output_labeled_df.head(100))
        # output_labeled_df.to_csv(r'output_labeled_df.csv', sep=';')

# BLOCCO 1 - FINE -  ESTRAZIONE FEATURES (Preprocessing: detrend e normalizzazione; Features selection: tramite pannello di controllo con dizionario variabile di features)
#####################################################################################################################################################################






###############################
# BLOCCO 2 - CLUSTERING K-MEANS
###############################
def main_clustering_kmeans(): # Applicazione Kmeans a features selezionate e salvate in formato pickle

    with open('./output/features', 'rb') as file:
        features1 = pickle.load(file)

                # FUNCTION SWITCH: set manual parameters (Kmeans)
        # struttura if-else per
        # seleziona mono o multi dimensionale (se presente): codici ^mono^ o ^multi^
        # in feature monodimensionali settare "feature selected"
        # in else, selezionare 'x' o 'y' per "feature selected"

    sel = "mono"

    if sel == "mono":
        # # Per feature monodimensionali
        # scegliere tra 'max_amplitude', 'standard_dev', 'phase', 'power' , 'A01',ecc....
        features_selected = features1

        feature_set = []  # Compilia lista vuota (come lista di liste) a partire da dataframe
        num_sens = 1
        for feat in features_selected:
             for sens in range(num_sens):
                 feature_set.append(eval('features1' + '[\'' + feat + '\']'))
        feature_set = np.transpose(np.array(feature_set))  # Trasformo la lista di liste in array
    else:
        # Per feature multidimensionali (se presenti)
        # scegliere tra features_selected = ['x', 'y'...ecc ]
        features_selected = 'x'
        feature_set = []  # Compiliamo lista vuota (come lista di liste) a partire da dataframe
                #        feature_set = np.array(feature_set)  # Trasformo la lista di liste in array
        # fine if-else

    feature_set.tofile('feature_set_INPUT_KMEANS.csv', sep=',', format='%10.5f')

      # STANDARDIZZAZIONE (dopo aver selezionato le features: standardizzo solo la parte di dataset utilizzata)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_set)

    scaled_features.tofile('scaled_features_INPUT_KMEANS.csv', sep=',', format='%10.5f')

    if(1):    #APPLICAZIONE KMEANS CON X CLUSTERS
        # Versione base (per rappresentazione grafica ed estrazione labels- numero di cluster manuale dopo averli
        # eventualmente indagati con la "versione iterativa" ... da fare)
        num_clust = 4
        lunghezza_DB=len(scaled_features)
        massimo_divisore_intero_DB= 10


        kmeans = KMeans(init="k-means++", n_clusters=num_clust, n_init=10, max_iter=10000, random_state=22)
        kmeans.fit(scaled_features)
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
        # 22 seme casuale

    if (0):  # TEST APPLICAZIONE KMEANS CON 100 CLUSTERS - disabilitato poichè superato dalclustering gerarchico
        # Versione base (per rappresentazione grafica ed estrazione labels- numero di cluster manuale dopo averli
        # eventualmente indagati con la "versione iterativa" ... da fare)
        num_clust = 100
        kmeans = KMeans(init="k-means++", n_clusters=num_clust, n_init=10, max_iter=10000, random_state=22)
        kmeans.fit(scaled_features)
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
        # 22 seme casuale



    # # A list holds the silhouette coefficients for each k (DISABILITATO POICHE' MOLTO LENTO E NON RICHIESTO)
    if(0):
        silhouette_coefficients = []
        # Notice you start at 2 clusters for silhouette coefficient
        for k in range(2, 6):
            print(k)
            kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
            print("a")
            kmeans.fit(scaled_features)
            print("b")
            score = silhouette_score(scaled_features, kmeans.labels_)
            print("c")
            silhouette_coefficients._append(score)
            print("d")
        # Rappresentazione silhouette score
        plt.style.use("fivethirtyeight")
        plt.plot(range(2, 6), silhouette_coefficients)
        plt.xticks(range(2, 6))
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Coefficient")
        plt.title('Silhouette coefficient - k-means applicato su fourier')
        plt.show()

    # FINE Versione iterativa

    # # individuazione numero ottimale clusters con metodo calinski_harabasz  (DISABILITATO MA FUNZIONANTE: GIA' USATO e GRAFICO SALVATO. dice 8 clusters è il meglio ma noi scegliamo 4 per indicazione della committenza)
    if(0):
        model = KMeans()
        # k is range of number of clusters.
        visualizer = KElbowVisualizer(model, k=(2, 20), metric='calinski_harabasz', timings=False)
        visualizer.fit(scaled_features)  # Fit the data to the visualizer
        plt.xticks(range(2, 20))
        visualizer.show()  # Finalize and render the figure


    # ESPORTAZIONE LABELS IN DATAFRAME E IN CSV
    labels_df = pd.DataFrame(labels)
    print(labels_df.head(100))
    # print(labels_df.info)
    #labels_df_label = labels_df.rename(columns={'1': 'LABEL'})
    # print(labels_df_label.info)
    # print(labels_df.head(100))
    labels_df.to_csv('./output/labels_df_KMEANS_' +
                     str(num_clust)+'.csv', sep=';')


    # PLOT LABELS (disabilitato ma funzionante)
    if(0):
        plt.hist(labels_df[0], bins=9)
        plt.show()

    # PLOT CLUSTERING

    # TEST PLOT 3D LABELS DELLE FASCE AF# (disabilitato ma funzionante se il csv è completo; per ora è completato a mano)
    if(0):
        data_df_3dx = pd.read_csv('output_labels_df.csv', sep=';')
        data_df_3dx = data_df_3dx.replace(',', '.', regex=True)
        data_df_3dx['AF1'] = data_df_3dx['AF1'].astype(float)
        data_df_3dx['AF2'] = data_df_3dx['AF2'].astype(float)
        data_df_3dx['AF3'] = data_df_3dx['AF3'].astype(float)
        data_df_3dx['LABEL'] = data_df_3dx['LABEL'].astype(float)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_title('Labels 3D Scatter Plot')
        ax.set_xlabel('AF1')
        ax.set_ylabel('AF2')
        ax.set_zlabel('AF3')

        ax.view_init(elev=12, azim=40)  # elevation and angle
        ax.dist = 10  # distance
        ax.scatter(
            data_df_3dx['AF1'], data_df_3dx['AF2'], data_df_3dx['AF3'],  # data
            c=data_df_3dx['LABEL'],                            # marker colour
            # marker='o',                                # marker shape
            s=60  # marker size
        )

        plt.show()


    ###########################################
    # INVIO FILE LABELS_DF.CSV a PIATTAFORMA TP (momentaneamente disabilitato ma funzionante)
    ###########################################
    # # lìid della track verrà dato in input al lancio dello script
    # if(0): 
    if(1):
        dest_track_id = 5104  # id track 5104 -> KMEANS9-Result  (5105  Grap-Result)

        wcf_url = "http://sdm.demat.develop.technoplants.lan/EngineAssets.svc?wsdl"
        client = Client(wcf_url)

        # carica le info della track padre
        trackInfo_file = client.service.GetTrackRawDataInfoByTrack(
            5104)  # file 1 -> id track 5104 -> KMEANS9-Result  (5105  Grap-Result)

        print(trackInfo_file)

        # se già presente elimina la raw con il precedendente file di risultato
        if trackInfo_file != None: client.service.StorageDeleteTrackRawData(trackInfo_file.RawId)

        # esempio risultati
        # data = {'col1': [11, 12, 13, 14],
        #         'col2': ['aA', 'bB', 'cC', 'dD']}
        # df = pd.DataFrame(data)

        output = io.BytesIO()

        csv_file_name = 'LABELS_4.csv'
        Mime = "application/octet-stream"

        # df.to_csv(output, index=False, sep=';')
        labels_df.to_csv(output, index=False, sep=';')

        byte_array = output.getvalue()
        base64_data = base64.b64encode(byte_array).decode('utf-8')

        response = ''
        try:

            response = client.service.UploadTrack(
                dest_track_id, csv_file_name, Mime, base64_data)

        except suds.WebFault as e:
            print(f"Errore durante il caricamento: {str(e)}")
        except Exception as e:
            print(f"Errore generico: {str(e)}")

        print(response)
    # #######################################################





    return labels_df

###############################
# BLOCCO 2 - CLUSTERING K-MEANS - FINE
###############################

###############################
# BLOCCO 3 - CLUSTERING AHC FASCE
###############################

def create_factors(n):
    for i in range(1, n+1):
        if n % i == 0:
            yield i


def main_clustering_AHC():
    with open('./output/features', 'rb') as file:
        features1 = pickle.load(file)
    # test estrazione in csv di features_4 e df_4... per verificare il codice univoco dell'evento e mapping row_ID
    features1.to_csv('./output/features_INPUT_AHC.csv')

    #apertura filedelle features del DF iniziale (ai fini della classificazione a valle di AHC applicato su campione stratificato
    with open('./output/features_iniziali', 'rb') as fileB:
        features2 = pickle.load(fileB)
    features2.to_csv('./output/features_inziali_INPUT_AHC.csv')

    # apertura filedelle features del DF iniziale (ai fini della classificazione a valle di AHC applicato su campione stratificato
    with open('./output/features_iniziali_pod', 'rb') as fileC:
        features3 = pickle.load(fileC)
    features3.to_csv('./output/features_iniziali_pod_INPUT_AHC.csv')

    #salvataggio features del DF iniziale (da usare in fase classificazione K-nearest (disabilitato per velocizzare: NON RIMUOVERE)
    #features2.to_csv(r'features_iniziali_INPUT_K-nearest.csv')

# FUNCTION SWITCH: set manual parameters (Kmeans)
        # struttura if-else per
        # seleziona mono o multi dimensionale (se presente): codici ^mono^ o ^multi^
        # in feature monodimensionali settare "feature selected"
        # in else, selezionare 'x' o 'y' per "feature selected"

    sel = "mono"

    if sel == "mono":
        # # Per feature monodimensionali
        # scegliere tra 'max_amplitude', 'standard_dev', 'phase', 'power' , 'A01',ecc....

        #OSS: Automatizzare con feature set selection 1-2-3-4
        features_selected = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24','AF1','AF2','AF3']
        features_selected_iniziale = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13','A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24']
        features_selected_iniziale_pod = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11','A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22','A23', 'A24']
        #features_selected = ['AF1','AF2','AF3']

        feature_set = []  # Compilia lista vuota (come lista di liste) a partire da dataframe
        feature_set_iniziale = []  # Compilia lista vuota (come lista di liste) a partire da dataframe
        feature_set_iniziale_pod = []  # Compilia lista vuota (come lista di liste) a partire da dataframe
        num_sens = 1

        if(0):
            for feat in features_selected:
                 for sens in range(num_sens):
                    #feature_set.append(eval('features1' + '[\'' + feat + '\']')) #numpy.core._exceptions._ArrayMemoryError: Unable to allocate 319. GiB for an array with shape (42864309615,) and data type float64 - problemi computazionali verificati per numerosità maggiori di nx10k record => necessario approccio statistico (approvato Prof. Zich)
                    #feature_set.append(eval('features1' + '[\'' + feat + '\']' + '.head(100)'))
                    feature_set.append(eval('features1' + '[\'' + feat + '\']' + '.sample(n=10000,replace=False,random_state=1)')) #funzionante per campionamento random non stratificato su DB non modificato (cioè senza inserire Aree_GME)
        if(1):
            for feat in features_selected:
                for sens in range(num_sens):
                    feature_set.append(eval('features1' + '[\'' + feat + '\']')) #applicato al dataframe contenente le Aree_GME stratificato

        #ripeto su DF iniziale (solo 24 features omogenee - cioè le registrazioni orarie, comunque successivamente standardizzate)
        if(1):
            for feat in features_selected_iniziale:
                for sens in range(num_sens):
                    feature_set_iniziale.append(eval('features2' + '[\'' + feat + '\']'))  # applicato al dataframe iniziale

        # ripeto su DF iniziale AGGREGATO PER POD  (solo 24 features omogenee - cioè le registrazioni orarie, comunque successivamente standardizzate)
        if (1):
            for feat in features_selected_iniziale_pod:
                for sens in range(num_sens):
                    feature_set_iniziale_pod.append(eval('features3' + '[\'' + feat + '\']'))  # applicato al dataframe iniziale aggregato POD

        feature_set = np.transpose(np.array(feature_set))  # Trasformo la lista di liste in array
        #feature_set = np.transpose(np.array(features1))  # Trasformo direttamente il csv in lista di array
        #feature_set=np.array(features1)

        #ripeto x DF iniziale
        feature_set_iniziale = np.transpose(np.array(feature_set_iniziale))  # Trasformo la lista di liste in array

        # ripeto x DF iniziale aggregato POD
        feature_set_iniziale_pod = np.transpose(np.array(feature_set_iniziale_pod))  # Trasformo la lista di liste in array

    else:
        # Per feature multidimensionali (se presenti)
        # scegliere tra features_selected = ['x', 'y'...ecc ]
        features_selected = 'x'
        feature_set = []  # Compiliamo lista vuota (come lista di liste) a partire da dataframe
        feature_set_iniziale = []  # ripeto: Compiliamo lista vuota (come lista di liste) a partire da dataframe
        feature_set_iniziale_pod = []  # ripeto: Compiliamo lista vuota (come lista di liste) a partire da dataframe (iniziale)
                #        feature_set = np.array(feature_set)  # Trasformo la lista di liste in array
        # fine if-else
    #stampa di test (NON RIMUOVERE)
    #feature_set.tofile('feature_set_INPUT_AHC.csv', sep=',', format='%10.5f')
    #ripeto
    #feature_set_iniziale.tofile('feature_set_iniziale_INPUT_K-nearest.csv', sep=',', format='%10.5f')

      #STANDARDIZZAZIONE (dopo aver selezionato le features: standardizzo solo la parte di dataset utilizzata) - possibile in questo caso essendo le features tutte misurate nella stessa unità di misura.
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_set)
    #ripeto x DF iniziale
    scaled_features_iniziali = scaler.fit_transform(feature_set_iniziale)
    #ripeto x DF aggregato per POD
    scaled_features_iniziali_pod = scaler.fit_transform(feature_set_iniziale_pod)

    #stampa di test (NON RIMUOVERE)
    #scaled_features.tofile('scaled_features_INPUT_AHC.csv', sep=',', format='%10.5f')
    #scaled_features_iniziali.tofile('scaled_features_iniziali_INPUT_K-nearest.csv', sep=',', format='%10.5f')
    #scaled_features_iniziali_pod.tofile('scaled_features_iniziali_pod_INPUT_K-nearest.csv', sep=',', format='%10.5f')


    ###########
    # sotto blocco TD-HC CLUSTERING GERARCHICO DIVISIVO - più veloce (segue n e non n^2) - FUNZIONA (manca export) - molto simile a K-MEANS
    ###########

    if(0): #METODO SUPERATO
        #clustered_class = DePDDP(max_clusters_number=8).fit_predict(scaled_features)
        #print(clustered_class)
        num_clust_HC=3


        #### CALCOLO DELLA DIMENSIONE OPPORTUNA DELLO SPLIT
        #CALCOLO DIVISORI

        n=len(scaled_features)

        # Create the generator object
        factors_gen = create_factors(n)
        df_divisori = pd.DataFrame(factors_gen, columns=['divisore']) #creo df vuoto

        # Generate and print all factors and save in dataframe
        print("Factors of", n)
        for factor in factors_gen:
            df_divisori[factor]=factor_gen[factor]
            print(factor, end=", ")

        print(df_divisori.head(20))

        #SCELTA DEL DIVISORE tale da dividere il DB in porzioni inferiori al tot% (20%)

        split_number=df_divisori['divisore'].iloc[6]
        print(split_number)

        #seleziono divisori >n/10 BOOLEAN
        df_divisori_grandi=pd.DataFrame(df_divisori['divisore']>n/20)
        print(df_divisori_grandi.head(10))

        #subset dataframe con divisori >n/10
        df_divisori_grandi=df_divisori[df_divisori['divisore']>n/20]
        print(df_divisori_grandi.head(10))

        #seleziono il minimo
        split_number=int(df_divisori_grandi['divisore'].min())

        print('splitnumber: ' )
        print(split_number)
        ###fare





        #clustered_class = DePDDP(max_clusters_number=8, bandwidth_scale=0.5, percentile=0.1, min_sample_split=1) #input originali
        clustered_class = DePDDP(max_clusters_number=num_clust_HC, bandwidth_scale=0.1, percentile=0.3, min_sample_split=split_number,visualization_utility=True) #input funzionanti con 10k campioni e grafici e anche con 290k un po' lento e i disegni si bloccano
        #NB: sembrache utilizzando un parametro min_sample_split meno restrittivo -per esempio 10k - riesca a clusterizzare bene; scegliere un sottomultiplo della numerosità
        #clustered_class = DePDDP(max_clusters_number=8, bandwidth_scale=0.1, percentile=0.3, min_sample_split=1000, visualization_utility=True) #input test

                            # decomposition_method (str, (optional)) – One of the (‘pca’, ‘kpca’, ‘ica’, ‘tsne’)
        #clustered_class.fit_predict(scaled_features)

        # ESPORTAZIONE LABELS_TDHC IN DATAFRAME E IN CSV
        clu_res = pd.DataFrame(clustered_class.fit_predict(scaled_features)) #array n-dimensionale


        #?
        # labels_TDHC = DePDDP.labels_
        # labels_df_HC=labels_TDHC
        # labels_df_TDHC = pd.DataFrame(labels_TDHC)
        #?

        print(clu_res.head(100))
        clu_res.to_csv('./output/labels_df_TDHC_'+str(num_clust_HC)+'.csv', sep=';')



        #### INSERIMENTO DELLE LABELS TDHC NEL CAMPIONE STRATIFICATO - rifare
        # dfTOTinputAE_stratified["label_TDHC"]=clu_res['0']
        # dfTOTinputAE_stratified_labeled=dfTOTinputAE_stratified
        # dfTOTinputAE_stratified_labeled.to_csv(r'DF_stratified_labeled_TDHC_'+str(num_clust_HC)+'.csv', sep=';')



    if (0): #DENDROGRAMMI DEL TD-HC (FUNZIONA - NON RIMUOVERE)
        m = viz.dendrogram_visualization(clustered_class, cmap='viridis', default_coloring=True)
        plt.show()

    if (0):
        viz.split_visualization(clustered_class, color_map='viridis')
        plt.show()

    #####
    # sottoblocco TD-HC - fine
    #####

    ###########
    # sotto blocco AHC CLUSTERING GERARCHICO AGGLOMERATIVO - più lento (segue n^2) - FUNZIONA
    ###########

    if(1): #tempo computazionale elevato oltre i 1000 campioni
        #scelta metodo di linkage: ward (migliore), single, complete, average, weighted, centroid, median
        ahc_method='ward'
        ahc_metric= 'euclidean' #oppure 'cityblock' (x Manhattan) o altre

        z = sch.linkage(scaled_features, method=ahc_method, metric=ahc_metric,  optimal_ordering=True) #metric='euclidean', (OSS:Method 'ward' requires the distance metric to be Euclidean)
        # Test sulla linkage matrix: nelle prime due colonne, il massimo numero che appare è:
        # (((numero_campioni * 2 - 1) - 1) - 1)

        # # Dendrogram without threshold
        # dendrogram_plot = sch.dendrogram(z)
        # plt.xticks([])
        # plt.show()

        # # Distanza euclidea tra campioni
        # d = []
        # for sig1 in scaled_features:
        #     for sig2 in scaled_features:
        #         d.append(distance.euclidean(sig1, sig2))
        # d = np.reshape(np.array(d), (len(scaled_features), len(scaled_features)))
        # Cambio forma per esigenze di visualizzazione
        # max_fs_distance = np.max(d)

        # # Distanza copenetica tra campioni: usa la posizione risultante dall'aggregazione precedente
        coph = sch.cophenet(z)
        coph_max = np.max(coph)
        # Rappresentazione alternativa in forma quadrata
        # coph_sq = distance.squareform(coph)

        ########## DENDROGRAM MANUAL THRESHOLD ##########

        threshold_percentage = 0.45
        thres_perc_str = format(threshold_percentage, ' .2f')  # conversione in stringa x denominazione automatica output
        threshold = coph_max * threshold_percentage


        if (0):  # FUNZIONA; disabilitato per velocizzare
            if feature_set_selection == 1:
                # # Dendrogram with manual threshold - FASCE ##################
                dendrogram_plot = sch.dendrogram(z)
                plt.xticks([])
                plt.axhline(y=threshold, color='k', linestyle='--', label='AHC Dendrogram - FASCE A1-A2-A3 - Manual threshold - '+ahc_method)
                plt.title("AHC Dendrogram - FASCE A1-A2-A3 - Manual threshold "+str(threshold_percentage)+' - '+ahc_method)
                plt.show()
                ######################################################
            if feature_set_selection == 2:
                # # Dendrogram with manual threshold FASCE A1-A2-A3 e A01 ##################
                dendrogram_plot = sch.dendrogram(z)
                plt.xticks([])
                plt.axhline(y=threshold, color='k', linestyle='--', label='AHC Dendrogram - FASCE A1-A2-A3 e A01 - Manual threshold - '+ahc_method)
                plt.title("AHC Dendrogram - FASCE A1-A2-A3 e A01 - Manual threshold "+str(threshold_percentage)+' - '+ahc_method)
                plt.show()
                ######################################################
            if feature_set_selection == 3:
                # # Dendrogram with manual threshold REGISTRAZIONI ORARIE A01-A24 ##################
                dendrogram_plot = sch.dendrogram(z)
                plt.xticks([])
                plt.axhline(y=threshold, color='k', linestyle='--', label='AHC Dendrogram - REGISTRAZIONI ORARIE A01-A24 - Manual threshold - '+ahc_method)
                plt.title("AHC Dendrogram - REGISTRAZIONI ORARIE A01-A24 - Manual threshold "+str(threshold_percentage)+' - '+ahc_method)
                plt.show()
                ######################################################
            if feature_set_selection == 4:
                # # Dendrogram with manual threshold REGISTRAZIONI ORARIE A01-A24 e FASCE ##################
                dendrogram_plot = sch.dendrogram(z)
                plt.xticks([])
                plt.axhline(y=threshold, color='k', linestyle='--', label='AHC Dendrogram - REGISTRAZIONI ORARIE A01-A24 e FASCE- Manual threshold - '+ahc_method)
                plt.title("AHC Dendrogram - REGISTRAZIONI ORARIE A01-A24 e FASCE- MManual threshold "+str(threshold_percentage)+' - '+ahc_method)
                plt.show()
                ######################################################

        #model = AgglomerativeClustering(distance_threshold=threshold, n_clusters=None, metric=ahc_metric, linkage=ahc_method) #funziona:utile per avere il taglio del dendrogramma ad una certa altezza
        model = AgglomerativeClustering(n_clusters=4, metric=ahc_metric, linkage=ahc_method) # funziona: utile per ottenere un predeterminato numero di clusters
        model.fit(scaled_features)
        labels_AHC = model.labels_  # fatto con soglia manuale x estrazione labels con SOGLIA MANUALE (impostando manualmente altezza dendrogramma o numero di clusters richiesto)
        # questo vettore andrà confrontato con le labels "reali":
        # salvare le label per ogni configurazione di parametri scelti per il run. NB: i run con fourier e autocorr
        # sono molto lenti su Surface7... fare run su altro pc
        # # Salvare in csv le labels:
        print(str(model.n_clusters_) + " verifica numero clusters calcolati per threshold manuale " + str(
            threshold_percentage))
        labels_AHC_df = pd.DataFrame(labels_AHC)
        labels_AHC_df.to_csv(
            './output/labels_AHC_df_MANUAL_4clusters_4strat_GEN-FEB-24.csv', sep=';')

        #EXPORT LABELED TRAIN DB


    # inserimento labels nel campione stratificato (non serve perché, per la successiva classificazione dell'intera popolazione, X è il DB iniziale e y è l'array delle labels e sono già separati)
    if(0):
        labels_AHC_df=pd.read_csv('labels_AHC_df_MANUAL_45_4strat_GEN-FEB-24.csv', sep=';') #poi rimuovere (già fatto prima e qui usato solo x saltare i calcoli precedenti) NB: senza questo passaggio su disco mi genera problemi di mappatura del DF... verificare...
        print(labels_AHC_df.head(10))
        dfTOTinputAE_4strat=pd.read_csv('curve_GEN-FEB_2024_AE_input_CSV_Aree_GME_4strat.csv') #già eseguito in fase AHC... riprendere il df già caricato x saltare il passggio su disco
        #dfTOTinputAE_4strat=dfTOTinputAE ... sistemare
        print(dfTOTinputAE_4strat.head(10))
        dfTOTinputAE_4strat_labeled_AHC = dfTOTinputAE_4strat
        #dfTOTinputAE_labeled_AHC['Labels_AHC'] = labels_AHC_df[.iloc[:,1]] #non va
        #dfTOTinputAE_4strat_labeled_AHC["Labels_AHC"] = labels_AHC_df["0"] #da fare se si vuole avere un CSV con labels... ma con milioni di righe non servirà
        dfTOTinputAE_4strat_labeled_AHC.to_csv('./output/curve_GEN-FEB-24_AE_input_CSV_Aree_GME_4strat_labeled_AHC.csv')





    if(0): # AGGLOMERATIVO GERARCHICO CON CALCOLO PRELIMINARE DELLA MATRICE DI LINKAGE (metodo alternativo funzionante ma superato)
        # scelta metodo di linkage: ward (migliore), single, complete, average, weighted, centroid, median
        ahc_method = 'ward'
        ahc_metric = 'euclidean'  # oppure 'cityblock' (x Manhattan) o altre
        z = sch.linkage(scaled_features, method=ahc_method, metric=ahc_metric,
                    optimal_ordering=True)  # metric='euclidean', (OSS:Method 'ward' requires the distance metric to be Euclidean)

        # #richiedo le labels impostando manualmente il numero di clusters; scelgo 8 perchè esce dal CH test sul kmeans
        k=8
        ahc_clusters = fcluster(z, k, criterion='maxclust')
        print("ahc_clusters")

        # # ESPORTAZIONE LABELS AHC IN DATAFRAME E IN CSV
        labels_df_AHC = pd.DataFrame(ahc_clusters)
        print(labels_df_AHC.head(1000))
        # print(labels_df.info)
        # labels_df_label = labels_df.rename(columns={'1': 'LABEL'})
        # print(labels_df_label.info)
        # print(labels_df.head(100))
        labels_df_AHC.to_csv('./output/labels_df_AHC_8.csv', sep=';')

        # plt.figure(figsize=(10, 8))
        # plt.scatter(scaled_features[:,1], scaled_features[:,2], c=ahc_clusters, cmap='prism')  # plot points with cluster dependent colors
        # plt.show()





    ######
    #sottoblocco CLASSIFICATION sull'intero DB a partire dal clustering AHC sui dati stratificati (usati come training set)
    ######

    #####valutazione prestazioni K-Nearest su campione stratificato (AUTOTEST)
    if(1):
        #estrazione campi numerici e trasformazione in type float
        #X = pd.DataFrame(columns=['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24','AF1','AF2','AF3'])
        X = pd.DataFrame(columns=['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14','A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24'])
        #X = pd.DataFrame(columns=['A01', 'A02', 'A03', 'A04', 'A05', 'A06'])
        #X['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24','AF1','AF2','AF3']=dfTOTinputAE_4strat['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24','AF1','AF2','AF3']
        #dfTOTinputAE_4strat_labeled_AHC['A01'] = dfTOTinputAE_4strat_labeled_AHC['A01'].str.replace(',', '.').astype(float)

        if(1): #serve per saltare  passaggi iniziali e recuperare le labels già calcolate con AHC (evita problema variabili globali)
            #import del campione stratificato e delle labels derivanti dal AHC
            labels_AHC_df = pd.read_csv('./output/labels_AHC_df_MANUAL_4clusters_4strat_GEN-FEB-24.csv', sep=';')  # poi rimuovere (già fatto prima e qui usato solo x saltare i calcoli precedenti) NB: senza questo passaggio su disco mi genera problemi di mappatura del DF... verificare...
            print(labels_AHC_df.head(10))
            dfTOTinputAE_4strat = pd.read_csv('./output/curve_GEN-FEB_2024_AE_input_CSV_Aree_GME_4strat.csv')  # già eseguito in fase AHC... riprendere il df già caricato x saltare il passggio su disco

           # esempio: X['A01'] = dfTOTinputAE_4strat['A01'].str.replace(',', '.').astype(float)  # automatizzare
        print('converti')
        X['A01']=  dfTOTinputAE_4strat['A01'] #automatizzare
        X['A02'] = dfTOTinputAE_4strat['A02']
        X['A03'] = dfTOTinputAE_4strat['A03']
        X['A04'] = dfTOTinputAE_4strat['A04']
        X['A05'] = dfTOTinputAE_4strat['A05']
        X['A06'] = dfTOTinputAE_4strat['A06']
        X['A07'] = dfTOTinputAE_4strat['A07']
        X['A08'] = dfTOTinputAE_4strat['A08']
        X['A09'] = dfTOTinputAE_4strat['A09']
        X['A10'] = dfTOTinputAE_4strat['A10']
        X['A11'] = dfTOTinputAE_4strat['A11']
        X['A12'] = dfTOTinputAE_4strat['A12']
        X['A13'] = dfTOTinputAE_4strat['A13']
        X['A14'] = dfTOTinputAE_4strat['A14']
        X['A15'] = dfTOTinputAE_4strat['A15']
        X['A16'] = dfTOTinputAE_4strat['A16']
        X['A17'] = dfTOTinputAE_4strat['A17']
        X['A18'] = dfTOTinputAE_4strat['A18']
        X['A19'] = dfTOTinputAE_4strat['A19']
        X['A20'] = dfTOTinputAE_4strat['A20']
        X['A21'] = dfTOTinputAE_4strat['A21']
        X['A22'] = dfTOTinputAE_4strat['A22']
        X['A23'] = dfTOTinputAE_4strat['A23']
        X['A24'] = dfTOTinputAE_4strat['A24']
        # X['AF1'] = dfTOTinputAE_4strat_labeled_AHC['AF1'].str.replace(',', '.').astype(float)
        # X['AF2'] = dfTOTinputAE_4strat_labeled_AHC['AF2'].str.replace(',', '.').astype(float)
        # X['AF3'] = dfTOTinputAE_4strat_labeled_AHC['AF3'].str.replace(',', '.').astype(float)

        print('fineconverti')

        #standardizzazione del DB stratificato (da rifare anche qui, poichè importato da CSV precedentemente esportato in fase AHC)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(X)


        scaled_features.tofile('./output/scaled_dfTOTinputAE_4strat.csv', sep=',', format='%10.5f')
        X = scaled_features

        print('finescaler')

        print(labels_AHC_df[["0"]].head(50))
        print(labels_AHC_df[["0"]].dtypes)
        labels_AHC_df = labels_AHC_df[["0"]].astype(float)
        labels_AHC_df['LABEL'] = labels_AHC_df['0'].astype(float)

        y = labels_AHC_df['0'].values
        print(labels_AHC_df.dtypes)
        print(labels_AHC_df['0'].values)
        print("y.ctypes:")
        print(y.ctypes)

        #data_df_3dx['LABEL'] = data_df_3dx['LABEL'].astype(float)

        print(X.shape, y.shape)

        # X = X.apply(pd.to_numeric, errors='coerce')
        # y = y.apply(pd.to_numeric, errors='coerce')

        #Sandro
        print(Counter(y))


        # Supponiamo che y sia un array numpy
        y = np.array(y)

        # Conta la frequenza delle classi
        counter = Counter(y)

        # Trova le classi con meno di 2 campioni
        rare_classes = [cls for cls, count in counter.items() if count < 2]

        # Filtra i dati per rimuovere queste classi
        # Crea una maschera per i campioni da mantenere
        mask = ~np.isin(y, rare_classes)
        X_filtered = X[mask]
        y_filtered = y[mask]
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y_filtered, test_size=0.2, random_state=24, stratify=y_filtered)
        #Sandro
        
        #classificazione K-nearest del train set x verifica prestazioni

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24, stratify=y)
        knn = KNeighborsClassifier(n_neighbors=2)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
       # print("Test set predictions:\\n {}\"".format(y_pred))
        print("knn score 2 neighbors")
        print(knn.score(X_test,y_test))



        #verifica accuratezza (AUTOTEST)

        train_accuracies = {}
        test_accuracies = {}
        neighbors = np.arange(1, 26)
        for neighbor in neighbors:
            knn = KNeighborsClassifier(n_neighbors=neighbor)
            knn.fit(X_train, y_train)
            train_accuracies[neighbor] = knn.score(X_train, y_train)
            print(train_accuracies[neighbor])
            test_accuracies[neighbor] = knn.score(X_test, y_test)
            print(test_accuracies[neighbor])

        if(0): #grafico verifica numero ottimo di nodi vicini da considerare in K_nearest
            plt.figure(figsize=(8, 6))
            plt.title("KNN: Varying Number of Neighbors")
            plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")
            plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy")
            plt.legend()
            plt.xlabel("Number of Neighbors")
            plt.ylabel("Accuracy")
            plt.show()


    ###########################################
    #applicazione k-nearest al dataset completo o aggregato per POD
    ###########################################

    # X_new = np.array([[56.8, 17.5],
    #                   [24.4, 24.1],
    #                   [50.1, 10.9]])

    #X_input= pd.DataFrame(columns=['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24','AF1','AF2','AF3'])
    X_input = pd.DataFrame(columns=['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14','A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24'])

    #ricarico il file di input iniziale da zero - NB: STANDARDIZZARE IL DB O USARE QUELLO GIA' MANIPOLATO (in memoria durante l'esecuzione completa)
    # Load data FROM CSV FILE TO DATAFRAME
    if (0):  #(DB COMPLETO) attivare per saltare la rielaborazione iniziale e utilizzare i csv intermedi salvati su disco
        # df1 = pd.read_csv('gennaio2023-1-2.csv', sep=';') # I DB PRECEDENTI USANO IL PUNTO E VIRGOLA COME SEPARATORE DEL CSV !!!!!!!!!!!!!!!!!!!!!!
        # df2 = pd.read_csv('gennaio2023-2-2.csv', sep=';')

        df1 = pd.read_csv('./Input/curve-gen-24.csv',
                          sep=';')  # USANO IL PUNTO E VIRGOLA COME SEPARATORE DEL CSV !!!!!!!!!!!!!!!!!!!!!!
        df2 = pd.read_csv('./Input/curve-feb-24.csv', sep=';')

        dfTOTinputAE = df1._append(df2)  # concatena (append) df1 e df2
        #dfTOTinputAE.to_csv(r'curve_GEN-FEB-24_AE_input_CSV.csv') # qui non serve risalvarlo in csv: già fatto
        print("ver2.3.4 - KNEAREST CLASSIFICATION da AHC (campione stratificato e classification dataset completo) con FEATURE SET SELECTION")
        print('curve_GEN-FEB_2024:')
        print(dfTOTinputAE.head(10))
    ########################################



    # print('converti')
    # X_input['A01'] = dfTOTinputAE['A01'].str.replace(',', '.').astype(float)  # automatizzare
    # X_input['A02'] = dfTOTinputAE['A02'].str.replace(',', '.').astype(float)
    # X_input['A03'] = dfTOTinputAE['A03'].str.replace(',', '.').astype(float)
    # X_input['A04'] = dfTOTinputAE['A04'].str.replace(',', '.').astype(float)
    # X_input['A05'] = dfTOTinputAE['A05'].str.replace(',', '.').astype(float)
    # X_input['A06'] = dfTOTinputAE['A06'].str.replace(',', '.').astype(float)
    # X_input['A07'] = dfTOTinputAE['A07'].str.replace(',', '.').astype(float)
    # X_input['A08'] = dfTOTinputAE['A08'].str.replace(',', '.').astype(float)
    # X_input['A09'] = dfTOTinputAE['A09'].str.replace(',', '.').astype(float)
    # X_input['A10'] = dfTOTinputAE['A10'].str.replace(',', '.').astype(float)
    # X_input['A11'] = dfTOTinputAE['A11'].str.replace(',', '.').astype(float)  # automatizzare
    # X_input['A12'] = dfTOTinputAE['A12'].str.replace(',', '.').astype(float)
    # X_input['A13'] = dfTOTinputAE['A13'].str.replace(',', '.').astype(float)
    # X_input['A14'] = dfTOTinputAE['A14'].str.replace(',', '.').astype(float)
    # X_input['A15'] = dfTOTinputAE['A15'].str.replace(',', '.').astype(float)
    # X_input['A16'] = dfTOTinputAE['A16'].str.replace(',', '.').astype(float)
    # X_input['A17'] = dfTOTinputAE['A17'].str.replace(',', '.').astype(float)
    # X_input['A18'] = dfTOTinputAE['A18'].str.replace(',', '.').astype(float)
    # X_input['A19'] = dfTOTinputAE['A19'].str.replace(',', '.').astype(float)
    # X_input['A20'] = dfTOTinputAE['A20'].str.replace(',', '.').astype(float)
    # X_input['A21'] = dfTOTinputAE['A21'].str.replace(',', '.').astype(float)  # automatizzare
    # X_input['A22'] = dfTOTinputAE['A22'].str.replace(',', '.').astype(float)
    # X_input['A23'] = dfTOTinputAE['A23'].str.replace(',', '.').astype(float)
    # X_input['A24'] = dfTOTinputAE['A24'].str.replace(',', '.').astype(float)
    # # X_input['AF1'] = dfTOTinputAE['AF1'].str.replace(',', '.').astype(float) #
    # # X_input['AF2'] = dfTOTinputAE['AF2'].str.replace(',', '.').astype(float)
    # # X_input['AF3'] = dfTOTinputAE['AF3'].str.replace(',', '.').astype(float)

    #X_input= X  #per test SW: test sul medesimo dataset di train/test ma senza splittarlo; poi da rimuovere; poi si dovrà lasciare qui il file di input vero con milioni di righe
    print("scaled_features_iniziali_shape:")
    print(scaled_features_iniziali.shape)
    print("scaled_features_iniziali_pod_shape:")
    print(scaled_features_iniziali_pod.shape)
    print("scaled_features_shape:")
    print(scaled_features.shape)

    if(0): #attivare per applicare K_nearest al DF COMPLETO
        X_input=scaled_features_iniziali # il DF iniziale è già stato scalato passo passo e non è necessario reimportarlo

    if(1): # attivare per applicare K_nearest al DF AGGREGATO PER POD
        X_input=scaled_features_iniziali_pod # il DF iniziale è già stato scalato passo passo e non è necessario reimportarlo

    X_new = X_input
    print("x_input_shape:")
    print(X_input.shape)
    print("x_new_shape:")
    print(X_new.shape)

    #X_new.drop(['AF1', 'AF2', 'AF3'])

    predictions = knn.predict(X_new)
    predictions = pd.DataFrame(predictions)
    #predictions.columns["predicted_K_nearest_label"]
    #prediction = pd.DataFrame(predictions, columns=['predictions']).to_csv(r'prediction.csv')
    predictions.to_csv("./output/prediction_results.csv")
    #print(classification_report(y_test, predictions))

    predictions_proba = knn.predict_proba(X_new)
    print('Predictions: {}'.format(predictions_proba))
    res = pd.DataFrame(predictions_proba)
    # res.index = X_input.index  # its important for comparison ...non riconosce l'attributo .index
    res.columns = ["prediction_proba_0","prediction_proba_1","prediction_proba_2","prediction_proba_3"]
    res.to_csv("./output/prediction_results_proba.csv")




###############################
# BLOCCO 3 - CLUSTERING AHC FASCE - FINE
###############################
    #?
    # return labels_df_HC

def disegni_AHC():

#importa labels (in formato numerico)
    labels_classificatione_basata_su_AHC_stratificato=pd.read_csv('prediction_results.csv', sep=';')
    print(labels_classificatione_basata_su_AHC_stratificato.head(90))
    labels_classificatione_basata_su_AHC_stratificato.info()
    print('test')


    labelsAHC=pd.DataFrame()
    labelsAHC.info()
    labelsAHC['labels_AHC_KNear']=labels_classificatione_basata_su_AHC_stratificato.iloc[:,[1]]
    labelsAHC['labels_AHC_KNear'] = labelsAHC['labels_AHC_KNear'].astype(str).astype(float).astype(int)
    #labelsAHC[0] = labelsAHC[0].astype(str).str.replace('.', ',')
    print('test2')
    labelsAHC.info()

    print(labelsAHC.head(90))
    labelsAHC.columns
    #labelsAHC=labelsAHC[0].astype(str).astype(int) non funziona
    #labelsAHC['labelAHC-KNear'] = pd.to_numeric(labelsAHC[0]) non funziona
    #labelsAHC['labelAHC-KNear']  = labels_classificatione_basata_su_AHC_stratificato[1:].astype(int) # non riconosce il punto come separatore # funziona ma non serve

#disegno base numerosità clusters
    if(0):
        labelsAHC.info()
        counts, edges, bars = plt.hist(labelsAHC)
        plt.bar_label(bars)
        plt.title("Numerosità CLusters AHC_KNear")
        #labelsAHC.plot.hist()
        plt.show()


#aggiunta colonna labels a DB iniziale GME

    if(1):
        LABELED_DB = pd.read_csv("./Output/curve_GEN-FEB_2024_AE_input_CSV_Aree_GME.csv")
        LABELED_DB['labels_AHC_KNear']=labelsAHC['labels_AHC_KNear']
        LABELED_DB.info()
        LABELED_DB.columns

        LABELED_DB_NUM=pd.DataFrame()
        LABELED_DB_NUM['A01'] = LABELED_DB['A01'].str.replace(',', '.').astype(float).round(3)  # automatizzare
        LABELED_DB_NUM['A02'] = LABELED_DB['A02'].str.replace(',', '.').astype(float).round(3)
        LABELED_DB_NUM['A03'] = LABELED_DB['A03'].str.replace(',', '.').astype(float).round(3)
        LABELED_DB_NUM['A04'] = LABELED_DB['A04'].str.replace(',', '.').astype(float).round(3)
        LABELED_DB_NUM['A05'] = LABELED_DB['A05'].str.replace(',', '.').astype(float).round(3)
        LABELED_DB_NUM['A06'] = LABELED_DB['A06'].str.replace(',', '.').astype(float).round(3)
        LABELED_DB_NUM['A07'] = LABELED_DB['A07'].str.replace(',', '.').astype(float).round(3)
        LABELED_DB_NUM['A08'] = LABELED_DB['A08'].str.replace(',', '.').astype(float).round(3)
        LABELED_DB_NUM['A09'] = LABELED_DB['A09'].str.replace(',', '.').astype(float).round(3)
        LABELED_DB_NUM['A10'] = LABELED_DB['A10'].str.replace(',', '.').astype(float).round(3)
        LABELED_DB_NUM['A11'] = LABELED_DB['A11'].str.replace(',', '.').astype(float).round(3) # automatizzare
        LABELED_DB_NUM['A12'] = LABELED_DB['A12'].str.replace(',', '.').astype(float).round(3)
        LABELED_DB_NUM['A13'] = LABELED_DB['A13'].str.replace(',', '.').astype(float).round(3)
        LABELED_DB_NUM['A14'] = LABELED_DB['A14'].str.replace(',', '.').astype(float).round(3)
        LABELED_DB_NUM['A15'] = LABELED_DB['A15'].str.replace(',', '.').astype(float).round(3)
        LABELED_DB_NUM['A16'] = LABELED_DB['A16'].str.replace(',', '.').astype(float).round(3)
        LABELED_DB_NUM['A17'] = LABELED_DB['A17'].str.replace(',', '.').astype(float).round(3)
        LABELED_DB_NUM['A18'] = LABELED_DB['A18'].str.replace(',', '.').astype(float).round(3)
        LABELED_DB_NUM['A19'] = LABELED_DB['A19'].str.replace(',', '.').astype(float).round(3)
        LABELED_DB_NUM['A20'] = LABELED_DB['A20'].str.replace(',', '.').astype(float).round(3)
        LABELED_DB_NUM['A21'] = LABELED_DB['A21'].str.replace(',', '.').astype(float).round(3)  # automatizzare
        LABELED_DB_NUM['A22'] = LABELED_DB['A22'].str.replace(',', '.').astype(float).round(3)
        LABELED_DB_NUM['A23'] = LABELED_DB['A23'].str.replace(',', '.').astype(float).round(3)
        LABELED_DB_NUM['A24'] = LABELED_DB['A24'].str.replace(',', '.').astype(float).round(3)
        LABELED_DB_NUM['labels_AHC_KNear'] = LABELED_DB['labels_AHC_KNear']
        LABELED_DB_NUM.info()

        # define index column
        #LABELED_DB_NUM.set_index(['A01','A02'], inplace=True)

        # group data by product and display sales as line chart
        df=pd.DataFrame(LABELED_DB_NUM.groupby('labels_AHC_KNear').aggregate('median').round(3))
        df.to_csv("./output/output_mediane_orarie_clusters.csv")
        df = pd.DataFrame(LABELED_DB_NUM.groupby('labels_AHC_KNear').aggregate('mean').round(3))
        df.to_csv("./output/output_medie_orarie_clusters.csv")
        df = pd.DataFrame(LABELED_DB_NUM.groupby('labels_AHC_KNear').aggregate('max').round(3))
        df.to_csv("./output/output_max_orarie_clusters.csv")
        df = pd.DataFrame(LABELED_DB_NUM.groupby('labels_AHC_KNear').aggregate('min').round(3))
        df.to_csv("./output/output_min_orarie_clusters.csv")
        #plt.show()


def disegni_Kmeans():

    if(1):
        labels_df = pd.read_csv('./output/labels_df_KMEANS_4.csv', sep=';')
        print(labels_df.head(90))
        labels_df.info()
        print('test')

        labelsKmeans = pd.DataFrame()
        labelsKmeans.info()
        labelsKmeans['labels_Kmeans'] = labels_df.iloc[:, [1]]
        labelsKmeans['labels_Kmeans'] = labelsKmeans['labels_Kmeans'].astype(str).astype(float).astype(int)
        # labelsAHC[0] = labelsAHC[0].astype(str).str.replace('.', ',')
        print('test2')
        labelsKmeans.info()

        print(labelsKmeans.head(90))
        labelsKmeans.columns
        labelsKmeans.info()

        # disegno base numerosità clusters
        counts, edges, bars = plt.hist(labelsKmeans)
        plt.bar_label(bars)
        plt.title("Numerosità CLusters Kmeans - GEN-FEB24")

        #labels_df.plot.hist()
        plt.show()

        # aggiunta colonna labels a DB iniziale GME

        if (1):
            LABELED_DB = pd.read_csv("./output/curve_GEN-FEB_2024_AE_input_CSV_Aree_GME.csv")
            LABELED_DB['labels_Kmeans'] = labelsKmeans['labels_Kmeans']
            LABELED_DB.info()
            LABELED_DB.columns

            LABELED_DB_NUM = pd.DataFrame()
            LABELED_DB_NUM['A01'] = LABELED_DB['A01'].str.replace(',', '.').astype(float).round(3)  # automatizzare
            LABELED_DB_NUM['A02'] = LABELED_DB['A02'].str.replace(',', '.').astype(float).round(3)
            LABELED_DB_NUM['A03'] = LABELED_DB['A03'].str.replace(',', '.').astype(float).round(3)
            LABELED_DB_NUM['A04'] = LABELED_DB['A04'].str.replace(',', '.').astype(float).round(3)
            LABELED_DB_NUM['A05'] = LABELED_DB['A05'].str.replace(',', '.').astype(float).round(3)
            LABELED_DB_NUM['A06'] = LABELED_DB['A06'].str.replace(',', '.').astype(float).round(3)
            LABELED_DB_NUM['A07'] = LABELED_DB['A07'].str.replace(',', '.').astype(float).round(3)
            LABELED_DB_NUM['A08'] = LABELED_DB['A08'].str.replace(',', '.').astype(float).round(3)
            LABELED_DB_NUM['A09'] = LABELED_DB['A09'].str.replace(',', '.').astype(float).round(3)
            LABELED_DB_NUM['A10'] = LABELED_DB['A10'].str.replace(',', '.').astype(float).round(3)
            LABELED_DB_NUM['A11'] = LABELED_DB['A11'].str.replace(',', '.').astype(float).round(3)  # automatizzare
            LABELED_DB_NUM['A12'] = LABELED_DB['A12'].str.replace(',', '.').astype(float).round(3)
            LABELED_DB_NUM['A13'] = LABELED_DB['A13'].str.replace(',', '.').astype(float).round(3)
            LABELED_DB_NUM['A14'] = LABELED_DB['A14'].str.replace(',', '.').astype(float).round(3)
            LABELED_DB_NUM['A15'] = LABELED_DB['A15'].str.replace(',', '.').astype(float).round(3)
            LABELED_DB_NUM['A16'] = LABELED_DB['A16'].str.replace(',', '.').astype(float).round(3)
            LABELED_DB_NUM['A17'] = LABELED_DB['A17'].str.replace(',', '.').astype(float).round(3)
            LABELED_DB_NUM['A18'] = LABELED_DB['A18'].str.replace(',', '.').astype(float).round(3)
            LABELED_DB_NUM['A19'] = LABELED_DB['A19'].str.replace(',', '.').astype(float).round(3)
            LABELED_DB_NUM['A20'] = LABELED_DB['A20'].str.replace(',', '.').astype(float).round(3)
            LABELED_DB_NUM['A21'] = LABELED_DB['A21'].str.replace(',', '.').astype(float).round(3)  # automatizzare
            LABELED_DB_NUM['A22'] = LABELED_DB['A22'].str.replace(',', '.').astype(float).round(3)
            LABELED_DB_NUM['A23'] = LABELED_DB['A23'].str.replace(',', '.').astype(float).round(3)
            LABELED_DB_NUM['A24'] = LABELED_DB['A24'].str.replace(',', '.').astype(float).round(3)
            LABELED_DB_NUM['labels_Kmeans'] = LABELED_DB['labels_Kmeans']
            LABELED_DB_NUM.info()

            # define index column
            # LABELED_DB_NUM.set_index(['A01','A02'], inplace=True)

            # group data by product and display sales as line chart
            df = pd.DataFrame(LABELED_DB_NUM.groupby('labels_Kmeans').aggregate('median').round(3))
            df.to_csv("./output/output_mediane_orarie_clusters_Kmeans.csv")
            df = pd.DataFrame(LABELED_DB_NUM.groupby('labels_Kmeans').aggregate('mean').round(3))
            df.to_csv("./output/output_medie_orarie_clusters_Kmeans.csv")
            df = pd.DataFrame(LABELED_DB_NUM.groupby('labels_Kmeans').aggregate('max').round(3))
            df.to_csv("./output/output_max_orarie_clusters_Kmeans.csv")
            df = pd.DataFrame(LABELED_DB_NUM.groupby('labels_Kmeans').aggregate('min').round(3))
            df.to_csv("./output/output_min_orarie_clusters_Kmeans.csv")
            # plt.show()


def print_test(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'TEST, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

    print('test')



##############################################################
#BLOCCO 0 - MAIN - Selezione Features da dizionario variabile
##############################################################


def main():
    print_test('PyCharm-test_AE_TP_RZ')
    print_test('PyCharm-test_AE_TP_RZ2')
    print_test('PyCharm-test_AE_TP_RZ2.L')
    options_features = {
                        'AF1': (True, 'feature_extraction_AF1', 'df'),
                        'AF2': (True, 'feature_extraction_AF2', 'df'),
                        'AF3': (True, 'feature_extraction_AF3', 'df'),
                        'A01': (True, 'feature_extraction_A01', 'df'),
                        'A02': (True, 'feature_extraction_A02', 'df'),
                        'A03': (True, 'feature_extraction_A03', 'df'),
                        'A04': (True, 'feature_extraction_A04', 'df'),
                        'A05': (True, 'feature_extraction_A05', 'df'),
                        'A06': (True, 'feature_extraction_A06', 'df'),
                        'A07': (True, 'feature_extraction_A07', 'df'),
                        'A08': (True, 'feature_extraction_A08', 'df'),
                        'A09': (True, 'feature_extraction_A09', 'df'),
                        'A10': (True, 'feature_extraction_A10', 'df'),
                        'A11': (True, 'feature_extraction_A11', 'df'),
                        'A12': (True, 'feature_extraction_A12', 'df'),
                        'A13': (True, 'feature_extraction_A13', 'df'),
                        'A14': (True, 'feature_extraction_A14', 'df'),
                        'A15': (True, 'feature_extraction_A15', 'df'),
                        'A16': (True, 'feature_extraction_A16', 'df'),
                        'A17': (True, 'feature_extraction_A17', 'df'),
                        'A18': (True, 'feature_extraction_A18', 'df'),
                        'A19': (True, 'feature_extraction_A19', 'df'),
                        'A20': (True, 'feature_extraction_A20', 'df'),
                        'A21': (True, 'feature_extraction_A21', 'df'),
                        'A22': (True, 'feature_extraction_A22', 'df'),
                        'A23': (True, 'feature_extraction_A23', 'df'),
                        'A24': (True, 'feature_extraction_A24', 'df')#,
                        #'A25': (True, 'feature_extraction_A25', 'df')#'Area_GME': (True, 'feature_extraction_Area_GME', 'df')#non funzionante (non è numerica) e non è una feature opportuna (serve solo per stratificare il campionamento per area geografica GME, non per clusterizzare)
                                                                        }
    print_test('PyCharm-test_AE_TP_RZ-function switch')
    # FUNCTION SWITCH: comment/decomment 'mains' to activate functions; set true/false to set features extraction options
    if(1):
        feature_set_selection = 4 #0=tutto; 1=FASCE; 2=FASCE+A01; 3=RILEVAZIONI ORARIE 0-24; 4=RILEVAZIONI ORARIE+FASCE
        main_features(options_features,feature_set_selection)

    # main_signal_selection() # eventualmente inserire opzioni come in main_features #QUESTO PASSAGGIO NON DOVREBBE ESSERE NECESSARIO POICHE' ARRIVANO DATI PULITI 6/11/23
    if(0): #per applicarlo al campione stratificato usato anche in AHC,attivare il blocco:
        # CAMPIONAMENTO STRATIFICATO (MULTIPLO) IN BASE ALLA CLASSIFICAZIONE REGIONALE GME e dati geografici (sottoinsiemi di GME)
        #altrimenti,se il blocco è disattivato, Kmeans viene applicato a tutto il DB di input
        label_df= main_clustering_kmeans() # eventualmente inserire opzioni mono-multi come in main_features (anche fou vs corr)

    if(1):#applicato al campione stratificato (comprende il successivo passaggio di classificazione KNearest, necessario per limitare i tempi di calcolo)
        label_df_AHC= main_clustering_AHC()

    if(0):
        disegni_Kmeans()

    if(0):
        disegni_AHC()


# Sandro
def main_2 ():
    main()
    
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

##############################################################

#BLOCCO 0 - MAIN - Selezione Features da dizionario variabile - FINE
##############################################################

#Sandro
# # # # Ritorna l'output catturato alla normalità
# sys.stdout = sys.__stdout__
# sys.stderr = sys.__stderr__

# # Ottieni l'output catturato come stringa
# output_content = output_capture.getvalue()

# # Chiudi l'oggetto StringIO
# output_capture.close()

