import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from MachineLearning.KNN import KNN
from MachineLearning.RandomForest import RandomForest
from MachineLearning.Xgboost import Xgboost

class FeatureSelection:
    def __init__(self, instance):
        self.__datos = None
        self.__clases = None
        self.__trainingData = None
        self.__trainingClass = None
        self.__testingData = None
        self.__testingClass = None
        self.__gamma = 0.99
        self.__totalFeature = None
        self.readInstance(instance)

    def setDatos(self, datos):
        self.__datos = datos
    def getDatos(self):
        return self.__datos
    def setClases(self, clases):
        self.__clases = clases
    def getClases(self):
        return self.__clases
    def setTrainingData(self, trainingData):
        self.__trainingData = trainingData
    def getTrainingData(self):
        return self.__trainingData
    def setTrainingClass(self, trainingClass):
        self.__trainingData = trainingClass
    def getTrainingClass(self):
        return self.__trainingClass
    def setTestingData(self, testingData):
        self.__testingData = testingData
    def getTestingData(self):
        return self.__testingData
    def setTestingClass(self, testingClass):
        self.__testingClass = testingClass
    def getTestingClass(self):
        return self.__testingClass
    def setGamma(self, gamma):
        self.__gamma = gamma
    def getGamma(self):
        return self.__gamma
    def setTotalFeature(self, totalFeature):
        self.__totalFeature = totalFeature
    def getTotalFeature(self):
        return self.__totalFeature

    def readInstance(self, instance):        
        print(instance)
        if instance == 'ionosphere':
            instance = instance+".data"
            classPosition = 34
            dataset = pd.read_csv('Problem/FS/Instances/'+instance, header=None)
            clases = dataset.iloc[:,classPosition]
            clases = clases.replace({
                'b':0,
                'g':1
            })
            clases = clases.values

            datos = dataset.drop(dataset.columns[classPosition],axis='columns')
            
        if instance == "nefrologia":
            instance = instance+".csv"
            classPosition = 62
            dataset = pd.read_csv('Problem/FS/Instances/'+instance)
            dataset = dataset.drop(['Sex','Diabetes','H0_Bano','H0_Dializador','Hypo_Type','TAS_Diff','Hypo_Type2'], axis=1)
            clases = dataset.iloc[:,classPosition]
            clases = clases.values
            
            datos = dataset.drop(dataset.columns[classPosition], axis='columns')
            
        if instance == "only_clinic":
            instance = instance+".csv"
            dataset = pd.read_csv('Problem/FS/Instances/'+instance)
            ## Method target encoding
            encode_columns = ['H0_Dializador_2', 'H0_Bano_2', 'Season']
            categorical_features = [*encode_columns, "Sex", "Hypo_Bin"]
            numeric_columns = dataset.columns[dataset.columns.isin(categorical_features) == False].values
            
            ## Encoding 
            encode_df = dataset[encode_columns]
            encode_df = encode_df.astype('str')
            one_hot_encoded = pd.get_dummies(encode_df)
            
            all_data = pd.concat([dataset, one_hot_encoded], axis=1)
            all_data.drop(encode_columns, axis=1, inplace=True)
            datos = all_data.loc[:, all_data.columns[all_data.columns != "Hypo_Bin"]]
            clases = all_data.loc[:, "Hypo_Bin"]
            
        if instance == "dat_3_3_1":
            instance = instance+".csv"
            dat3 = pd.read_csv('Problem/FS/Instances/'+instance)
            
            ## Drop columns with > 10% of missing data
            cols2drop = dat3.columns[((dat3.isna().sum(axis=0) / dat3.shape[0]) > 0.1)] 

            dat3.drop(cols2drop, axis=1, inplace=True)
            dat3.dropna(inplace=True)

            whisker_width_na = 3.5 ##TODO: This is something we may want to tune
            whisker_width_median = 3 ##TODO: This is something we may want to tune
            check_for_outliers = ["NEUTROFILOS#", "MONOCITOS#", "LINFOCITOS#", "V.C.M.", "H.C.M.", "EOSINOFILOS#", "BASOFILOS#", "LEUCOCITOS", "LUC#", "FILTRACIONGLOMERULARCKD-EPI",
                        "GGT", "AST/GOT", "PCR", "BILIRRUBINATOTAL", "FOSFATASAALCALINA", "FERRITINA", "PTHintacta", "TRIGLICERIDOS", "BETA2MICROGLOBULINASUERO", 
                        "VITAMINAB12", "H0_Ganancia_2", "H0_UF_2", "H0_Pulso_2", "H0_ConductividadBano_2", "H0_FlujoSangre_2", "H0_Ganancia_2", "H0_PresionArterial_2", "H0_TAD_2"]
            # for i in dat3.columns[(dat3.dtypes == "int64")  | (dat3.dtypes == "float64")].values:
            for i in check_for_outliers:
                q1, q3 = dat3[i].quantile([0.25, 0.75])
                iqr = q3 - q1
                # dat3.loc[dat3[i] < q1 - whisker_width_na * iqr, i] = q1 - whisker_width_na * iqr
                # dat3.loc[dat3[i] > q3 + whisker_width_na * iqr, i] = q3 + whisker_width_na * iqr
                dat3.loc[dat3[i] < q1 - whisker_width_na * iqr, i] = np.nan
                dat3.loc[dat3[i] > q3 + whisker_width_na * iqr, i] = np.nan
                # dat3.loc[(dat3[i] >= q1 - whisker_width_median * iqr) & (dat3[i] <= q3 + whisker_width_na * iqr), i] = dat3[i].median()
                # dat3.loc[(dat3[i] < q1 - whisker_width_median * iqr) & (dat3[i] > q1 - whisker_width_na * iqr), i] = dat3[i].median()
                # dat3.loc[(dat3[i] > q3 + whisker_width_median * iqr) & (dat3[i] < q3 - whisker_width_na * iqr), i] = dat3[i].median()
            dat3[dat3.TAS_Diff.abs()>105] = np.nan
            # dat3[dat3.TAS_Diff.abs()>105] = np.nan
            dat3[dat3.H0_TAS_2.abs()>200] = np.nan
            dat3[dat3.H0_TAS_2.abs()<80] = np.nan
            dat3[dat3.H0_TAD_2.abs()<35] = np.nan
            dat3[dat3.H0_TAD_2.abs()>115] = np.nan
            dat3[dat3.H0_PTM_2>300] = np.nan
            dat3[dat3.H0_PTM_2<=0] = np.nan
            dat3[dat3.H0_PresionArterial_2<=-350] = np.nan
            dat3[dat3.H0_PresionArterial_2>=-70] = np.nan
            dat3[dat3.H0_PresionVenosa_2<=75] = np.nan
            dat3[dat3.H0_PresionVenosa_2>=350] = np.nan
            dat3[dat3.Edad>100] = np.nan
            # dat3[dat3.Edad<] = np.nan
            dat3[dat3.H0_UF_2<=0] = np.nan

            dat3["PCR"] = np.log10(dat3.PCR)

            dat3.dropna(inplace=True)

            encode_columns = ['H0_Dializador_2', 'H0_Bano_2', 'Season']
            ## Encoding 
            encode_df = dat3[encode_columns]
            encode_df = encode_df.astype('str')
            one_hot_encoded = pd.get_dummies(encode_df)
            
            all_data = pd.concat([dat3, one_hot_encoded], axis=1)
            all_data.drop(encode_columns, axis=1, inplace=True)
            all_data.drop("Hypo_Type", axis=1, inplace=True)
            all_data.drop("TAS_Diff", axis=1, inplace=True)
            datos = all_data.loc[:, all_data.columns[all_data.columns != "Hypo_Bin"]]
            clases = all_data.loc[:, "Hypo_Bin"]
            
        if instance == 'sonar':
            instance = instance+".all-data"
            classPosition = 60
            dataset = pd.read_csv('Problem/FS/Instances/'+instance, header=None)
            clases = dataset.iloc[:,classPosition]
            clases = clases.replace({
                'R':0,
                'M':1
            })
            clases = clases.values

            datos = dataset.drop(dataset.columns[classPosition],axis='columns')
        
        if instance == 'Cervical Cancer':
            instance = "sobar-72.csv"
            classPosition = 19
            dataset = pd.read_csv('Problem/FS/Instances/'+instance, header=None)
            clases = dataset.iloc[:,classPosition]
            clases = clases.values

            datos = dataset.drop(dataset.columns[classPosition],axis='columns')
            
        if instance == 'Immunotherapy':
            instance = instance+".csv"
            classPosition = 7
            dataset = pd.read_csv('Problem/FS/Instances/'+instance, header=None)
            clases = dataset.iloc[:,classPosition]
            clases = clases.values

            datos = dataset.drop(dataset.columns[classPosition],axis='columns')
            
        if instance == 'Divorce':
            instance = "divorce.csv"
            classPosition = 54
            dataset = pd.read_csv('Problem/FS/Instances/'+instance, header=None)
            clases = dataset.iloc[:,classPosition]
            clases = clases.values

            datos = dataset.drop(dataset.columns[classPosition],axis='columns')
            
            
        if instance == 'Hill-Valley-with-noise.data' or instance == 'Hill-Valley-without-noise.data':
            classPosition = 100
            dataset = pd.read_csv('Problem/FS/Instances/'+instance, header=None)
            clases = dataset.iloc[:,classPosition]
            clases = clases.values

            datos = dataset.drop(dataset.columns[classPosition],axis='columns')
        
        # posee datos nulos (?)
        
        if instance == 'breast-cancer-wisconsin':
            instance = instance+".data"
            classPosition = 10
            dataset = pd.read_csv('Problem/FS/Instances/'+instance, header=None)
            clases = dataset.iloc[:,classPosition]
            clases = clases.replace({
                2:0,
                4:1
            })
            clases = clases.values

            datos = dataset.drop(dataset.columns[classPosition],axis='columns')
        
        if instance == 'wdbc':
            instance = instance+".data"
            classPosition = 1
            dataset = pd.read_csv('Problem/FS/Instances/'+instance, header=None)
            clases = dataset.iloc[:,classPosition]
            clases = clases.replace({
                'M':0,
                'B':1
            })
            clases = clases.values

            datos = dataset.drop(dataset.columns[classPosition],axis='columns')

        self.setClases(clases)
        self.setDatos(datos)
        self.setTotalFeature(len(datos.columns))

    def selection(self, seleccion):

        datos = self.getDatos().iloc[:, seleccion]

        escalador = preprocessing.MinMaxScaler()
        # escalador = preprocessing.StandardScaler()

        train_ratio = 0.8
        test_ratio = 0.2
        SEED = 12
        
        trainingData, testingData, trainingClass, testingClass  = train_test_split(
            datos,
            self.getClases(),
            test_size= 1 - train_ratio,
            random_state=SEED,
            stratify=self.getClases()
        )

        trainingData = escalador.fit_transform(trainingData)
        testingData = escalador.fit_transform(testingData)

        return trainingData, testingData, trainingClass, testingClass

    def fitness(self, individuo, clasificador, parametrosC):
        accuracy = 0 
        f1Score = 0
        presicion = 0
        recall = 0
        mcc = 0
        trainingData, testingData, trainingClass, testingClass = self.selection(individuo)
        # cm, accuracy, f1Score, presicion, recall, mcc = self.KNN(trainingData, testingData, trainingClass, testingClass)

        if clasificador == 'KNN':
            accuracy, f1Score, presicion, recall, mcc = KNN(trainingData, testingData, trainingClass, testingClass, int(parametrosC.split(":")[1]))
        if clasificador == 'RandomForest':
            accuracy, f1Score, presicion, recall, mcc = RandomForest(trainingData, testingData, trainingClass, testingClass)       
        if clasificador == 'Xgboost':
            accuracy, f1Score, presicion, recall, mcc = Xgboost(trainingData, testingData, trainingClass, testingClass)
            
        errorRate = np.round((1 - accuracy), decimals=3)

        fitness = np.round(( self.getGamma() * errorRate ) + ( ( 1 - self.getGamma() ) * ( len(individuo) / self.getTotalFeature() ) ), decimals=3)

        # return fitness, cm, accuracy, f1Score, presicion, recall, mcc, errorRate
        return fitness, accuracy, f1Score, presicion, recall, mcc, errorRate, len(individuo)

    def factibilidad(self, individuo):
        suma = np.sum(individuo)
        if suma > 0:
            return True
        else:
            return False

    def nuevaSolucion(self):
        return np.random.randint(low=0, high=2, size = self.getTotalFeature())
