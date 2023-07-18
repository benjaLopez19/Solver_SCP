import pandas as pd
from BD.sqlite import BD
import numpy as np


bd = BD()

# instance = "nefrologia.csv"
# classPosition = 62
# dataset = pd.read_csv('Problem/FS/Instances/'+instance)
# dataset = dataset.drop(['Sex','Diabetes','H0_Bano','H0_Dializador','Hypo_Type','TAS_Diff','Hypo_Type2'], axis=1)
# clases = dataset.iloc[:,classPosition]
# clases = clases.values

# datos = dataset.drop(dataset.columns[classPosition], axis='columns')

# column_names = list(datos.columns)

# print("-----------------------------------------------------------------------------------------------------------------------")
# print(column_names)
# print("-----------------------------------------------------------------------------------------------------------------------")

instance = "dat_3_3_1.csv"
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
datos = all_data.loc[:, all_data.columns[all_data.columns != "Hypo_Bin"]]
clases = all_data.loc[:, "Hypo_Bin"]

column_names = list(datos.columns)

print("-----------------------------------------------------------------------------------------------------------------------")
print(column_names)
print("-----------------------------------------------------------------------------------------------------------------------")


clasificadores = ["KNN"]

binarizaciones = ["100","200","300","400","500"]

for clasificador in clasificadores:
    for bss in binarizaciones:
        
        blob = bd.obtenerMejoresArchivosconClasificadorBSS('dat_3_3_1',"",clasificador,bss)
        
        for d in blob:

            seleccionGWO = []
            seleccionMFO = []
            seleccionPSA = []
            seleccionSCA = []
            seleccionWOA = []



            caracteristicas = []
            mh = d[1]
            solucion = d[9].replace("[","").replace("]","").replace("0.0",str(0)).replace("1.0",str(1)).split(",")
            fitness = d[8]
            print(f'''dat_3_3_1 - {mh} - {bss} - {clasificador} - {fitness}''')
            i = 0
            for feature in solucion:
                
                if int(feature) == 1:
                    caracteristicas.append(column_names[i])
                    if mh == 'GA':
                        seleccionGWO.append(i)
                    
                    if mh == 'MFO':
                        seleccionMFO.append(i)
                        
                    if mh == 'PSA':
                        seleccionPSA.append(i)
                        
                    if mh == 'SCA':
                        seleccionSCA.append(i)
                        
                    if mh == 'WOA':
                        seleccionWOA.append(i)
                
                i+=1
            print("-----------------------------------------------------------------------------------------------------------------------")
            print(caracteristicas)
            print("-----------------------------------------------------------------------------------------------------------------------")
            print(seleccionGWO)
            print(len(seleccionGWO))
            print("-----------------------------------------------------------------------------------------------------------------------")
            
            # print("-----------------------------------------------------------------------------------------------------------------------")
            # print(seleccionMFO)
            # print(len(seleccionMFO))
            # print("-----------------------------------------------------------------------------------------------------------------------")
            # print(seleccionPSA)
            # print(len(seleccionPSA))
            # print("-----------------------------------------------------------------------------------------------------------------------")
            # print(seleccionSCA)
            # print(len(seleccionSCA))
            # print("-----------------------------------------------------------------------------------------------------------------------")
            # print(seleccionWOA)
            # print(len(seleccionWOA))
            # print("-----------------------------------------------------------------------------------------------------------------------")

# diccionario = {}

# for caracteristica in seleccionGWO:
#     if column_names[caracteristica] in diccionario:
#         diccionario[column_names[caracteristica]] =  diccionario[column_names[caracteristica]] + 1
#     else:
#         diccionario[column_names[caracteristica]] = 1
        
# for caracteristica in seleccionMFO:
#     if column_names[caracteristica] in diccionario:
#         diccionario[column_names[caracteristica]] =  diccionario[column_names[caracteristica]] + 1
#     else:
#         diccionario[column_names[caracteristica]] = 1

# for caracteristica in seleccionPSA:
#     if column_names[caracteristica] in diccionario:
#         diccionario[column_names[caracteristica]] =  diccionario[column_names[caracteristica]] + 1
#     else:
#         diccionario[column_names[caracteristica]] = 1

# for caracteristica in seleccionSCA:
#     if column_names[caracteristica] in diccionario:
#         diccionario[column_names[caracteristica]] =  diccionario[column_names[caracteristica]] + 1
#     else:
#         diccionario[column_names[caracteristica]] = 1

# for caracteristica in seleccionWOA:
#     if column_names[caracteristica] in diccionario:
#         diccionario[column_names[caracteristica]] =  diccionario[column_names[caracteristica]] + 1
#     else:
#         diccionario[column_names[caracteristica]] = 1


# import operator

# diccionarionOrdenado = sorted(diccionario.items(), key=operator.itemgetter(1), reverse=True)

# # print(diccionarionOrdenado)

# for campo in diccionarionOrdenado:
#     print(campo)