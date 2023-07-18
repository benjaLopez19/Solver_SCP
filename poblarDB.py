from BD.sqlite import BD
import json

bd = BD()


fs  = False
scp = False
ben = False
empatia = True
mkp = False
ml = False
mhs = ['GA']
cantidad = 0

DS_actions = [
    'V1-STD', 'V1-COM', 'V1-PS', 'V1-ELIT',
    'V2-STD', 'V2-COM', 'V2-PS', 'V2-ELIT',
    'V3-STD', 'V3-COM', 'V3-PS', 'V3-ELIT',
    'V4-STD', 'V4-COM', 'V4-PS', 'V4-ELIT',
    'S1-STD', 'S1-COM', 'S1-PS', 'S1-ELIT',
    'S2-STD', 'S2-COM', 'S2-PS', 'S2-ELIT',
    'S3-STD', 'S3-COM', 'S3-PS', 'S3-ELIT',
    'S4-STD', 'S4-COM', 'S4-PS', 'S4-ELIT',
]

paramsML = json.dumps({
    'MinMax'        : 'min',
    'DS_actions'    : DS_actions,
    'gamma'         : 0.4,
    'policy'        : 'e-greedy',
    'qlAlphaType'   : 'static',
    'rewardType'    : 'withPenalty1',
    'stateQ'        : 2
})


if empatia:
    # poblar ejecuciones FS
    instancias = bd.obtenerInstancias(f'''
                                      'EMPATIA-9'
                                      ''')
    iteraciones = 500
    experimentos = 3
    poblacion = 50
    for instancia in instancias:

        for mh in mhs:
            if ml:
                data = {}
                data['MH']          = mh
                data['paramMH']     = f'iter:{str(iteraciones)},pop:{str(poblacion)},cros:0.6;mut:0.01'
                data['ML']          = 'Q-Learning'
                data['paramML']     = paramsML
                data['ML_FS']       = ''
                data['paramML_FS']  = ''
                data['estado']      = 'pendiente'

                cantidad +=experimentos
                bd.insertarExperimentos(data, experimentos, instancia[0])
            else:
                data = {}
                data['MH']          = mh
                data['paramMH']     = f'iter:{str(iteraciones)},pop:{str(poblacion)},DS:V4-STD,cros:0.9;mut:0.01'
                # data['paramMH']     = f'iter:{str(iteraciones)},pop:{str(poblacion)},DS:V4-ELIT'
                # data['paramMH']     = f'iter:{str(iteraciones)},pop:{str(poblacion)},DS:X4-ELIT'
                # data['paramMH']     = f'iter:{str(iteraciones)},pop:{str(poblacion)},DS:Z4-ELIT'
                data['ML']          = ''
                data['paramML']     = ''
                data['ML_FS']       = ''
                data['paramML_FS']  = ''
                data['estado']      = 'pendiente'

                cantidad +=experimentos
                bd.insertarExperimentos(data, experimentos, instancia[0])

if fs:
    # poblar ejecuciones FS
    # instancias = bd.obtenerInstancias(f'''
    #                                   "sonar","ionosphere","Immunotherapy","Divorce","wdbc","breast-cancer-wisconsin"
    #                                   ''')
    instancias = bd.obtenerInstancias(f'''
                                      "dat_3_3_1"
                                      ''')
    iteraciones = 400
    experimentos = 1
    poblacion = 20
    # clasificadores = ["KNN","RandomForest","Xgboost"]
    clasificadores = ["KNN"]
    # DS_actions = [
    #     'S4-STD', 'S4-COM', 'S4-ELIT',
    #     'V4-STD', 'V4-COM', 'V4-ELIT',
    #     'X4-STD', 'X4-COM', 'X4-ELIT',
    #     'Z4-STD', 'Z4-COM', 'Z4-ELIT']
    DS_actions = ['S4-STD']
    # clasificadores = ["KNN"]
    for instancia in instancias:

        for mh in mhs:
            for clasificador in clasificadores:
                if ml:
                    data = {}
                    data['MH']          = mh
                    data['paramMH']     = f'iter:{str(iteraciones)},pop:{str(poblacion)}'
                    data['ML']          = 'Q-Learning'
                    data['paramML']     = paramsML
                    data['ML_FS']       = clasificador
                    data['paramML_FS']  = f'k:5'
                    data['estado']      = 'pendiente'

                    cantidad +=experimentos
                    bd.insertarExperimentos(data, experimentos, instancia[0])
                else:
                    for ds in DS_actions:
                        data = {}
                        data['MH']          = mh
                        data['paramMH']     = f'iter:{str(iteraciones)},pop:{str(poblacion)},DS:{ds}'
                        # data['paramMH']     = f'iter:{str(iteraciones)},pop:{str(poblacion)},DS:S4-COM'
                        data['ML']          = ''
                        data['paramML']     = ''
                        data['ML_FS']       = clasificador
                        data['paramML_FS']  = f'k:5'
                        data['estado']      = 'pendiente'

                        cantidad +=experimentos
                        bd.insertarExperimentos(data, experimentos, instancia[0])

if scp:
    # poblar ejecuciones SCP
    instancias = bd.obtenerInstancias(f'''
                                      "scp41"
                                      ''')
    iteraciones = 1000
    experimentos = 1
    poblacion = 50
    for instancia in instancias:

        for mh in mhs:
            data = {}
            data['MH']          = mh
            data['paramMH']     = f'iter:{str(iteraciones)},pop:{str(poblacion)},DS:V4-ELIT,repair:complex'
            data['ML']          = ''
            data['paramML']     = ''
            data['ML_FS']       = ''
            data['paramML_FS']  = ''
            data['estado']      = 'pendiente'

            cantidad +=experimentos
            bd.insertarExperimentos(data, experimentos, instancia[0])

if mkp:
    # poblar ejecuciones MKP
    instancias = bd.obtenerInstancias(f'''
                                      "mknap1_2","mknap2_2"
                                      ''')
    iteraciones = 500
    experimentos = 1
    poblacion = 10
    for instancia in instancias:

        for mh in mhs:
            data = {}
            data['MH']          = mh
            data['paramMH']     = f'iter:{str(iteraciones)},pop:{str(poblacion)},DS:S4-ELIT'
            data['ML']          = ''
            data['paramML']     = ''
            data['ML_FS']       = ''
            data['paramML_FS']  = ''
            data['estado']      = 'pendiente'

            cantidad +=experimentos
            bd.insertarExperimentos(data, experimentos, instancia[0])
            
if ben:
    # poblar ejecuciones Benchmark
    instancias = bd.obtenerInstancias(f'''
                                      "F7","F8","F9","F10"
                                      ''')
    iteraciones = 500
    experimentos = 31
    poblacion = 30
    for instancia in instancias:
        for mh in mhs:
            data = {}
            data['MH']          = mh
            data['paramMH']     = f'iter:{str(iteraciones)},pop:{str(poblacion)}'
            data['ML']          = ''
            data['paramML']     = ''
            data['ML_FS']       = ''
            data['paramML_FS']  = ''
            data['estado']      = 'pendiente'

            cantidad +=experimentos
            bd.insertarExperimentos(data, experimentos, instancia[0])

print("------------------------------------------------------------------")
print(f'Se ingresaron {cantidad} experimentos a la base de datos')
print("------------------------------------------------------------------")

