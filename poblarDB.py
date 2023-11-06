from BD.sqlite import BD
import json

bd = BD()


scp = True
ben = False
mhs = ['PSO','GWO','RSA']
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
binarizaciones = []

if scp:
    # poblar ejecuciones SCP
    instancias = bd.obtenerInstancias(f'''
                                      'scp41','scp51','scp61','scpa1','scpb1','scpc1','scpd1','scpnre1','scpnrf1'
                                      ''')
    print(instancias)
    iteraciones = 500
    experimentos = 31
    poblacion = 20
    for instancia in instancias:

        for mh in mhs:
            binarizaciones = ['Z4-STD','Z4-COM','Z4-ELIT','X4-STD','X4-COM','X4-ELIT']
            for binarizacion in binarizaciones:
                data = {}
                data['MH']          = mh
                data['paramMH']     = f'iter:{str(iteraciones)},pop:{str(poblacion)},DS:{binarizacion},repair:complex,cros:0.9;mut:0.20'
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
                                      "F3","F4"
                                      ''')
    iteraciones = 500
    experimentos = 5 
    poblacion = 40
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

