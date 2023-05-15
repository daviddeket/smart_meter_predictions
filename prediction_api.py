## The birt.py file has to be replaced bert.py -> nilmtk-contrib\nilmtk_contrib\disaggregate\bert.py

from nilmtk.api import API
from nilmtk.disaggregate import Mean
from nilmtk_contrib.disaggregate import DAE,Seq2Point, Seq2Seq, RNN, BERT

epochs = 0
batch_size = 32

def get_modelpaths(model_name,appliances):

    base_path = r'C:\Users\stefa\OneDrive - FHWN\Privat\Studium\MIT_2-Semester\case_study\smart_meter_predictions\models'

    load_model_paths = {}
    for appliance in appliances:
        load_model_paths[appliance] = f'{base_path}\\{model_name}\\{model_name}-temp-weights-{appliance}-epoch0.h5'

    return load_model_paths


appliances = ['microwave', 'dish washer', 'kettle', 'broadband router','computer monitor','laptop computer','external hard disk','computer']

config = {
    'power': {
        'mains': ['apparent','active'],
        'appliance': ['apparent','active']
    },
    'sample_rate': 20,
    # 'chunk_size':20,

    'appliances': appliances,
    'methods': {
        'RNN': RNN({
            'n_epochs': 0,
            'batch_size': batch_size,
            'load_model_path': get_modelpaths('rnn',appliances)
        }),
        'DAE': DAE({
            'n_epochs': 0,
            'batch_size': batch_size,
            'load_model_path': get_modelpaths('dae',appliances)
        }),
        'Seq2Point': Seq2Point({
            'n_epochs': 0,
            'batch_size': batch_size,
            'load_model_path': get_modelpaths('seq2point',appliances)
        }),
        'Seq2Seq': Seq2Seq({
            'n_epochs': 0,
            'batch_size': batch_size,
            'load_model_path': get_modelpaths('seq2seq',appliances)
        }),
        'Mean': Mean({}),
    },
    'train':{
        'datasets': {
            'Datport': {
                'path': 'ukdale2.h5',
                'buildings': {
                    1: {
                        'start_time': '2017-02-15',
                        'end_time': '2017-03-08'
                    },
                    2: {
                        'start_time': '2013-09-15',
                        'end_time': '2013-10-03'
                    },
                }
            }
        }
    },

    'test': {
        'datasets': {
            'Datport': {
                'path': 'ukdale2.h5',
                'buildings': {
                    1: {
                        'start_time': '2017-02-15',
                        'end_time': '2017-03-08'
                    },
                    2: {
                        'start_time': '2013-09-15',
                        'end_time': '2013-10-03'
                    },
                }
            }
        },
        'metrics':['mae', 'rmse']
    }
}
###############################################
config_Bert = {
    'power': {
        'mains': ['apparent','active'],
        'appliance': ['apparent','active']
    },
    'sample_rate': 30,

    'appliances': appliances,
    'methods': {
        'BERT': BERT({
            'n_epochs': 0,
            'batch_size': batch_size,
            'load_model_path': get_modelpaths('BERT',appliances)
        }),
        'Mean': Mean({}),
    },
    'train':{
        'datasets': {
            'Datport': {
                'path': 'ukdale2.h5',
                'buildings': {
                    1: {
                        'start_time': '2017-02-15',
                        'end_time': '2017-03-08'
                    },
                    2: {
                        'start_time': '2013-09-15',
                        'end_time': '2013-10-03'
                    },
                }
            }
        }
    },

    'test': {
        'datasets': {
            'Datport': {
                'path': 'ukdale2.h5',
                'buildings': {
                    1: {
                        'start_time': '2017-02-15',
                        'end_time': '2017-03-08'
                    },
                    2: {
                        'start_time': '2013-09-15',
                        'end_time': '2013-10-03'
                    },
                }
            }
        },
        'metrics':['mae', 'rmse']
    }
}




if __name__ == '__main__':

    import pickle
    bert = True

    if bert:
        api_res_bert = API(config_Bert)

        # with open('api_bert.pickle','w') as f:
        #     pickle.dump(api_res_bert(),f)

    else: 
        api_res = API(config)
        print(api_res.errors)

        # with open('api.pickle','wb') as f:
            # pickle.dump(api_res(),f)
