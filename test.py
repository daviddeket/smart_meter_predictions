from nilmtk.api import API
from nilmtk.disaggregate import Mean
from nilmtk_contrib.disaggregate import DAE,Seq2Point, Seq2Seq, RNN

epochs = 1
batch_size = 128

config = {
    'power': {
        'mains': ['apparent','active'],
        'appliance': ['apparent','active']
    },
    'sample_rate': 10,

    'appliances': ['microwave'],
    'methods': {
        'RNN':RNN({'n_epochs':epochs,'batch_size':batch_size}),
        'DAE':DAE({'n_epochs':epochs,'batch_size':batch_size}),
        'Seq2Point':Seq2Point({'n_epochs':epochs,'batch_size':batch_size}),
        'Seq2Seq':Seq2Seq({'n_epochs':epochs,'batch_size':batch_size}),
        'Mean': Mean({}),
    },
    'train': {
        'datasets': {
            'Dataport': {
                'path': 'C:/Users/david/Desktop/smart_meter_predictions/ukdale2.h5',
                'buildings': {
                    1: {
                        'start_time': '2015-01-01',
                        'end_time': '2015-02-01'
                    }
                }
            }
        }
    },
    'test': {
        'datasets': {
            'Datport': {
                'path': 'C:/Users/david/Desktop/smart_meter_predictions/ukdale2.h5',
                'buildings': {
                    1: {
                        'start_time': '2015-02-01',
                        'end_time': '2015-02-02'
                    }
                }
            }
        },
        'metrics':['mae', 'rmse']
    }
}

api_res = API(config)

#%%
