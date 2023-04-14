from nilmtk.api import API
from nilmtk.disaggregate import Mean
from nilmtk_contrib.disaggregate import DAE,Seq2Point, Seq2Seq, RNN

epochs = 1
batch_size = 32

config = {
    'power': {
        'mains': ['apparent','active'],
        'appliance': ['apparent','active']
    },
    'sample_rate': 10,

    'appliances': ['microwave', 'dish washer', 'kettle', 'broadband router','computer monitor','laptop computer','external hard disk','computer'],
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
                'path': 'ukdale2.h5',
                'buildings': {
                    1: {
                        'start_time': '2016-11-01',
                        'end_time': '2017-02-01'
                    },
                    2: {
                        'start_time': '2013-05-20',
                        'end_time': '2013-08-20'
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
api_res = API(config)