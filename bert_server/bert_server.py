import os

from bert_serving.server import BertServer
from bert_serving.server.helper import get_args_parser

# Avoid conflict
USE_CPU = True
arg_list = ['-model_dir', os.path.join(os.getcwd(), 'uncased_L-12_H-768_A-12'),
            '-port', '23333',
            '-num_worker=1']
if USE_CPU:
    arg_list.append('-cpu')
args = get_args_parser().parse_args(arg_list)

if __name__ == '__main__':
    server = BertServer(args)
    server.start()
