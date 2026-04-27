import argparse
import logging
import time

LOG = logging.getLogger('main')
__all__ = []


def create_parser():
    parser = argparse.ArgumentParser(description='DTI analysis')

    parser.add_argument('--data-path', default='D:\\Datasets\\HCP_S1200', type=str)
    parser.add_argument('--input-features', '--INPUT-FEATURES', dest='INPUT_FEATURES', nargs='+',
                        default=['Num_Points'], type=str)
    parser.add_argument('--hemispheres', '--HEMISPHERES', dest='HEMISPHERES', nargs='+',
                        default=['right-hemisphere', 'left-hemisphere', 'commissural'], type=str)
    parser.add_argument('--target', default='sex', type=str)
    parser.add_argument('--fold-num', '--FOLD-NUM', dest='FOLD_NUM', default=5, type=int)

    parser.add_argument('--model', '--MODEL', dest='MODEL', default='1D-CNN', type=str)
    parser.add_argument('--num-classes', '--NUM_CLASSES', dest='NUM_CLASSES', default=2, type=int)
    parser.add_argument('--load-path', '--LOAD-PATH', dest='LOAD_PATH',
                        default='C:\\Users\\admin\\PycharmProjects\\dti\\LOG\\{}.pkl', type=str)

    parser.add_argument('--ensemble', default=True, type=bool)

    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch-size', '--BATCH-SIZE', default=8, type=int)
    parser.add_argument('--num-workers', '--NUM-WORKERS', default=2, type=int)
    parser.add_argument('--display-batch', '--DISPLAY-BATCH', default=50, type=int)
    parser.add_argument('--loss', '--LOSS', dest='LOSS', default='CE', type=str)
    parser.add_argument('--lr', '--LR', dest='LR', default=0.01, type=float)
    parser.add_argument('--l2', '--L2', dest='L2', default=0, type=float)

    parser.add_argument('--record-name', '--RECORD-NAME', dest='RECORD_NAME',
                        default='{}'.format(time.strftime("%Y%m%d-%H%M%S", time.localtime())), type=str)
    parser.add_argument('--demographics-csv', '--DEMOGRAPHICS-CSV', default='', type=str)
    parser.add_argument('--subject-col', '--SUBJECT-COL', default='SUB_ID', type=str)
    parser.add_argument('--label-col', '--LABEL-COL', default='', type=str)
    parser.add_argument('--dx-col', '--DX-COL', default='DX_GROUP', type=str)

    return parser
