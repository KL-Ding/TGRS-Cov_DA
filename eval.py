import argparse
import os
import os.path as osp
import pprint
import warnings

from torch.utils import data

# from UDA.model.deeplabv2 import get_deeplab_v2
from UDA.model.CO_Litev2 import get_deeplab_v2
from UDA.dataset.targetdata import TargetDataSet
from UDA.domain_adaptation.config import cfg, cfg_from_file
from UDA.domain_adaptation.eval_UDA import evaluate_domain_adaptation

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")

def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for evaluation")
    parser.add_argument('--cfg', type=str, default="C:\\Users\\ZSY\\Desktop\\Cov-DA\\intrada_trained.yml",
                        help='optional config file', )
    parser.add_argument("--exp-suffix", type=str, default=None,
                        help="optional experiment suffix")
    return parser.parse_args()


# get args
args = get_arguments()
config_file = args.cfg

assert config_file is not None, 'Failed to load cfg file'
cfg_from_file(config_file)
cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}_{cfg.TRAIN.MODEL}_{cfg.TRAIN.DA_METHOD}'
if cfg.TEST.SNAPSHOT_DIR[0] == '':
    cfg.TEST.SNAPSHOT_DIR[0] = osp.join(cfg.EXP_ROOT_SNAPSHOT, cfg.EXP_NAME)
    os.makedirs(cfg.TEST.SNAPSHOT_DIR[0], exist_ok=True)

print('Args.cfg:')
pprint.pprint(cfg)

# load models
models = []
num_models = len(cfg.TEST.MODEL)

# test the best model
for i in range(num_models):
    model = get_deeplab_v2(num_classes = cfg.NUM_CLASSES, multi_level = cfg.TEST.MULTI_LEVEL[i])
    models.append(model)
# if os.environ.get('ADVENT_DRY_RUN', '0') == '1':
#     return


# dataloaders
test_dataset = TargetDataSet(root=cfg.DATA_DIRECTORY_TARGET,
                                     list_path='./UDA/dataset/cityscapes_list/{}.txt',
                                     set='val',
                                     info_path=cfg.TEST.INFO_TARGET,
                                     crop_size=cfg.TEST.INPUT_SIZE_TARGET,
                                     mean=cfg.TEST.IMG_MEAN,
                                     labels_size=(512, 512))
test_loader = data.DataLoader(test_dataset,
                                  batch_size=1,
                                  num_workers=4,
                                  shuffle=False,
                                  pin_memory=True)

evaluate_domain_adaptation(models, test_loader, cfg)







