import logging
import time
from ruamel import yaml
import torch
import os
import torch.distributed as dist


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    # args.dist_backend = 'nccl' 
    args.dist_backend = 'gloo' 
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()


def cleanup():
    dist.destroy_process_group()


def is_dist_avail_and_initialized():
    """Check whether it is distributed environment"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value
    
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def Prepare_logger(args, eval=False):
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(0)
    logger.addHandler(handler)

    date = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    logfile = args.snapshot_pref+date+'.log' if not eval else args.snapshot_pref + f'/{date}-Eval.log'
    file_handler = logging.FileHandler(logfile, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_configs(dataset):
    data = yaml.load(open('./configs/dataset_cfg.yaml'))
    return data[dataset]


def get_and_save_args(parser):
    args = parser.parse_args()
    # dataset = args.dataset

    default_config = yaml.load(open('./configs/default_config.yaml', 'r'), Loader=yaml.RoundTripLoader)
    current_config = vars(args)
    for k, v in current_config.items():
        if k in default_config:
            if (v != default_config[k]) and (v is not None):
                print(f"Updating:  {k}: {default_config[k]} (default) ----> {v}")
                default_config[k] = v
    yaml.dump(default_config, open('./current_configs.yaml', 'w'), indent=4, Dumper=yaml.RoundTripDumper)
    return default_config

def load_backbone_state_dict(args, logger, model):
    if os.path.isfile(args.backbone):
        logger.info(f"\nLoading Vggsound Backbone: {args.backbone}\n")
        model_tmp = torch.load(args.backbone)
        # modules = ['module.spatial_channel_att', 'module.v_fc', 'module.video_encoder', 'module.video_decoder', 'module.audio_encoder', 'module.audio_decoder']
        # 'module.video_encoder', 'module.audio_encoder': 79.13%
 
        modules = ['module.video_encoder', 'module.audio_encoder']
        module_list = []
        for module in modules:
            for k in model_tmp.keys():
                if module in k:
                    module_list.append(k)
                    
        pretrained_dict = {k: v for k, v in model_tmp.items() if k in module_list}
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
    elif args.backbone != "" and (not os.path.isfile(args.backbone)):
        raise FileNotFoundError
    
    return model