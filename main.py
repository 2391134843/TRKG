import torch
import json
import torch.backends.cudnn as cudnn

from config import args
from trainer import Trainer
from logger_config import logger

def main():
    ngpus_per_node = torch.cuda.device_count()
    cudnn.benchmark = True

    if args.deepspeed and args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)

    is_main = (args.local_rank <= 0)
    if is_main:
        logger.info("Use {} gpus for training".format(ngpus_per_node))
        logger.info('Args={}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))

    trainer = Trainer(args, ngpus_per_node=ngpus_per_node)
    trainer.train_loop()

if __name__ == '__main__':
    main()
