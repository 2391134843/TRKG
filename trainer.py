import glob
import json
import os
import torch
import shutil

import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
from tqdm import tqdm

from typing import Dict
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

import deepspeed

from doc import Dataset, collate
from utils import AverageMeter, ProgressMeter
from utils import save_checkpoint, delete_old_ckt, report_num_trainable_parameters, move_to_cuda, get_model_obj
from metric import accuracy, ranking_metrics
from models import build_model, ModelOutput
from dict_hub import build_tokenizer
from logger_config import logger

class Trainer:

    def __init__(self, args, ngpus_per_node):
        self.args = args
        self.ngpus_per_node = ngpus_per_node
        self.use_deepspeed = bool(args.deepspeed)
        self.local_rank = args.local_rank if self.use_deepspeed else -1
        self.is_main_process = (self.local_rank <= 0)
        build_tokenizer(args)

        logger.info("=> creating model with LLM backbone")
        self.model = build_model(self.args)
        report_num_trainable_parameters(self.model)

        train_dataset = Dataset(path=args.train_path, task=args.task)
        valid_dataset = Dataset(path=args.valid_path, task=args.task) if args.valid_path else None
        test_dataset = Dataset(path=args.test_path, task=args.task) if args.test_path else None
        num_training_steps = args.epochs * len(train_dataset) // max(args.batch_size * args.gradient_accumulation_steps * max(ngpus_per_node, 1), 1)
        args.warmup = min(args.warmup, num_training_steps // 10)
        logger.info('Total training steps: {}, warmup steps: {}'.format(num_training_steps, args.warmup))
        self.best_metric = None

        if self.use_deepspeed:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=args.lr,
                weight_decay=args.weight_decay)
            scheduler = self._create_lr_scheduler_raw(optimizer, num_training_steps)

            import json as _json
            with open(args.deepspeed, 'r') as _f:
                ds_config = _json.load(_f)
            if ds_config.get("train_micro_batch_size_per_gpu") == "auto":
                ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
            if ds_config.get("gradient_accumulation_steps") == "auto":
                ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
            if ds_config.get("train_batch_size") == "auto":
                ds_config["train_batch_size"] = (
                    ds_config["train_micro_batch_size_per_gpu"]
                    * ds_config["gradient_accumulation_steps"]
                    * ngpus_per_node
                )

            self.model_engine, self.optimizer, _, self.scheduler = deepspeed.initialize(
                args=args,
                model=self.model,
                optimizer=optimizer,
                lr_scheduler=scheduler,
                config=ds_config,
            )
            self.model = self.model_engine
            self.criterion = nn.CrossEntropyLoss().to(self.model_engine.device)

            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, shuffle=True)
            self.train_sampler = train_sampler
        else:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=args.lr,
                weight_decay=args.weight_decay)
            self.scheduler = self._create_lr_scheduler_raw(self.optimizer, num_training_steps)
            self._setup_training()
            self.criterion = nn.CrossEntropyLoss().cuda()
            self.train_sampler = None

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            collate_fn=collate,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True)

        self.train_eval_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            collate_fn=collate,
            num_workers=args.workers,
            pin_memory=True)

        self.valid_loader = None
        if valid_dataset:
            self.valid_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=args.batch_size * 2,
                shuffle=True,
                collate_fn=collate,
                num_workers=args.workers,
                pin_memory=True)

        self.test_loader = None
        if test_dataset:
            self.test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.batch_size * 2,
                shuffle=False,
                collate_fn=collate,
                num_workers=args.workers,
                pin_memory=True)

    def train_loop(self):
        if self.args.use_amp and not self.use_deepspeed:
            self.scaler = torch.cuda.amp.GradScaler()

        self.last_test_metric = None
        self.last_valid_metric = None
        self.best_test_metric = None

        for epoch in range(self.args.epochs):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            self.train_epoch(epoch)
            self._run_eval(epoch=epoch)

        if self.is_main_process:
            self._dump_final_summary()

    def _dump_final_summary(self):
        """Emit a clean banner with all metrics + dump final_metrics.json.

        Called at the end of `train_loop`. Prints both the LATEST test
        metrics (from the last epoch boundary eval) and the BEST-by-valid
        test metrics so paper-time comparisons are easy to grep for.
        """
        last_test = getattr(self, 'last_test_metric', None) or {}
        last_valid = getattr(self, 'last_valid_metric', None) or {}
        best_valid = getattr(self, 'best_metric', None) or {}
        best_test = getattr(self, 'best_test_metric', None) or {}

        def _fmt(m):
            if not m:
                return '(none)'
            return (
                f'Acc@1={m.get("Acc@1", 0):.3f}  Acc@3={m.get("Acc@3", 0):.3f}  '
                f'Hits@1={m.get("Hits@1", 0):.3f}  Hits@3={m.get("Hits@3", 0):.3f}  '
                f'Hits@5={m.get("Hits@5", 0):.3f}  Hits@10={m.get("Hits@10", 0):.3f}  '
                f'MRR={m.get("MRR", 0):.3f}  MR={m.get("MR", 0):.2f}  '
                f'loss={m.get("loss", 0):.3f}'
            )

        banner = '=' * 78
        logger.info(banner)
        logger.info('[FINAL SUMMARY] task=%s  model_dir=%s', self.args.task, self.args.model_dir)
        logger.info('[FINAL SUMMARY] LAST  test : %s', _fmt(last_test))
        logger.info('[FINAL SUMMARY] LAST  valid: %s', _fmt(last_valid))
        logger.info('[FINAL SUMMARY] BEST  test : %s', _fmt(best_test))
        logger.info('[FINAL SUMMARY] BEST  valid: %s', _fmt(best_valid))
        logger.info(banner)

        out_path = os.path.join(self.args.model_dir, 'final_metrics.json')
        try:
            with open(out_path, 'w') as f:
                json.dump({
                    'task': self.args.task,
                    'model_dir': self.args.model_dir,
                    'epochs': self.args.epochs,
                    'last_epoch_test': last_test,
                    'last_epoch_valid': last_valid,
                    'best_epoch_test': best_test,
                    'best_epoch_valid': best_valid,
                }, f, indent=2)
            logger.info('[FINAL SUMMARY] metrics saved to %s', out_path)
        except Exception as e:
            logger.warning('[FINAL SUMMARY] failed to save final_metrics.json: %s', e)

    @torch.no_grad()
    def _run_eval(self, epoch, step=0):
        train_metric = self._eval_loader(self.train_eval_loader, epoch, split_name='train')
        valid_metric = self._eval_loader(self.valid_loader, epoch, split_name='valid')
        test_metric = self._eval_loader(self.test_loader, epoch, split_name='test')

        if not self.is_main_process:
            if self.use_deepspeed:
                dist.barrier()
            return

        summary_parts = []
        for name, m in [('train', train_metric), ('valid', valid_metric), ('test', test_metric)]:
            if m:
                summary_parts.append(
                    f'{name}: MRR={m["MRR"]:.3f} H@1={m["Hits@1"]:.3f} H@3={m["Hits@3"]:.3f} '
                    f'H@5={m.get("Hits@5", 0.0):.3f} H@10={m["Hits@10"]:.3f} MR={m["MR"]:.1f}')
        if summary_parts:
            logger.info('Epoch {} summary | {}'.format(epoch, ' | '.join(summary_parts)))

        if test_metric:
            self.last_test_metric = test_metric
        if valid_metric:
            self.last_valid_metric = valid_metric

        metric_dict = valid_metric if valid_metric else train_metric
        is_best = metric_dict and (self.best_metric is None or metric_dict['Acc@1'] > self.best_metric['Acc@1'])
        if is_best:
            self.best_metric = metric_dict
            if test_metric:
                self.best_test_metric = test_metric

        filename = '{}/checkpoint_{}_{}.mdl'.format(self.args.model_dir, epoch, step)
        if step == 0:
            filename = '{}/checkpoint_epoch{}.mdl'.format(self.args.model_dir, epoch)
        save_checkpoint({
            'epoch': epoch,
            'args': self.args.__dict__,
            'state_dict': self.model.state_dict(),
        }, is_best=is_best, filename=filename)
        delete_old_ckt(path_pattern='{}/checkpoint_*.mdl'.format(self.args.model_dir),
                       keep=self.args.max_to_keep)

        if self.use_deepspeed:
            dist.barrier()

    @torch.no_grad()
    def _eval_loader(self, loader, epoch, split_name='valid') -> Dict:
        """Evaluate on a given data loader (train/valid/test)."""
        if not loader:
            return {}

        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')
        hits1 = AverageMeter('Hits@1', ':6.2f')
        hits3 = AverageMeter('Hits@3', ':6.2f')
        hits5 = AverageMeter('Hits@5', ':6.2f')
        hits10 = AverageMeter('Hits@10', ':6.2f')
        mrr_meter = AverageMeter('MRR', ':6.2f')
        mr_meter = AverageMeter('MR', ':6.2f')

        pbar = tqdm(loader, desc=f'Eval {split_name:>5} Epoch {epoch}', leave=False)
        for i, batch_dict in enumerate(pbar):
            self.model.eval()

            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            batch_size = len(batch_dict['batch_data'])

            outputs = self.model(**batch_dict)
            outputs = get_model_obj(self.model).compute_logits(output_dict=outputs, batch_dict=batch_dict)
            outputs = ModelOutput(**outputs)
            logits, labels = outputs.logits, outputs.labels
            loss = self.criterion(logits, labels)
            losses.update(loss.item(), batch_size)

            acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
            top1.update(acc1.item(), batch_size)
            top3.update(acc3.item(), batch_size)

            rank_m = ranking_metrics(logits, labels)
            hits1.update(rank_m['Hits@1'], batch_size)
            hits3.update(rank_m['Hits@3'], batch_size)
            hits5.update(rank_m.get('Hits@5', 0.0), batch_size)
            hits10.update(rank_m['Hits@10'], batch_size)
            mrr_meter.update(rank_m['MRR'], batch_size)
            mr_meter.update(rank_m['MR'], batch_size)
            pbar.set_postfix(loss=f'{losses.avg:.4f}', MRR=f'{mrr_meter.avg:.3f}', H1=f'{hits1.avg:.3f}')

        pbar.close()
        metric_dict = {'Acc@1': round(top1.avg, 3),
                       'Acc@3': round(top3.avg, 3),
                       'Hits@1': round(hits1.avg, 3),
                       'Hits@3': round(hits3.avg, 3),
                       'Hits@5': round(hits5.avg, 3),
                       'Hits@10': round(hits10.avg, 3),
                       'MRR': round(mrr_meter.avg, 3),
                       'MR': round(mr_meter.avg, 3),
                       'loss': round(losses.avg, 3)}
        logger.info('Epoch {}, {} metric: {}'.format(epoch, split_name, json.dumps(metric_dict)))
        return metric_dict

    def train_epoch(self, epoch):
        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')
        inv_t = AverageMeter('InvT', ':6.2f')
        hits1 = AverageMeter('Hits@1', ':6.2f')
        hits3 = AverageMeter('Hits@3', ':6.2f')
        hits5 = AverageMeter('Hits@5', ':6.2f')
        hits10 = AverageMeter('Hits@10', ':6.2f')
        mrr_meter = AverageMeter('MRR', ':6.2f')
        mr_meter = AverageMeter('MR', ':6.2f')
        progress = ProgressMeter(
            len(self.train_loader),
            [losses, inv_t, top1, top3, hits1, hits3, hits5, hits10, mrr_meter, mr_meter],
            prefix="Epoch: [{}]".format(epoch))

        if not self.use_deepspeed:
            self.optimizer.zero_grad()
        disable_tqdm = not self.is_main_process
        train_pbar = tqdm(self.train_loader, desc=f'Train Epoch {epoch}', leave=True, disable=disable_tqdm)
        for i, batch_dict in enumerate(train_pbar):
            self.model.train()

            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            batch_size = len(batch_dict['batch_data'])

            if self.use_deepspeed:
                outputs = self.model(**batch_dict)
            elif self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch_dict)
            else:
                outputs = self.model(**batch_dict)
            outputs = get_model_obj(self.model).compute_logits(output_dict=outputs, batch_dict=batch_dict)
            outputs = ModelOutput(**outputs)
            logits, labels = outputs.logits, outputs.labels
            assert logits.size(0) == batch_size

            loss = self.criterion(logits, labels)
            loss += self.criterion(logits[:, :batch_size].t(), labels)

            acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
            top1.update(acc1.item(), batch_size)
            top3.update(acc3.item(), batch_size)
            inv_t.update(outputs.inv_t.item() if torch.is_tensor(outputs.inv_t) else outputs.inv_t, 1)

            if self.use_deepspeed:
                self.model_engine.backward(loss)
                self.model_engine.step()
                losses.update(loss.item(), batch_size)
            else:
                loss = loss / self.args.gradient_accumulation_steps
                losses.update(loss.item() * self.args.gradient_accumulation_steps, batch_size)

                if self.args.use_amp:
                    self.scaler.scale(loss).backward()
                    if (i + 1) % self.args.gradient_accumulation_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        self.scheduler.step()
                else:
                    loss.backward()
                    if (i + 1) % self.args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.scheduler.step()

            rank_m = ranking_metrics(logits, labels)
            hits1.update(rank_m['Hits@1'], batch_size)
            hits3.update(rank_m['Hits@3'], batch_size)
            hits5.update(rank_m.get('Hits@5', 0.0), batch_size)
            hits10.update(rank_m['Hits@10'], batch_size)
            mrr_meter.update(rank_m['MRR'], batch_size)
            mr_meter.update(rank_m['MR'], batch_size)

            if i % self.args.print_freq == 0 and self.is_main_process:
                progress.display(i)
                if not disable_tqdm:
                    cur_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.args.lr
                    train_pbar.set_postfix(
                        loss=f'{losses.avg:.4f}', MRR=f'{mrr_meter.avg:.3f}',
                        H1=f'{hits1.avg:.3f}', H10=f'{hits10.avg:.3f}',
                        lr=f'{cur_lr:.2e}')
            if (i + 1) % self.args.eval_every_n_step == 0:
                self._run_eval(epoch=epoch, step=i + 1)

        train_pbar.close()
        if self.is_main_process:
            cur_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.args.lr
            logger.info('Learning rate: {}'.format(cur_lr))

    def _setup_training(self):
        if torch.cuda.is_available():
            self.model.cuda()
            logger.info('Using single GPU for training')
        else:
            logger.info('No gpu will be used')

    def _create_lr_scheduler_raw(self, optimizer, num_training_steps):
        if self.args.lr_scheduler == 'linear':
            return get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.args.warmup,
                num_training_steps=num_training_steps)
        elif self.args.lr_scheduler == 'cosine':
            return get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.args.warmup,
                num_training_steps=num_training_steps)
        else:
            assert False, 'Unknown lr scheduler: {}'.format(self.args.lr_scheduler)
