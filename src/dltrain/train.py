import os
import torch
import pandas as pd

from .utils import set_plt, LoggerContext

from torch.utils.data import DataLoader

from abc import ABCMeta, abstractmethod
from tqdm import tqdm

from .options import TrainOptions
from .evaluation import Evaluation
from .checkpoint import CheckPoint

from time import sleep
from datetime import datetime
from matplotlib import pyplot as plt
from os import makedirs

__all__ = [
    'Trainer',
    'SimpleTrainer'
]


class Trainer(metaclass=ABCMeta):
    @abstractmethod
    def run(self, options: TrainOptions):
        pass


class SimpleTrainer(Trainer):

    def run(self, options: TrainOptions):

        logger = LoggerContext()
        logger.info('Starting training')

        set_plt()

        checkpoint: CheckPoint = None
        if options.start_checkpoint is not None:
            checkpoint = torch.load(options.start_checkpoint)
            options = checkpoint.options

            logger.info(f'Loading checkpoint at {options.start_checkpoint}')

        if checkpoint is None:
            makedirs(options.task_name, exist_ok=True)
            makedirs(os.path.join(options.task_name, 'weights'), exist_ok=True)

        logger.info(f'Train folder address at {options.task_name}')

        model = options.model.to(options.device)
        logger.info(f'Model structure:{str(model)}')

        optimizer = options.optimizer_type(model.parameters(), **options.optimizer_parameters)
        logger.info(f'Optimizer parameters is', **options.optimizer_parameters)
        if checkpoint is not None:
            optimizer.load_state_dict(checkpoint.optimizer_state_dict)
            logger.info(f'Using optimizer state dict')

        criterion = options.criterion
        scheduler = options.scheduler_type(optimizer,
                                           **options.scheduler_parameters) if options.scheduler_type is not None else None

        if scheduler is not None and checkpoint is not None:
            scheduler.load_state_dict(checkpoint.scheduler_state_dict)
            logger.info(f'Using scheduler state dict')

        delineator = options.delineator

        train_set, eval_set = delineator.get_train_set(), delineator.get_eval_set()
        train_loader, eval_loader = DataLoader(train_set, batch_size=options.batch_size), \
            DataLoader(eval_set, batch_size=options.batch_size)

        train_evaluation = Evaluation()
        eval_evaluation = Evaluation()

        train_evaluation_handlers = options.train_evaluation_handlers
        eval_evaluation_handlers = options.eval_evaluation_handlers

        train_evaluation.exacts = train_set.get_data()[1].to(options.device)
        eval_evaluation.exacts = eval_set.get_data()[1].to(options.device)

        features_transform = options.features_transform
        targets_transform = options.targets_transform

        forward = options.forward

        total_train_loss, total_eval_loss = [], []
        total_train_evaluation, total_eval_evaluation = {}, {}

        if checkpoint is not None:
            total_train_loss, total_eval_loss = checkpoint.total_train_loss, checkpoint.total_eval_loss
            total_train_evaluation, total_eval_evaluation = (checkpoint.total_train_evaluation,
                                                             checkpoint.total_eval_evaluation)

        start_epoch = 0 if checkpoint is None else checkpoint.epoch

        best_eval_loss = 999 if checkpoint is None else checkpoint.best_eval_loss

        for epoch in range(start_epoch, options.epochs):
            sleep(1)
            start_time = datetime.now()
            with tqdm(total=len(train_loader)) as pbar:
                model.train()
                model.to(options.device)
                for idx, (x, y) in enumerate(train_loader):
                    pbar.set_description(f'Train[{epoch + 1}/{options.epochs}]')

                    x, y = x.to(options.device), y.to(options.device)
                    if features_transform is not None:
                        x = features_transform(x)

                    if targets_transform is not None:
                        y = targets_transform(y)

                    loss = forward(model=model,
                                   criterion=criterion,
                                   x=x, y=y, eval=False,
                                   evaluation=train_evaluation,
                                   optimizer=optimizer)

                    pbar.set_postfix(loss=float(loss))
                    pbar.update(1)

                pbar.set_postfix(loss=float(loss))

            if scheduler is not None:
                scheduler.step()

            model.eval()
            with torch.no_grad():
                for idx, (x, y) in enumerate(eval_loader):

                    x, y = x.to(options.device), y.to(options.device)

                    if features_transform is not None:
                        x = features_transform(x)

                    if targets_transform is not None:
                        y = targets_transform(y)

                    loss = forward(model=model,
                                   criterion=criterion,
                                   x=x, y=y, eval=True,
                                   evaluation=eval_evaluation)

                # 计算训练集与测试集Loss
                train_loss = criterion(train_evaluation.predictions, train_evaluation.exacts)
                eval_loss = criterion(eval_evaluation.predictions, eval_evaluation.exacts)
                total_train_loss.append(float(train_loss))
                total_eval_loss.append(float(eval_loss))

                if eval_loss < best_eval_loss:
                    torch.save(model.cpu(), f'{options.task_name}/weights/best.pt')

                # 分别计算验证Handler
                if train_evaluation_handlers is not None:
                    for key, handler in train_evaluation_handlers.items():
                        val = handler.compute(train_evaluation.predictions, train_evaluation.exacts)
                        if key in total_train_evaluation:
                            total_train_evaluation[key].append(val)
                        else:
                            total_train_evaluation[key] = [val]

                if eval_evaluation_handlers is not None:
                    for key, handler in eval_evaluation_handlers.items():
                        val = handler.compute(eval_evaluation.predictions, eval_evaluation.exacts)
                        if key in total_eval_evaluation:
                            total_eval_evaluation[key].append(val)
                        else:
                            total_eval_evaluation[key] = [val]

                train_evaluation.reset()
                eval_evaluation.reset()

                end_time = datetime.now()
                # Eval the Epoch Results
                logger.info(f'Eval {epoch + 1}/{options.epochs}')
                logger.info(f'Consumes Time:{end_time - start_time}')
                logger.info(f"LR:{optimizer.state_dict()['param_groups'][0]['lr']}")
                logger.info(f'Train Loss: {total_train_loss[-1]:.4f}')

                if train_evaluation_handlers is not None:
                    for key in train_evaluation_handlers:
                        logger.info(f'Train {key}: {total_train_evaluation[key][-1]}')

                logger.info(f'Eval Loss: {total_eval_loss[-1]:.4f}')
                if eval_evaluation_handlers is not None:
                    for key in eval_evaluation_handlers:
                        logger.info(f'Eval {key}: {total_eval_evaluation[key][-1]}')

                if options.save_checkpoint:
                    # Save checkpoint
                    checkpoint = CheckPoint(
                        epoch,
                        optimizer.state_dict(),
                        scheduler.state_dict() if scheduler is not None else None,
                        total_train_loss,
                        total_eval_loss,
                        total_train_evaluation,
                        total_eval_evaluation,
                        options,
                        best_eval_loss
                    )

                    checkpoint.save(options.task_name)

                print()

        # Draw results of total training and evaluation
        plt.figure()
        plt.plot(total_train_loss, label='Epoch Train Loss')
        plt.plot(total_eval_loss, label='Epoch Eval Loss')
        plt.legend()
        plt.savefig(f'{options.task_name}/loss.png', dpi=300)

        torch.save(model.cpu(), f'{options.task_name}/weights/last.pt')

        loss_frame = pd.DataFrame({
            'Train Loss': total_train_loss,
            'Eval Loss': total_eval_loss
        })
        loss_frame.index.name = 'Epoch'
        loss_frame.to_csv(f'{options.task_name}/loss_result.csv', encoding='utf-8')

        if train_evaluation_handlers is not None:
            train_evaluation_frame = pd.DataFrame(
                {
                    **total_train_evaluation
                }
            )
            train_evaluation_frame.index.name = 'Epoch'
            train_evaluation_frame.to_csv(f'{options.task_name}/train_evaluation_result.csv', encoding='utf-8')

        if eval_evaluation_handlers is not None:
            eval_evaluation_frame = pd.DataFrame(
                {
                    **total_eval_evaluation
                }
            )
            eval_evaluation_frame.index.name = 'Epoch'
            eval_evaluation_frame.to_csv(f'{options.task_name}/eval_evaluation_result.csv', encoding='utf-8')

        logger.save(f'{options.task_name}/log.txt')