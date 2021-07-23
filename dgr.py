import abc
import utils
from tqdm import tqdm
import torch
from torch import nn
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data import ConcatDataset


# ============
# Base Classes
# ============

class GenerativeMixin(object):
    """Mixin which defines a sampling iterface for a generative model."""
    def sample(self, size):
        raise NotImplementedError


class BatchTrainable(nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract base class which defines a generative-replay-based training
    interface for a model.
    """
    @abc.abstractmethod
    def train_a_batch(self, x, y, x_=None, y_=None, importance_of_new_task=.5):
        raise NotImplementedError


# ==============================
# Deep Generative Replay Modules
# ==============================

class Generator(GenerativeMixin, BatchTrainable):
    """Abstract generator module of a scholar module"""


class Solver(BatchTrainable):
    """Abstract solver module of a scholar module"""
    def __init__(self):
        super().__init__()
        self.optimizer = None
        self.criterion = None

    @abc.abstractmethod
    def forward(self, x):
        raise NotImplementedError

    def solve(self, x):
        #print("in solve of SOlver")
        scores = self(x)
        #print("scores in solve")
        #print(scores)
        _, predictions = torch.max(scores, 1)
        #print("print prediction")
        #print(predictions)
        return predictions

    def train_a_batch(self, x, y, x_=None, y_=None, importance_of_new_task=.5):
        #assert x_ is None or x.size() == x_.size()
        #assert y_ is None or y.size() == y_.size()

        # clear gradients.
        batch_size = x.size(0)
        self.optimizer.zero_grad()

        # run the model on the real data.
        real_scores = self.forward(x)
        #print("real_scores and y size")
        #print(real_scores.size(), y.size())
        # print("this is y")
        # print(y)
        # print("this is y ends")
        
        
        real_scores1, real_scores2 = torch.max(real_scores,1)
        #real_scores2 = real_scores2.view(:,1)
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        y = y.view(-1) #.to(DEVICE)
        #print("y and real_scores2")
        #print(y, real_scores2)
        criterion = nn.CrossEntropyLoss() #.to(DEVICE)
        real_loss = criterion(real_scores, y)
        
        _, real_predicted = real_scores.max(1)
        real_prec = (y == real_predicted).sum().data[0] / batch_size

        # run the moduleel on the replayed data.
        if x_ is not None and y_ is not None:
            replay_scores = self.forward(x_)
            replay_loss = self.criterion(replay_scores, y_)
            _, replay_predicted = replay_scores.max(1)
            replay_prec = (y_ == replay_predicted).sum().data[0] / batch_size

            # calculate joint loss of real data and replayed data.
            loss = (
                importance_of_new_task * real_loss +
                (1-importance_of_new_task) * replay_loss
            )
            precision = (real_prec + replay_prec) / 2
        else:
            loss = real_loss
            precision = real_prec

        loss.backward()
        self.optimizer.step()
        return {'loss': loss.data[0], 'precision': precision}

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_criterion(self, criterion):
        self.criterion = criterion


class Scholar(GenerativeMixin, nn.Module):
    """Scholar for Deep Generative Replay"""
    def __init__(self, label, generator, solver):
        super().__init__()
        self.label = label
        self.generator = generator
        self.solver = solver

    def train_with_replay(
            self, dataset, scholar=None, previous_datasets=None,
            importance_of_new_task=.5, batch_size=32,
            generator_iterations=2000,
            generator_training_callbacks=None,
            solver_iterations=1000,
            solver_training_callbacks=None,
            collate_fn=None):
        # scholar and previous datasets cannot be given at the same time.
        #print("inside train_with_replay of schoalrs")
        mutex_condition_infringed = all([
            scholar is not None,
            bool(previous_datasets)
        ])
        assert not mutex_condition_infringed, (
            'scholar and previous datasets cannot be given at the same time'
        )

        #print("inside train_with_replay of schoalrs before _train_batch_trainable_with_replay")
        # train the generator of the scholar.
        self._train_batch_trainable_with_replay(
            self.generator, dataset, scholar,
            previous_datasets=previous_datasets,
            importance_of_new_task=importance_of_new_task,
            batch_size=batch_size,
            iterations=generator_iterations,
            training_callbacks=generator_training_callbacks,
            collate_fn=collate_fn,
        )
        #print("inside train_with_replay of schoalrs before _train_batch_trainable_with_replay 2")
        # train the solver of the scholar.
        self._train_batch_trainable_with_replay(
            self.solver, dataset, scholar,
            previous_datasets=previous_datasets,
            importance_of_new_task=importance_of_new_task,
            batch_size=batch_size,
            iterations=solver_iterations,
            training_callbacks=solver_training_callbacks,
            collate_fn=collate_fn,
        )

    @property
    def name(self):
        return self.label

    def sample(self, size):
        x = self.generator.sample(size)
        y = self.solver.solve(x)
        #print("in sample print y and y.data")
        #print(y)
        #print(y.data)
        return x.data, y #y.data

    def _train_batch_trainable_with_replay(
            self, trainable, dataset, scholar=None, previous_datasets=None,
            importance_of_new_task=.5, batch_size=32, iterations=1000,
            training_callbacks=None, collate_fn=None):
        # do not train the model when given non-positive iterations.
        if iterations <= 0:
            return

        # create data loaders.
        data_loader = iter(utils.get_data_loader(
            dataset, batch_size, cuda=self._is_on_cuda(),
            collate_fn=collate_fn,
        ))
        data_loader_previous = iter(utils.get_data_loader(
            ConcatDataset(previous_datasets), batch_size,
            cuda=self._is_on_cuda(), collate_fn=collate_fn,
        )) if previous_datasets else None
        print(data_loader_previous)

        # define a tqdm progress bar.
        progress = tqdm(range(1, iterations+1))

        for batch_index in progress:
            # decide from where to sample the training data.
            from_scholar = scholar is not None
            from_previous_datasets = bool(previous_datasets)
            #print(" printing scholar and previous datasets in dgr _train_batch_trainable_with_replay")
            #print(from_scholar,from_previous_datasets)
            cuda = self._is_on_cuda()

            # sample the real training data.
            x, y = next(data_loader)
            x = Variable(x).cuda() if cuda else Variable(x)
            y = Variable(y).cuda() if cuda else Variable(y)

            inscholar = 0
            # sample the replayed training data.
            if from_previous_datasets:
                #print("in dgr from_previous_datasets")
                #x_, y_ = next(data_loader_previous)
                for a, b in data_loader_previous:
                    x_,y_ = a ,b 
            elif from_scholar:
                inscholar = 1
                print("in dgr from_scholar")
                x_, y_ = scholar.sample(batch_size)
                #print("in from scholar, print y_")
                #print(y_)
            else:
                #print("in dgr both x_ andy y_ are none")
                x_ = y_ = None

            if x_ is not None and y_ is not None:
                x_ = Variable(x_).cuda() if cuda else Variable(x_)
                if inscholar == 1:
                    y_ = y_.cuda()
                else:
                    y_ = Variable(y_).cuda() if cuda else Variable(y_)

            # train the model with a batch.
            result = trainable.train_a_batch(
                x, y, x_=x_, y_=y_,
                importance_of_new_task=importance_of_new_task
            )

            # fire the callbacks on each iteration.
            for callback in (training_callbacks or []):
                callback(trainable, progress, batch_index, result)

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda
