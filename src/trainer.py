import wandb
import numpy as np
import torch
import pickle

import torch.nn.functional as F
import matplotlib.pyplot as plt

from collections import namedtuple
from types import SimpleNamespace
from typing import List, Tuple

from .helpers import DataLoader, DirHelper, Batcher
from .utils.torch_utils import no_grad
from .models import select_model

class Trainer():
    """"base class for running basic transformer classification models"""
    
    def __init__(self, exp_name:str, m_args:namedtuple):
        self.dir = DirHelper(exp_name)
        self.dir.save_args('model_args.json', m_args)
        self.set_up_helpers(m_args)
        
    ############  MAIN TRAIN LOOP  #################################
    
    def set_up_helpers(self, m_args:namedtuple):
        self.model_args = m_args
        self.data_loader = DataLoader(m_args.transformer)
        self.batcher = Batcher(max_len=m_args.max_len)
        
        #temp routing for backward compatibility
        if hasattr(m_args, 'num_classes'):
            self.model = select_model(model_name=m_args.transformer, num_classes=m_args.num_classes)
        else:
            self.model = select_model(model_name=m_args.transformer)

        self.device = m_args.device
                 
    def train(self, args:namedtuple):
        self.dir.save_args('train_args.json', args)
        if args.wandb: self.set_up_wandb(args)
 
        train, dev, test = self.data_loader(args.data_set, args.lim)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)
        best_epoch = (-1, 10000, 0)
        self.to(self.device)
        
        for epoch in range(args.epochs):
            ######  TRAINING  ##############################
            self.model.train()
            self.dir.reset_metrics()
            train_b = self.batcher(data=train, bsz=args.bsz, shuffle=True)
            
            for k, batch in enumerate(train_b, start=1):
                output = self.model_output(batch)

                optimizer.zero_grad()
                output.loss.backward()
                optimizer.step()

                # accuracy logging
                self.dir.update_avg_metrics(loss=output.loss)
                self.dir.update_acc_metrics(hits=output.hits, 
                                            num_preds=output.num_preds)
                
                # print train performance every now and then
                if k%args.print_len == 0:
                    perf = self.dir.print_perf('train', epoch, k)
                    if args.wandb:
                         wandb.log({'epoch':epoch, 'loss':perf.loss, 'acc':perf.acc})
            
            ######  DEV  ##################################
            self.model.eval()
            perf = self.system_eval(dev, epoch, mode='dev')
            if args.wandb:
                wandb.log({"dev_loss":perf.loss, "dev_acc":perf.acc})

            ######  TEST  #################################
            test_perf = self.system_eval(test, epoch, mode='test')
            
            # save performance if best dev performance 
            if perf.acc > best_epoch[2]:
                best_epoch = (epoch, perf.loss, perf.acc)
                if args.save: self.save_model()
                else: self.generate_probs(data=test, data_name=args.data_set)
                
            if epoch - best_epoch[0] >= 3:
                break
             
        self.dir.log(f'best dev epoch: {best_epoch}')
        return test_perf
    
    def model_output(self, batch):
        if getattr(self, 'bias', False):
            return self.bias_model_output(batch)
             
        output = self.model(input_ids=batch.ids, 
                            attention_mask=batch.mask)
                
        loss = F.cross_entropy(output.y, batch.labels)
        
        # return accuracy metrics
        hits = torch.argmax(output.y, dim=-1) == batch.labels
        hits = torch.sum(hits[batch.labels != -100]).item()
        num_preds = torch.sum(batch.labels != -100).item()

        return SimpleNamespace(loss=loss, y=output.y, h=output.h,
                               hits=hits, num_preds=num_preds)

    ############# EVAL METHODS ####################################
    @no_grad
    def system_eval(self, data, epoch:int, mode='dev'):
        self.dir.reset_metrics()         
        batches = self.batcher(data=data, bsz=1, shuffle=False)
        for k, batch in enumerate(batches, start=1):
            output = self.model_output(batch)
            self.dir.update_avg_metrics(loss=output.loss)
            self.dir.update_acc_metrics(hits=output.hits, 
                                        num_preds=output.num_preds)
        perf = self.dir.print_perf(mode, epoch, 0)
        return perf

    def generate_probs(self, data:list, data_name:str):
        probabilties = self._probs(data)
        self.dir.save_probs(probabilties, data_name, mode='test')

    @no_grad
    def _probs(self, data):
        """get model predictions for given data"""
        self.model.eval()
        self.to(self.device)
        eval_batches = self.batcher(data=data, bsz=1, shuffle=False)

        probabilties = {}
        for batch in eval_batches:
            sample_id = batch.sample_id[0]
            output = self.model_output(batch)

            y = output.y.squeeze(0)
            if y.shape and y.shape[-1] > 1:  # Get probabilities of predictions
                y = F.softmax(y, dim=-1)
            probabilties[sample_id] = y.cpu().numpy()
        return probabilties
    
    #############  MODEL UTILS  ###################################
    
    def save_model(self, name:str='base'):
        device = next(self.model.parameters()).device
        self.model.to("cpu")
        torch.save(self.model.state_dict(), 
                   f'{self.dir.abs_path}/models/{name}.pt')
        self.model.to(self.device)

    def load_model(self, name:str='base'):
        self.model.load_state_dict(
            torch.load(self.dir.abs_path + f'/models/{name}.pt'))

    def to(self, device):
        assert hasattr(self, 'model') and hasattr(self, 'batcher')
        self.model.to(device)
        self.batcher.to(device)

    ############  WANDB UTILS  ####################################
    
    def set_up_wandb(self, args:namedtuple):
        wandb.init(project=args.wandb, entity="adian",
                   name=self.dir.exp_name, group=self.dir.base_name, 
                   dir=self.dir.abs_path, reinit=True)

        # save experiment config details
        cfg = {}
        
        cfg['epochs']      = args.epochs
        cfg['bsz']         = args.bsz
        cfg['lr']          = args.lr
        cfg['transformer'] = self.model_args.transformer       
        cfg['data_set']    = args.data_set

        wandb.config.update(cfg) 
        wandb.watch(self.model)
        
    
