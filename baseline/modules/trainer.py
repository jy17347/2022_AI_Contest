"""Trainer
"""

import torch
import time

class Trainer():

    def __init__(self,
                 model,
                 optimizer,
                 loss,
                 metric,
                 device,
                 logger, 
                 num_workers,
                 n_epochs,
                 interval=100):
        
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric
        self.device = device
        self.logger = logger
        self.n_epochs = n_epochs
        self.num_workers = num_workers
        self.interval = interval
        self.log_format = "mode: {:s}, epoch: {:4d}/{:4d}, step: {:4d}/{:4d}, loss: {:.6f}, " \
                          "cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h, lr: {:.6f}"

        # History
        self.loss_sum = 0  # Epoch loss sum
        self.loss_mean = 0 # Epoch loss mean
        self.filenames = list()
        self.y = list()
        self.y_preds = list()
        self.score_dict = dict()  # metric score
        self.elapsed_time = 0


        

    def train(self, mode, queue, epoch_time_step, epoch_index):
        """
        x: (batch, width*height)
        y: (batch, 1)
        y_pred_proba: (batch, class)
        """
        begin_time = time.time()
        self.model.train() if mode == 'train' else self.model.eval()

        total_num = 0
        timestep = 0

        num_workers = self.num_workers
        while True:
            inputs, targets, input_lengths, target_lengths = queue.get()
            if inputs.shape[0] == 0:
                if mode == 'test':
                    break
                # Empty feats means closing one loader
                num_workers -= 1
                self.logger.debug('left train_loader: %d' % num_workers)

                if num_workers == 0:
                    break
                else:
                    continue

            # To Device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            input_lengths = input_lengths.to(self.device)
            target_lengths = torch.as_tensor(target_lengths).to(self.device)

            # Foward
            outputs, output_lengths = self.model(inputs, input_lengths)
            loss = self.loss(
                outputs.transpose(0, 1), targets[:, 1:], output_lengths, target_lengths
            )

            # Update
            if mode == 'train':
                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step(self.model)

            elif mode in ['val', 'test']:
                pass

            # History
            timestep += 1
            total_num += int(input_lengths.sum())
            self.loss_sum += loss.item()
            torch.cuda.empty_cache()

            y_hats = outputs.max(-1)[1].detach().cpu().numpy()
            targets = targets.cpu().numpy()
            self.y_preds.append(y_hats)
            self.y.append(targets)

            # Logging
            if timestep % self.interval == 0:
                current_time = time.time()
                elapsed = current_time - begin_time
                cer = self.metric([targets], [y_hats])
                self.logger.info(self.log_format.format(mode, epoch_index, self.n_epochs,
                    timestep, epoch_time_step, loss.item(), cer,
                    elapsed, elapsed / 60.0, elapsed / 3600.0, # Second, Minute, Hour
                    self.optimizer.get_lr()
                ))

        # Epoch history
        self.loss_mean = self.loss_sum / timestep  # Epoch loss mean

        # Metric
        cer = self.metric(self.y, self.y_preds)
        self.score_dict['CER'] = cer

        # Elapsed time
        end_timestamp = time.time()
        self.elapsed_time = end_timestamp - begin_time

    def clear_history(self):
        self.loss_sum = 0
        self.loss_mean = 0
        self.y_preds = list()
        self.y = list()
        self.score_dict = dict()
        self.elapsed_time = 0

