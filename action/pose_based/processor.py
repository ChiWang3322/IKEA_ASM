import logging, torch, numpy as np
from tqdm import tqdm
from time import time

from . import utils as U
from .initializer import Initializer


class Processor(Initializer):
    def __init__(self, args):
        super().__init__(args)
        self.running_tloss = 0
        self.running_vloss = 0
        self.running_tacc = []
        self.running_vacc = []
        self.max_steps = args.n_epochs
        self.train_num_batch = len(self.train_dataloader)
        self.test_num_batch = len(self.test_dataloader)
        self.refine_flag = True
        self.best_acc = 0
        self.steps = 0
        self.epoch_index = 0
        self.test_batchind = -1
        self.test_fraction_done = 0.0
        self.test_enum = enumerate(test_dataloader, 0)
        self.tot_loss = 0.0
        self.num_iter = 0
    def train_one_epoch(self):

        self.optimizer.zero_grad()
        self.running_tloss = 0
        self.running_vloss = 0
        self.running_tacc = []
        self.running_vacc = []
        self.tot_loss = 0.0
        self.num_iter = 0
        self.test_batchind = -1
        self.test_fraction_done = 0.0
        self.test_enum = enumerate(self.test_dataloader, 0)

        for train_batchind, data in enumerate(tqdm(self.train_dataloader)):
            self.num_iter += 1
            # get the inputs
            inputs, labels, vid_idx, frame_pad = data
            if self.arch == 'EGCN':
                inputs = batch_multi_input(inputs)
            # Wrap them in Variable
            inputs = Variable(inputs.cuda(), requires_grad=True)
            labels = Variable(labels.cuda())
            labels = torch.argmax(labels, dim=1)
            if self.arch == 'EGCN':
                logits, _ = self.model(inputs)
            else:
                logits = self.model(inputs)
            #print('output size:', logits.size())
            
            # print('logits size:', logits.size())
            # Number of frames in one clip
            if self.arch == 'EGCN':
                t = inputs.size(3)
            else:
                t = inputs.size(2)
            
            per_frame_logits = torch.nn.functional.interpolate(logits.unsqueeze(-1), t, mode='linear', align_corners=True)
            # print('size per frame logits:', per_frame_logits.size())
            # print('output for T', per_frame_logits)
            #print('per frame logits:',per_frame_logits.size())
            #print('grount truth size:', labels.size())
            probs = torch.nn.functional.softmax(per_frame_logits, dim=1)
            # Use probs to do crossentropy or logits?

            loss = nn.CrossEntropyLoss()(per_frame_logits, labels)

            tot_loss += loss.item()
            self.running_tloss += loss.item()
            loss.backward()

            acc = utils.accuracy_v2(torch.argmax(per_frame_logits, dim=1), labels)

            self.train_acc.append(acc.item())
            train_fraction_done = (train_batchind + 1) / self.train_num_batch
            # Every num_steps_per_update do optimizer.step()
            if (self.num_iter == self.num_steps_per_update or train_batchind == len(self.train_dataloader)-1) :

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.num_iter = 0
                self.tot_loss = 0.
            
            # Evaluate one batch
            if self.test_fraction_done <= train_fraction_done and test_batchind + 1 < self.test_num_batch:
            #    print('----------Evaluation----------')
                self.model.train(False)  # Set model to evaluate mode
                test_batchind, data = next(self.test_enum)
                inputs, labels, _, _ = data
                if self.arch == 'EGCN':
                    #print('Generating joint, velocity, bone data...')
                    inputs = batch_multi_input(inputs)
                    #print('Generating new information successfully!------------')

                # wrap them in Variable
                inputs = Variable(inputs.cuda(), requires_grad=True)
                labels = Variable(labels.cuda())
                labels = torch.argmax(labels, dim=1)

                with torch.no_grad():
                    if self.arch == 'EGCN':
                        logits, _ = self.model(inputs)
                        t = inputs.size(3)
                    else:
                        logits = self.model(inputs)
                        t = inputs.size(2)
                       
                    per_frame_logits = torch.nn.functional.interpolate(logits.unsqueeze(-1), t, mode='linear',
                                                                       align_corners=True)
                    # probs = torch.nn.functional.softmax(per_frame_logits, dim=1)

                    loss = nn.CrossEntropyLoss()(per_frame_logits, labels)
                    self.running_vloss += loss.item()
                    acc = utils.accuracy_v2(torch.argmax(per_frame_logits, dim=1), labels)
                    self.val_acc.append(acc.item())
                self.test_fraction_done = (test_batchind + 1) / self.test_num_batch
                self.model.train(True)
        # # One epoch finished, report loss and accuracy
        # if self.steps % 100 == 0:
        #     torch.save({"model_state_dict": self.model.state_dict(),
        #                 "optimizer_state_dict": self.optimizer.state_dict(),
        #                 "lr_state_dict": self.lr_sched.state_dict()},
        #                self.logdir + str(self.steps).zfill(6) + '.pt')

        # # remember best prec@1 and save checkpoint
        # is_best = np.mean(self.val_acc) > self.best_acc
        # self.best_acc = max(acc, self.best_acc)
        # if (is_best):
        #     print('Best accuracy achieved, updating the model...')
        #     print('Best val_acc:', np.mean(self.val_acc))
        #     model_tmp = copy.deepcopy(self.model.state_dict())
        #     self.model.load_state_dict(model_tmp)
        #     torch.save({"model_state_dict": self.model.state_dict(),
        #                 "optimizer_state_dict": self.optimizer.state_dict(),
        #                 "lr_state_dict": self.lr_sched.state_dict()}, os.path.join(self.logdir, 'best_classifier.pth'))


    def eval(self):
        self.model.eval()
        start_eval_time = time()
        with torch.no_grad():
            num_top1, num_top5 = 0, 0
            num_sample, eval_loss = 0, []
            cm = np.zeros((self.num_class, self.num_class))
            eval_iter = self.eval_loader if self.no_progress_bar else tqdm(self.eval_loader, dynamic_ncols=True)
            for num, (x, y, _) in enumerate(eval_iter):

                # Using GPU
                x = x.float().to(self.device)
                y = y.long().to(self.device)

                # Calculating Output
                out, _ = self.model(x)

                # Getting Loss
                loss = self.loss_func(out, y)
                eval_loss.append(loss.item())

                # Calculating Recognition Accuracies
                num_sample += x.size(0)
                reco_top1 = out.max(1)[1]
                num_top1 += reco_top1.eq(y).sum().item()
                reco_top5 = torch.topk(out,5)[1]
                num_top5 += sum([y[n] in reco_top5[n,:] for n in range(x.size(0))])

                # Calculating Confusion Matrix
                for i in range(x.size(0)):
                    cm[y[i], reco_top1[i]] += 1

                # Showing Progress
                if self.no_progress_bar and self.args.evaluate:
                    logging.info('Batch: {}/{}'.format(num+1, len(self.eval_loader)))

        # Showing Evaluating Results
        acc_top1 = num_top1 / num_sample
        acc_top5 = num_top5 / num_sample
        eval_loss = sum(eval_loss) / len(eval_loss)
        eval_time = time() - start_eval_time
        eval_speed = len(self.eval_loader) * self.eval_batch_size / eval_time / len(self.args.gpus)
        logging.info('Top-1 accuracy: {:d}/{:d}({:.2%}), Top-5 accuracy: {:d}/{:d}({:.2%}), Mean loss:{:.4f}'.format(
            num_top1, num_sample, acc_top1, num_top5, num_sample, acc_top5, eval_loss
        ))
        logging.info('Evaluating time: {:.2f}s, Speed: {:.2f} sequnces/(second*GPU)'.format(
            eval_time, eval_speed
        ))
        logging.info('')
        if self.scalar_writer:
            self.scalar_writer.add_scalar('eval_acc', acc_top1, self.global_step)
            self.scalar_writer.add_scalar('eval_loss', eval_loss, self.global_step)

        torch.cuda.empty_cache()
        return acc_top1, acc_top5, cm

    def start(self):
        start_time = time()
        best_state = {'acc_top1':0, 'acc_top5':0, 'cm':0}
        # if self.args.resume:
        #     logging.info('Loading checkpoint ...')
        #     checkpoint = U.load_checkpoint(self.args.work_dir)
        #     self.model.module.load_state_dict(checkpoint['model'])
        #     self.optimizer.load_state_dict(checkpoint['optimizer'])
        #     self.scheduler.load_state_dict(checkpoint['scheduler'])
        #     start_epoch = checkpoint['epoch']
        #     best_state.update(checkpoint['best_state'])
        #     self.global_step = start_epoch * len(self.train_loader)
        #     logging.info('Start epoch: {}'.format(start_epoch+1))
        #     logging.info('Best accuracy: {:.2%}'.format(best_state['acc_top1']))
        #     logging.info('Successful!')
        #     logging.info('')

        # Training
        logging.info('Starting training ...')
        for steps in range(self.max_steps):

            # Training
            self.train(epoch)
            # One epoch finised, report acc and loss
            print('INFO: Training----------------')
            print('Loss: {}, Accuracy: {}'.format(self.running_tloss/self.train_num_batch, np.mean(self.train_acc)))
            print('INFO: Validation--------------')
            print('Loss: {}, Accuracy: {}'.format(self.running_vloss/self.test_num_batch, np.mean(self.val_acc)))
            self.lr_sched.step()
            # Save model
            if self.steps % 100 == 0:
                torch.save({"model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "lr_state_dict": self.lr_sched.state_dict()},
                        self.logdir + str(self.steps).zfill(6) + '.pt')

            # remember best prec@1 and save checkpoint
            is_best = np.mean(self.val_acc) > self.best_acc
            self.best_acc = max(np.mean(self.val_acc), self.best_acc)
            if (is_best):
                print('Best accuracy achieved, updating the model...')
                print('Best val_acc:', np.mean(self.val_acc))
                model_tmp = copy.deepcopy(self.model.state_dict())
                self.model.load_state_dict(model_tmp)
                torch.save({"model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "lr_state_dict": self.lr_sched.state_dict()}, os.path.join(self.logdir, 'best_classifier.pth'))

            
        logging.info('Finish training!')
        logging.info('')


