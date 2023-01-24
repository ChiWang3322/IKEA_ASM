import os, yaml, thop, warnings, logging, pynvml, torch, numpy as np, math
from copy import deepcopy
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

# from . import utils as U
# from . import dataset
# from . import model
# from . import scheduler


class Initializer():
    def __init__(self, args):
        self.args = args
        self.init_save_dir()

        logging.info('')
        logging.info('Starting preparing ...')
        self.init_log_dir()
        self.init_environment()
        self.init_device()
        self.init_dataloader()
        self.init_model()
        self.init_optimizer()
        self.init_lr_scheduler()
        self.init_loss_func()
        logging.info('Successful!')
        logging.info('')

    def init_log_dir(self):
        os.makedirs(self.args.logdir, exist_ok=True)
        logging.info('Saving folder path: {}'.format(self.args.logdir))

    def init_environment(self):
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        self.global_step = 0
        if self.args.debug:
            self.no_progress_bar = True
            self.model_name = 'debug'
            self.scalar_writer = None
        elif self.args.evaluate or self.args.extract:
            self.no_progress_bar = self.args.no_progress_bar
            self.model_name = '{}_{}_{}'.format(self.args.config, self.args.model_type, self.args.dataset)
            self.scalar_writer = None
            warnings.filterwarnings('ignore')
        else:
            self.no_progress_bar = self.args.no_progress_bar
            self.model_name = '{}_{}_{}'.format(self.args.config, self.args.model_type, self.args.dataset)
            self.scalar_writer = SummaryWriter(logdir=self.save_dir)
            warnings.filterwarnings('ignore')
        logging.info('Saving model name: {}'.format(self.model_name))

    def init_device(self):
        if type(self.args.gpus) is int:
            self.args.gpus = [self.args.gpus]
        if len(self.args.gpus) > 0 and torch.cuda.is_available():
            pynvml.nvmlInit()
            for i in self.args.gpus:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memused = meminfo.used / 1024 / 1024
                logging.info('GPU-{} used: {}MB'.format(i, memused))
                if memused > 1000:
                    pynvml.nvmlShutdown()
                    logging.info('')
                    logging.error('GPU-{} is occupied!'.format(i))
                    raise ValueError()
            pynvml.nvmlShutdown()
            self.output_device = self.args.gpus[0]
            self.device =  torch.device('cuda:{}'.format(self.output_device))
            torch.cuda.set_device(self.output_device)
        else:
            logging.info('Using CPU!')
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            self.output_device = None
            self.device =  torch.device('cpu')

    def init_dataloader(self):
        train_transforms = None
        test_transforms = None
        self.train_dataset = Dataset(self.args.dataset_path, db_filename=self.args.db_filename, 
                                    train_filename=self.args.train_filename,
                                    transform=train_transforms, set='train', 
                                    camera=self.args.camera, frame_skip=self.args.frame_skip,
                                    frames_per_clip=self.args.frames_per_clip, 
                                    mode=self.args.load_mode, pose_path=self.args.pose_path, arch=self.args.arch)
        
        weights = utils.make_weights_for_balanced_classes(self.train_dataset.clip_set, self.train_dataset.clip_label_count)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.args.batch_size, 
                                                    sampler=sampler, num_workers=6, pin_memory=False)

        self.test_dataset = Dataset(self.args.dataset_path, db_filename=self.args.db_filename, 
                            train_filename=self.args.train_filename,
                            test_filename=self.args.testset_filename, transform=test_transforms, 
                            set='test', camera=self.args.camera, frame_skip=self.args.frame_skip, 
                            frames_per_clip=self.args.frames_per_clip, mode=self.args.load_mode,
                            pose_path=self.args.pose_path, arch=self.args.arch)

        self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=6,
                                                    pin_memory=False)

        logging.info('Dataset: {}'.format('IKEA_ASM'))
        logging.info("Number of clips in the dataset:{}".format(len(self.train_dataset)))
        logging.info('Batch size: train-{}, eval-{}'.format(self.train_batch_size, self.eval_batch_size))
        logging.info('Data shape (branch, channel, frame, joint, person): {}'.format(self.data_shape))
        logging.info('Number of action classes: {}'.format(self.train_dataset.num_classes))

    def init_model(self):
        if self.args.arch == 'STGCN':
            graph_args = {'layout': 'openpose', 'strategy': 'spatial'} #ntu-rgb+d
            self.model = st_gcn.Model(in_channels=2, num_class=num_classes, graph_args=graph_args,
                                edge_importance_weighting=True, dropout=0.5)
        elif self.args.arch == 'AGCN':
            self.model = agcn.Model(num_class=num_classes, num_point=18, num_person=1, 
                            graph='graph.kinetics.Graph', graph_args={'labeling_mode':'spatial'}, in_channels=2)
        elif self.args.arch =='STGAN':
            config = [ [64, 64, 16, 1], [64, 64, 16, 1],
                [64, 128, 32, 2], [128, 128, 32, 1],
                [128, 256, 64, 2], [256, 256, 64, 1],
                [256, 256, 64, 1], [256, 256, 64, 1],
            ]
            self.model = st2ransformer_dsta.DSTANet(num_class=num_classes, num_point=18, num_frame=num_frame, num_subset=4, dropout=0., config=config, num_person=1,
                    num_channel=2)                      
        elif self.args.arch == 'EGCN':
            #B4
            graph = g()
            __activations = {
                'relu': nn.ReLU(inplace=True),
                'relu6': nn.ReLU6(inplace=True),
                'hswish': HardSwish(inplace=True),
                'swish': Swish(inplace=True),
            }
            def rescale_block(block_args, scale_args, scale_factor):
                channel_scaler = math.pow(scale_args[0], scale_factor)
                depth_scaler = math.pow(scale_args[1], scale_factor)
                new_block_args = []
                for [channel, stride, depth] in block_args:
                    channel = max(int(round(channel * channel_scaler / 16)) * 16, 16)
                    depth = int(round(depth * depth_scaler))
                    new_block_args.append([channel, stride, depth])
                return new_block_args


            block_args = rescale_block([[48,1,0.5],[24,1,0.5],[64,2,1],[128,2,1]], 
                                        scale_args=[1.2,1.35],
                                        scale_factor=4)
            print('New block args:', block_args)
            act_type = 'swish'
            model = EGCN(data_shape=(3, 4, self.args.frames_per_clip, 18, 1),stem_channel = 64,
                    block_args = block_args,
                    fusion_stage = 2,
                    num_class = num_classes,
                    act =  __activations[act_type],
                    att_type =  'stja',
                    layer_type = 'Sep',
                    drop_prob = 0.25,
                    kernel_size = [5,2],
                    # scale_args = [1.2,1.35],
                    expand_ratio = 2,
                    reduct_ratio = 4,
                    bias = True,
                    edge = True,
                    A = graph.A)
        else:
            raise ValueError("Unsupported architecture: please select HCN | ST_GCN | AGCN | STGAN")

        if self.args.refine:
            if self.args.refine_epoch == 0:
                raise ValueError("You set the refine epoch to 0. No need to refine, just retrain.")
            refine_model_filename = os.path.join(logdir, str(refine_epoch).zfill(6)+'.pt')
            checkpoint = torch.load(refine_model_filename)
            model.load_state_dict(checkpoint["model_state_dict"])
        logging.info('Model: {}'.format(self.args.arch))
        flops, params = thop.profile(deepcopy(self.model), inputs=torch.rand([1,1]+self.data_shape), verbose=False)
        logging.info('Model profile: {:.2f}G FLOPs and {:.2f}M Parameters'.format(flops / 1e9, params / 1e6))
        logging.info('Create model randomly.')

















        # kwargs = {
        #     'data_shape': self.data_shape,
        #     'num_class': self.num_class,
        #     'A': torch.Tensor(self.A),
        #     'parts': self.parts,
        # }
        # self.model = model.create(self.args.model_type, **(self.args.model_args), **kwargs)
        # logging.info('Model: {} {}'.format(self.args.model_type, self.args.model_args))
        # with open('{}/model.txt'.format(self.save_dir), 'w') as f:
        #     print(self.model, file=f)
        # flops, params = thop.profile(deepcopy(self.model), inputs=torch.rand([1,1]+self.data_shape), verbose=False)
        # logging.info('Model profile: {:.2f}G FLOPs and {:.2f}M Parameters'.format(flops / 1e9, params / 1e6))
        # self.model = torch.nn.DataParallel(
        #     self.model.to(self.device), device_ids=self.args.gpus, output_device=self.output_device
        # )
        # pretrained_model = '{}/{}.pth.tar'.format(self.args.pretrained_path, self.model_name)
        # if os.path.exists(pretrained_model):
        #     checkpoint = torch.load(pretrained_model, map_location=torch.device('cpu'))
        #     self.model.module.load_state_dict(checkpoint['model'])
        #     self.cm = checkpoint['best_state']['cm']
        #     logging.info('Pretrained model: {}'.format(pretrained_model))
        # elif self.args.pretrained_path:
        #     logging.warning('Warning: Do NOT exist this pretrained model: {}!'.format(pretrained_model))
        #     logging.info('Create model randomly.')

    def init_optimizer(self):
        if self.args.arch == 'STGCN':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.init_lr, weight_decay=1E-6)
            
        
        elif self.args.arch == 'AGCN':

            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.init_lr,
                weight_decay=weight_decay)
            
        elif self.args.arch == 'STGAN':

            
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.init_lr, weight_decay=self.args.weight_decay)
            lr_sched = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[50, 100, 150, 200, 250], gamma=0.1)
        elif self.args.arch == 'EGCN':

            
            self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=self.args.init_lr, 
                                    weight_decay=self.args.weight_decay, 
                                    betas=[0.9,0.99])
            lr_sched = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[10, 30, 50, 70, 90], gamma=0.1)
        if self.args.refine:
            lr_sched.load_state_dict(checkpoint["lr_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logging.info('Optimizer: {} {}'.format(self.args.optimizer, optimizer_args))

    def init_lr_scheduler(self):
        self.lr_sched = optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[10, 30, 50, 70, 90], gamma=0.1)
        logging.info('LR_Scheduler: '.format('MultiStepLR'))

    def init_loss_func(self):
        self.loss_func = torch.nn.CrossEntropyLoss().to(self.device)
        logging.info('Loss function: {}'.format(self.loss_func.__class__.__name__))
