import math
from torch.optim import lr_scheduler


def get_scheduler(optimizer, args):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions.
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    # https://hasty.ai/docs/mp-wiki/scheduler/steplr

    if args.lr_policy == 'lambda':
        """ Create a schedule with a learning rate that decreases following the
            values of the cosine function between 0 and `pi * cycles` after a warmup
            period during which it increases linearly between 0 and 1.
        """
        def lambda_rule(epoch):
            num_cycles = 0.5

            if epoch < args.warmup:
                return float(epoch) / float(max(1, args.warmup))

            progress = float(epoch - args.warmup_epochs) / \
                float(max(1, args.epochs - args.warmup))

            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        scheduler = lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda_rule, last_epoch=-1, verbose=False)
    elif args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1, last_epoch=-1, verbose=False)
    elif args.lr_policy == 'mstep':
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=[20, 40, 60, 80], gamma=0.1, last_epoch=-1, verbose=False)
    elif args.lr_policy == 'exp':
        scheduler = lr_scheduler.ExponentialLR(
            optimizer, gamma=0.85, last_epoch=-1, verbose=False)
    elif args.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=7,
                                                   threshold=0.01, verbose=False)
    elif args.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-5, last_epoch=-1, verbose=False)
    elif args.lr_policy == 'clr':
        scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-2, step_size_up=4,
                                          step_size_down=None, cycle_momentum=False, gamma=1.0,
                                          mode="triangular2", last_epoch=-1, verbose=False)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler
