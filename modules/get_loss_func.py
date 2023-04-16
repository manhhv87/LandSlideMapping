from losses import SegLoss as segloss


def get_loss_function(args):
    if args.loss_func == 'dice':
        return segloss.DiceLoss('multiclass')
    elif args.loss_func == 'ce':
        return segloss.FocalLoss('multiclass', alpha=None, gamma=0.0)
    elif args.loss_func == 'focal':
        return segloss.FocalLoss('multiclass', alpha=0.5, gamma=2.0)
    elif args.loss_func == 'jaccard':
        return segloss.JaccardLoss('multiclass', smooth=1.0)
    elif args.loss_func == 'sjc':
        return segloss.JDTLoss(log_loss=True, gamma=1.2, threshold=0.01, class_weight=[0.9, 0.1])
    elif args.loss_func == 'lovasz':
        return segloss.LovaszLoss('multiclass')
    elif args.loss_func == 'tversky':
        return segloss.TverskyLoss('multiclass')
    elif args.loss_func == 'mcc':
        return segloss.MCCLoss('multiclass')
    elif args.loss_func == 'mix':
        return segloss.ComboLoss(alpha=0.5, dice_kwargs={'mode': 'multiclass'}, mcc_kwargs={'mode': 'multiclass'})
    else:
        raise ValueError("Choice of loss function")
