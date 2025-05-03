import os
import numpy as np
import time
import sys
import torch

from ChexnetTrainer import ChexnetTrainer
from arguments import  parse_args


def main ():

    args = parse_args()
    seed = 1002
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    try:  
        os.mkdir(args.save_dir)  
    except OSError as error:
        print(error) 
    
    trainer = ChexnetTrainer(args)

    #CHATGPT WRITTEN FOR RESUME_FROM
    if args.resume_from is not None and os.path.exists(args.resume_from):
        print(f"Resuming training from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location='cuda', weights_only=False)
        trainer.model.load_state_dict(checkpoint['state_dict'])
        if 'optimizer' in checkpoint and hasattr(trainer, 'optimizer'):
            trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'epoch' in checkpoint:
            trainer.start_epoch = checkpoint['epoch'] + 1
        else:
            trainer.start_epoch = 0
    else:
        trainer.start_epoch = 0
    trainer()

    checkpoint = torch.load(f'{args.save_dir}/min_loss_checkpoint.pth.tar', weights_only=False)
    print('checkpoint: ', checkpoint)
    trainer.model.load_state_dict(checkpoint['state_dict'])
    print ('Testing the min loss model')
    test_ind_auroc = trainer.test()
    test_ind_auroc = np.array(test_ind_auroc)
    


    trainer.print_auroc(test_ind_auroc[trainer.test_dl.dataset.seen_class_ids], trainer.test_dl.dataset.seen_class_ids, prefix='\ntest_seen')
    trainer.print_auroc(test_ind_auroc[trainer.test_dl.dataset.unseen_class_ids], trainer.test_dl.dataset.unseen_class_ids, prefix='\ntest_unseen')

    checkpoint = torch.load(f'{args.save_dir}/best_auroc_checkpoint.pth.tar', weights_only=False)
    trainer.model.load_state_dict(checkpoint['state_dict'])
    print ('Testing the best AUROC model')
    test_ind_auroc = trainer.test()
    test_ind_auroc = np.array(test_ind_auroc)
    


    trainer.print_auroc(test_ind_auroc[trainer.test_dl.dataset.seen_class_ids], trainer.test_dl.dataset.seen_class_ids, prefix='\ntest_seen')
    trainer.print_auroc(test_ind_auroc[trainer.test_dl.dataset.unseen_class_ids], trainer.test_dl.dataset.unseen_class_ids, prefix='\ntest_unseen')

if __name__ == '__main__':
    main()





