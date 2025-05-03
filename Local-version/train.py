import os
import numpy as np
import time
import sys
import torch
import torch.cuda

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
    print(torch.cuda.memory_summary(device=torch.device('cuda'), abbreviated=False))

    trainer()


    checkpoint = torch.load(f'{args.save_dir}/min_loss_checkpoint.pth.tar')
    trainer.model.load_state_dict(checkpoint['state_dict'])
    print ('Testing the min loss model')

    ## next two chunks are new testing the no checkpoint approach

    #min_loss_path = f'{args.save_dir}/min_loss_checkpoint.pth.tar'
    #if os.path.exists(min_loss_path):
    #    checkpoint = torch.load(min_loss_path)
    #    trainer.model.load_state_dict(checkpoint['state_dict'])
    #    print('Testing the min loss model')
    #    test_ind_auroc = trainer.test()
    #    test_ind_auroc = np.array(test_ind_auroc)
    #    trainer.print_auroc(test_ind_auroc[trainer.test_dl.dataset.seen_class_ids], trainer.test_dl.dataset.seen_class_ids, prefix='\ntest_seen')
    #    trainer.print_auroc(test_ind_auroc[trainer.test_dl.dataset.unseen_class_ids], trainer.test_dl.dataset.unseen_class_ids, prefix='\ntest_unseen')
    #else:
    #    print(f"Checkpoint not found at {min_loss_path}, skipping min loss testing.")


    #best_auroc_path = f'{args.save_dir}/best_auroc_checkpoint.pth.tar'
    #if os.path.exists(best_auroc_path):
    #    checkpoint = torch.load(best_auroc_path)
    #    trainer.model.load_state_dict(checkpoint['state_dict'])
    #    print('Testing the best AUROC model')
    #    test_ind_auroc = trainer.test()
    #    test_ind_auroc = np.array(test_ind_auroc)
    #    trainer.print_auroc(test_ind_auroc[trainer.test_dl.dataset.seen_class_ids], trainer.test_dl.dataset.seen_class_ids, prefix='\ntest_seen')
    #    trainer.print_auroc(test_ind_auroc[trainer.test_dl.dataset.unseen_class_ids], trainer.test_dl.dataset.unseen_class_ids, prefix='\ntest_unseen')
    #else:
    #    print(f"Checkpoint not found at {best_auroc_path}, skipping best AUROC testing.")

    test_ind_auroc = trainer.test()
    test_ind_auroc = np.array(test_ind_auroc)
    
    


    trainer.print_auroc(test_ind_auroc[trainer.test_dl.dataset.seen_class_ids], trainer.test_dl.dataset.seen_class_ids, prefix='\ntest_seen')
    trainer.print_auroc(test_ind_auroc[trainer.test_dl.dataset.unseen_class_ids], trainer.test_dl.dataset.unseen_class_ids, prefix='\ntest_unseen')

    checkpoint = torch.load(f'{args.save_dir}/best_auroc_checkpoint.pth.tar')
    trainer.model.load_state_dict(checkpoint['state_dict'])
    print ('Testing the best AUROC model')
    test_ind_auroc = trainer.test()
    test_ind_auroc = np.array(test_ind_auroc)
    

    trainer.print_auroc(test_ind_auroc[trainer.test_dl.dataset.seen_class_ids], trainer.test_dl.dataset.seen_class_ids, prefix='\ntest_seen')
    trainer.print_auroc(test_ind_auroc[trainer.test_dl.dataset.unseen_class_ids], trainer.test_dl.dataset.unseen_class_ids, prefix='\ntest_unseen')

if __name__ == '__main__':
    main()





