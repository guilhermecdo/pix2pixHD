import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
import math

# Fix for Python 3.9+ (fractions.gcd was removed)
def lcm(a, b): 
    return abs(a * b) // math.gcd(a, b) if a and b else 0

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    
    # Handle continuation of training
    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print(f'Resuming from epoch {start_epoch} at iteration {epoch_iter}')        
    else:    
        start_epoch, epoch_iter = 1, 0

    opt.print_freq = int(lcm(opt.print_freq, opt.batchSize))    

    # Initialize Data and Model
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print(f'#training images = {dataset_size}')

    model = create_model(opt)
    visualizer = Visualizer(opt)
    
    # Optimizer setup
    optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D

    total_steps = (start_epoch - 1) * dataset_size + epoch_iter
    display_delta = total_steps % opt.display_freq
    print_delta = total_steps % opt.print_freq
    save_delta = total_steps % opt.save_latest_freq

    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size
            
        for i, data in enumerate(dataset, start=epoch_iter):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # Whether to collect output images for visualizer
            save_fake = total_steps % opt.display_freq == display_delta

            ############## Forward Pass ######################
            # Variable is deprecated in newer PyTorch but kept for pix2pixHD compatibility
            losses, generated = model(Variable(data['label']), Variable(data['inst']), 
                                    Variable(data['image']), Variable(data['feat']), infer=save_fake)

            # Sum per-device losses and create dict
            losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
            loss_dict = dict(zip(model.module.loss_names, losses))

            # Calculate scalars for display
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0)

            ############### Backward Pass ####################
            # Update Generator
            optimizer_G.zero_grad()
            loss_G.backward()          
            optimizer_G.step()

            # Update Discriminator
            optimizer_D.zero_grad()
            loss_D.backward()        
            optimizer_D.step()        

            ############## Display results and errors ##########
            # Explicit console print so it doesn't look stuck
            if total_steps % opt.print_freq == print_delta:
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}            
                t = (time.time() - iter_start_time)
                print(f"(Epoch: {epoch}, It: {epoch_iter}/{dataset_size}) Loss_G: {loss_G.item():.4f} Loss_D: {loss_D.item():.4f} Time: {t:.3f}s")
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)

            ### display output images to web
            if save_fake and not opt.no_html:
                visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                       ('synthesized_image', util.tensor2im(generated.data[0])),
                                       ('real_image', util.tensor2im(data['image'][0]))])
                visualizer.display_current_results(visuals, epoch, total_steps)

            ### save latest model
            if total_steps % opt.save_latest_freq == save_delta:
                print(f'Saving the latest model (epoch {epoch}, total_steps {total_steps})')
                model.module.save('latest')            
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

            if epoch_iter >= dataset_size:
                break
           
        # End of epoch 
        print(f'End of epoch {epoch} / {opt.niter + opt.niter_decay} \t Time Taken: {time.time() - epoch_start_time:.2f} sec')

        ### save model for this epoch
        if epoch % opt.save_epoch_freq == 0:
            print(f'Saving the model at the end of epoch {epoch}, iters {total_steps}')        
            model.module.save('latest')
            model.module.save(epoch)
            np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

        ### linearly decay learning rate
        if epoch > opt.niter:
            model.module.update_learning_rate()