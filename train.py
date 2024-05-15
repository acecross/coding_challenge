import hydra
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
from hydra.utils import get_original_cwd
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from customn_data_loader import CustomDatasetFromCSV
from network import Net


@hydra.main(config_name="train.yaml", config_path="cfg")
def train(cfg):
    #get some params from cfg
    #hydra changes cwd and creates and output folder. Nice for tracing results (can be overwritten)
    cwd = get_original_cwd()
    data_folder = cfg.io.data_folder
    image_folder = cfg.io.image_folder
    csv_file = cfg.io.csv_file
    device = cfg.params.device
    torch.manual_seed(cfg.params.seed)
    #todo: cross validation or simulate more data
    train_split = .8
    dataset = CustomDatasetFromCSV(os.path.join(cwd,data_folder,csv_file), os.path.join(cwd,data_folder,image_folder), resize=(cfg.params.image_resize,cfg.params.image_resize))
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    #split dataset in train and test. Reproducible with same seed
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    #create function to load tensors to device on collection
    to_device_lambda = lambda x: tuple(x_.to(device) for x_ in default_collate(x))
    #use dataloader to load data fast with multiprocessing
    train_dataloader = DataLoader(train_dataset,batch_size=50, shuffle=True, collate_fn=to_device_lambda)
    test_dataloader = DataLoader(test_dataset,batch_size=50, shuffle=True, collate_fn=to_device_lambda)
    iterations = cfg.params.iterations
    epoch = 0
    #todo: no network loader always trains from scratch
    #load network image size is important to compute dense layer dimension
    net = Net(size=(cfg.params.image_resize,cfg.params.image_resize)).to(device)
    #optimizer adam input is normed so we use default learning rate
    opt = torch.optim.Adam(net.parameters())
    loss_list = []
    #loss function
    lossf = torch.nn.MSELoss()
    #compute validation loss always with MSE to keep it reproducible
    lossf_v = torch.nn.MSELoss()
    #create training folder if it doesnt exist
    if not os.path.exists(os.path.join(cwd,cfg.io.training_folder)):
        os.mkdir(os.path.join(cwd,cfg.io.training_folder))
    save_path = os.path.join(cwd,cfg.io.training_folder,cfg.io.training_name)
    #load network
    for i in range(iterations):
        for images,ground_truth in train_dataloader:

            #set to none speeds up simple_cnn because gradients are not deleted from memory
            opt.zero_grad(set_to_none=True)
            out = net(images)
            loss = lossf(out, ground_truth)
            loss.backward()
            opt.step()
        epoch+=1
        #each save point is 2 epochs
        if i%2 ==0:
            #cast eval to disable dropout
            net.eval()
            # no grad needed in evaluation

            with torch.no_grad():
                i=0
                v_loss = torch.zeros((1)).to(device)
                for im,gt in test_dataloader:
                    i+=1
                    v_out = net(im)
                    v_loss += lossf_v(v_out, gt)
                    print(f"loss: {loss} validation_loss = {v_loss[0]}", i)
                #normalize loss by number of batches in validation
                loss_list.append([loss.cpu().numpy(), v_loss.cpu().numpy()/i])
            #set network back to train mode
            net.train()
            #save training
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': loss_list,
                #'optimizer_params': cfg.optimizer.params,
                #'training_time': time.time()-t1
            }, save_path)

if __name__ == '__main__':
    train()