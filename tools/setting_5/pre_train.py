import os
import tqdm
import torch
import argparse
from torch.utils.data import DataLoader
from utils.tools import get_logger, get_optimizer, get_scheduler, set_seed
from datasets.func import get_sun_pesudo_video_datasetV2
from models.models.setting_5 import SegNet_FFormer as Setting_5


def train(args, model, train_loader, val_loader, snapshot_path):
    logger = get_logger(log_dir=snapshot_path, log_file=f'/log.log')
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)
    
    best_eval_dice = 0
    iter_num = 0
    for epoch in range(args.epochs):
        model.train()
        dataloader = iter(train_loader)
        iters_per_epoch = len(train_loader)
        pbar = tqdm.tqdm(range(iters_per_epoch))
        
        for idx in pbar:
            optimizer.zero_grad()
            data_batch = dataloader.__next__()
            losses, _ = model.pre_train(data_batch, args.seg_weight, args.temp_weight)
            loss = losses['total_loss'] if "total_loss" in losses else losses['seg_loss']
            loss.backward()
            optimizer.step()
            model.update_net_2_encoder()
            
            info = f'epoch[{epoch+1}/{args.epochs}], iter[{idx+1}/{iters_per_epoch}], lr: {optimizer.state_dict()["param_groups"][0]["lr"]}, total loss: {round(losses["total_loss"].item(), 4)}, seg loss: {round(losses["seg_loss"].item(), 4)}, temp loss: {round(losses["temp_loss"].item(), 4)}'
            pbar.set_description(info)
            logger.info(info)
            scheduler.step()
        
            if (iter_num + 1) % args.eval_iter == 0:
                eval_dict = model.evaluate(val_loader)
                info = f'------------->> epoch[{epoch+1}/{args.epochs}], iter[{idx+1}/{iters_per_epoch}], dice: {eval_dict["dice"]}'
                print(info)
                logger.info(info)
                if eval_dict['dice'] > best_eval_dice:
                    best_eval_dice = eval_dict['dice']
                    torch.save(model.state_dict(), f'{snapshot_path}/best_eval_model.pth')
                    print(f'---->> save best ckpt in {snapshot_path}/best_eval_model.pth')
                    logger.info(f'---->> save best ckpt in {snapshot_path}/best_eval_model.pth')
            iter_num += 1
        
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), f'{snapshot_path}/epoch_{epoch+1}_model.pth')
            print(f'---->> save ckpt in {snapshot_path}/epoch_{epoch+1}_model.pth')
            logger.info(f'---->> save ckpt in {snapshot_path}/epoch_{epoch+1}_model.pth')
    
    # save final model
    torch.save(model.state_dict(), f'{snapshot_path}/final_model.pth')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='mySemi')
    # model settings
    args.add_argument('--model-name', type=str, default='setting_5', help='the model name')
    args.add_argument('--batch_size', type=int, default=8)
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--device', type=str, default='cuda:0')
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--t', type=str)
    args.add_argument('--seg-weight', type=float, default=1.0)
    args.add_argument('--temp-weight', type=float, default=2.0)
    # optimizier and scheduler settings
    args.add_argument('--lr', type=float, default=0.001)
    args.add_argument('--optimizer', type=str, default='adam')
    args.add_argument('--sched', type=str, default='car')
    # eval
    args.add_argument('--eval-iter', type=int, default=500)
    # dataset settings
    args.add_argument('--img_size', type=int, default=352)
    args.add_argument('--memory_size', type=int, default=3)
    args.add_argument('--tfpc', type=int, default=10)   # train_frames_per_clip
    args.add_argument('--data-type', type=str, default='normal', choices=['small', 'normal'])
    args = args.parse_args()
    
    # set seed
    set_seed(args.seed)
    
    # save model path
    snapshot_path = f'outputs/{args.model_name}/{args.t}/pre_train/'
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    # model
    model = Setting_5(args.device).to(args.device)
    
    # train_loader, val_loader
    train_ds, val_ds = get_sun_pesudo_video_datasetV2(img_size=args.img_size, 
                                                      train_frames_per_clip=args.tfpc, 
                                                      memory_size=args.memory_size,
                                                      type=args.data_type
                                                      )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # train model
    train(args, model, train_loader, val_loader, snapshot_path)

