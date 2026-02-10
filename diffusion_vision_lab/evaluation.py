
import torch
from timm.utils import ModelEmaV3 
from unet import Unet
from dataset import  NYUDepthV2Test
from scheduler import DDPM_Scheduler
from metrics import depth_metrics
from tqdm import tqdm



def evaluate_monocular_depth(
    diffusion_checkpoint: str='ddpm_checkpoint_monocular_depth.pt',
    num_time_steps: int=1000,
    ema_decay: float=0.999,
    batch_size: int=8
):

    checkpoint = torch.load(diffusion_checkpoint)
    model = Unet(input_dim=4,output_dim=1).cuda()
    model.load_state_dict(checkpoint['weights'])
    ema = ModelEmaV3(model, decay=ema_decay)
    ema.load_state_dict(checkpoint['ema'])
    model.eval()
    
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    scheduler.alpha = scheduler.alpha.cuda()
    scheduler.beta = scheduler.beta.cuda()


    test_dataset = NYUDepthV2Test("/kaggle/input/nyu-depth-v2/nyu_data/data/nyu2_test", size=(64,64))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    totals = {k:0 for k in ["δ1","δ2","δ3","REL","SqRel","RMSE","RMSE_log"]}
    n = 0
    
    with torch.no_grad():
        model = ema.module.eval()                  
        for batch in tqdm(test_loader):
            batch = batch.cuda()         
            rgb = batch[:, :3, :, :]              
            gt  = batch[:, 3:, :, :]   
            B, _, H, W = rgb.shape
            depth = torch.randn(B, 1, H, W).cuda()
            for t in reversed(range(1, num_time_steps)):
                t_tensor = torch.full((B,), t, dtype=torch.long).cuda()
                model_in = torch.cat([rgb, depth], dim=1)
                temp = (scheduler.beta[t]/( (torch.sqrt(1-scheduler.alpha[t]))*(torch.sqrt(1-scheduler.beta[t])) ))
                depth = ((1 / torch.sqrt(1 - scheduler.beta[t])) * depth - (temp * model(model_in, t_tensor)))
                noise = torch.randn_like(depth)
                depth = depth + (noise * torch.sqrt(scheduler.beta[t]))

            t_tensor = torch.full((B,), 0, dtype=torch.long).cuda()
            model_in = torch.cat([rgb, depth], dim=1)
            temp = scheduler.beta[0] / (torch.sqrt(1 - scheduler.alpha[0]) * torch.sqrt(1 - scheduler.beta[0]))
            depth = ((1 / torch.sqrt(1 - scheduler.beta[0])) * depth - temp * model(model_in, t_tensor))
            
            pred_m = (depth + 1) / 2 * 10
            gt_m   = (gt + 1) / 2 * 10

        m = depth_metrics(pred_m, gt_m)
        for k in totals:
            totals[k] += m[k]
        n += 1 

    for k in totals:
        totals[k] /= n

    print(totals)
    
    return totals



