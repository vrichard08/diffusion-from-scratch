import torch


@torch.no_grad()
def depth_metrics(pred, gt, eps=1e-6):
    
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if gt.dim() == 4:
        gt = gt.squeeze(1)

    pred = pred.float()
    gt = gt.float()

    pred = torch.clamp(pred, min=eps)
    gt = torch.clamp(gt, min=eps)

    pred = pred.view(-1)
    gt = gt.view(-1)


    diff = pred - gt

    rmse = torch.sqrt(torch.mean(diff ** 2))

    rmse_log = torch.sqrt(
        torch.mean((torch.log(pred) - torch.log(gt)) ** 2)
    )

    abs_rel = torch.mean(torch.abs(diff) / gt)

    sq_rel = torch.mean((diff ** 2) / gt)

    ratio = torch.max(pred / gt, gt / pred)

    d1 = torch.mean((ratio < 1.25).float())
    d2 = torch.mean((ratio < 1.25 ** 2).float())
    d3 = torch.mean((ratio < 1.25 ** 3).float())

    return {
        "δ1": d1.item(),
        "δ2": d2.item(),
        "δ3": d3.item(),
        "REL": abs_rel.item(),
        "SqRel": sq_rel.item(),
        "RMSE": rmse.item(),
        "RMSE_log": rmse_log.item(),
    }

