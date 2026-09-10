"""
Trainer and Tester for **R2Net**
"""
import torch
import torch.nn as nn
import random
import os
import os.path as osp

from accelerate import Accelerator
from utils.logger import setup_logger
from models.diffusion.ddpm import DDPM
from models.diffusion.ddim import DDIM
from models.physics_model import RSS_project
from models.physics_model import estimate_tx_map
from models.utils import height_embed

from utils.metre import AverageMeter
from utils.timestamp import what_time_is_it

from tqdm import tqdm
import time as tm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def do_train(
        model,
        optimizer,
        train_loader,
        accelerator: Accelerator,
        scheduler,
        loss_fn,
        Logger, # logger for train
        ckpt_dir,
        cfg=None
):
    """
    ### Function:
    - Trainer for R2Net
    """
    loss_meter = AverageMeter()
    height = cfg.CONDITION.TARGET_HEIGHT
    device = next(model.parameters()).device

    best_loss = float("inf")
    for epoch in range(1, cfg.SOLVER.EPOCHS + 1):
        tic = tm.time()
        loss_meter.reset()
        scheduler.step(epoch)
        model.train()

        for n_iter, batch in tqdm(enumerate(train_loader), 
                                    total=len(train_loader), desc=f"Ep-{epoch}/{cfg.SOLVER.EPOCHS}",
                                    disable=not accelerator.is_main_process):
            radiomap, sampled_map, sampled_mask, building, terrain, frequency, env, pointcloud = batch

            B = radiomap.shape[0]

            if height == "Hybrid":
                height_idx = random.randint(0, 2)
            else:
                height_idx = int(height)

            physics_rm, physics_mask = RSS_project(
                sampled_map=sampled_map.cpu(),
                sampled_mask=sampled_mask.cpu(),
                building=building.cpu(),
                env=env,
                target_height_idx=height_idx,
                cfg=cfg,
                pointcloud=pointcloud.cpu()
            )

            # NOTE when at single height, RSS_project() is equivalent as:
            # physics_rm = sampled_map[:, height_idx:height_idx + 1]
            # physics_mask = sampled_mask[:, height_idx:height_idx + 1]

            if epoch == 1 and n_iter == 0:
                counts = sampled_mask[0].sum(dim=(1, 2))

                expected = torch.zeros_like(counts)
                expected[height_idx] = cfg.INPUT.NUM_SAMPLE

                observed = sampled_map[:, height_idx:height_idx + 1].cpu()
                observed_mask = sampled_mask[:, height_idx:height_idx + 1].cpu().bool()

                Logger.info(
                    f"height_idx={height_idx}, "
                    f"sample_counts={counts.tolist()}, "
                    f"steps_per_epoch={len(train_loader)}"
                )
                assert torch.equal(counts, expected)
                assert height_idx == int(cfg.CONDITION.TARGET_HEIGHT)
                assert torch.equal(physics_rm[observed_mask], observed[observed_mask])

            # tx_map = estimate_tx_map(
            #    sampled_map=sampled_map.cpu(),
            #    sampled_mask=sampled_mask.cpu(),
            #    building=building.cpu(),
            #    target_height_idx=height_idx,
            #    cfg=cfg
            #)

            building = building.to(device)
            radiomap = radiomap.to(device)
            sampled_map = sampled_map.to(device)
            terrain = terrain.to(device)
            physics_rm = physics_rm.to(device)
            physics_mask = physics_mask.to(device)
            #tx_map = tx_map.to(device)

            canvas = radiomap[:, height_idx: height_idx+1] # GroudnTruth

            height_map = height_embed(
                height_idx=height_idx,
                cur_batch_size=B,
                device=device,
                cfg=cfg,
                is_test=False
            )

            # NOTE DEBUG
            # normed_bd, normed_terr = map(
            #     lambda x: x * 2 - 1,
            #     (building[:, height_idx: height_idx + 1], terrain)
            # )
            normed_bd, normed_terr = map(
                lambda x: x,
                (building[:, height_idx: height_idx + 1], terrain)
            )


            viz = False
            if viz:
                names = ["building", "terrain", "height_map", "radiomap"]
                variables = [normed_bd[0, 0], normed_terr[0, 0], height_map[0, 0], radiomap[0, 0]]
                for name, var in zip(names, variables):                
                    plt.imsave(
                            osp.join(f"./{name}.png"),
                                var.detach().cpu().numpy(),
                                cmap="viridis"
                            )
                    
            input = torch.cat(
                tensors=(physics_rm, physics_mask, normed_bd, normed_terr),
                dim=1
            )
            
            predict = model(input)
            loss = loss_fn(predict, canvas)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            loss_meter.update(loss.item(), B)

            # if n_iter % cfg.SOLVER.LOG_ITER == 0:
            #    Logger.info(f"Current loss: {loss_meter.avg}")

        loss_stats = torch.tensor(
            [loss_meter.sum, loss_meter.count],
            dtype=torch.float64,
            device=accelerator.device
        )

        loss_stats = accelerator.reduce(
            loss_stats,
            reduction="sum"
        )
            
        epoch_loss = (loss_stats[0] / loss_stats[1]).item()

        tok = tm.time()
        elapsed = tok - tic
        Logger.info(f"Epoch {epoch} finished in {elapsed:.2f}s")
        Logger.info(f"Epoch {epoch} Avg Loss: {epoch_loss:.6f}")
            
        if epoch_loss < best_loss:
        # if False:
            best_loss = epoch_loss
            best_epoch = epoch     

            Logger.info(f"Best CKPT is found at Epoch {best_epoch}.")
            Logger.info(f"Best CKPT is stored at {ckpt_dir}.")

            accelerator.wait_for_everyone()
            state_dict = accelerator.get_state_dict(model, unwrap=True)

            if accelerator.is_main_process:
                ckpt_path = osp.join(ckpt_dir, f"best.pth")
                accelerator.save(state_dict, ckpt_path)
               

        if epoch == cfg.SOLVER.EPOCHS:
            Logger.info(f"Training finished.")
            accelerator.wait_for_everyone()
            last_state_dict = accelerator.get_state_dict(model, unwrap=True)

            if accelerator.is_main_process:
                last_ckpt_path = osp.join(ckpt_dir, f"latest.pth")
                accelerator.save(last_state_dict, last_ckpt_path)

            accelerator.wait_for_everyone()


from utils.metrics import Evaluator
def do_test(
        model,
        test_loader,
        geo_type,
        height_idx,
        evaluator: Evaluator,
        accelerator: Accelerator,
        cfg=None,
        Logger=None,
):
    """
    ### Functions:
    - Tester for RadioLAM
    """
    model.eval()

    predicts_list = []
    targets_list = []
    device = accelerator.device 

    accelerator.wait_for_everyone()
    with torch.no_grad():
        for idx, (radiomap, sampled_map, sampled_mask, building, terrain, freq, env, pointcloud) in tqdm(enumerate(test_loader), desc=f"{geo_type}",
                                                        disable=not accelerator.is_main_process,
                                                        total=len(test_loader)):

            physics_rm, physics_mask = RSS_project(
                sampled_map=sampled_map.cpu(),
                sampled_mask=sampled_mask.cpu(),
                building=building.cpu(),
                env=env,
                target_height_idx=height_idx,
                cfg=cfg,
                pointcloud=pointcloud.cpu()
            )

            radiomap, sampled_map, building, terrain, physics_rm, physics_mask = map(
                lambda x: x.to(device), [radiomap, sampled_map, building, terrain, physics_rm, physics_mask]
            )

            B = radiomap.shape[0]
            height_map = height_embed(
                height_idx=height_idx,
                cur_batch_size=B,
                device=device,
                cfg=cfg,
                is_test=False # Correct, cuz need to avoid reranking
            )

            # for GT, [-1, 1] -> [0, 1]
            # NOTE DEBUG
            # groundtruth = (radiomap[:, height_idx: height_idx + 1] + 1) / 2
            groundtruth = radiomap[:, height_idx: height_idx + 1]

            # for input condition, [0, 1] -> [-1, 1]
            # height_map is natually [-1, 1] due to embedding

            # NOTE DEBUG
            # normed_bd, normed_terr = map(
            #     lambda x: x * 2 - 1,
            #     (building[:, height_idx: height_idx + 1], terrain)
            # )
            normed_bd, normed_terr = map(
                lambda x: x,
                (building[:, height_idx: height_idx + 1], terrain)
            )

            input = torch.cat(
                tensors=(physics_rm, physics_mask, normed_bd, normed_terr),
                dim=1
            )

            predicts = model(input)

            # NOTE DEBUG
            # predicts = (predicts.clamp(-1, 1) + 1) / 2
            predicts = predicts.clamp(0, 1)

            vis = True
            if vis:
                if accelerator.is_main_process:
                    root = f"./_PICs/{geo_type}"
                    if not osp.exists(root):
                        os.makedirs(root, exist_ok=True)

                    for batch_idx in range(predicts.shape[0]):                
                        plt.imsave(
                            osp.join(root, f"{geo_type}_height{height_idx}_{idx}_B{batch_idx}_best_img.png"),
                            predicts[batch_idx, 0].detach().cpu().numpy(),
                            cmap="viridis"
                        ) # , vmin=0, vmax=1 deleted
                        plt.imsave(
                            osp.join(root, f"{geo_type}_height{height_idx}_{idx}_B{batch_idx}_GroundTruth.png"),
                            groundtruth[batch_idx, 0].detach().cpu().numpy(),
                            cmap="viridis"
                        ) # , vmin=0, vmax=1 deleted
            
            # BUG DDP Eval evaluator.update(preds=best_img.float(),
            # BUG                 targets=groundtruth.float())
            # Only the main thread calculates the metrics
            # Other threads only collect the data
            all_preds, all_targets = accelerator.gather_for_metrics(input_data=(predicts.float(), groundtruth.float()))
            if accelerator.is_main_process:
                evaluator.update(preds=all_preds, targets=all_targets)

    results = None

    if accelerator.is_main_process:
        results = evaluator.compute()

    accelerator.wait_for_everyone()
    return results



            
        




