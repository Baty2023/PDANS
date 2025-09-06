import os
import open3d
import numpy as np
import torch
import shutil

from util import rescale, find_max_epoch, print_size, sampling, sampling_ddim,calc_diffusion_hyperparams, AverageMeter,pc_normalization,numpy_to_pc,pc_normalize
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist,hausdorff_distance

import time
from sklearn.decomposition import PCA

def p2f_distance(X, Y, k=10):
    """
    Calculate the Point-to-Feature (P2F) distance from each point in X to Y's local feature planes.
    
    Parameters
    ----------
    X : torch.Tensor
        Source point cloud of shape (B, N, C)
    Y : torch.Tensor
        Target point cloud of shape (B, M, C)
    k : int
        Number of neighbors to use for local plane fitting in Y
    
    Returns
    -------
    p2f_mean : float
        Mean P2F distance across all batches
    p2f_rms : float
        RMS (Root Mean Square) P2F distance across all batches
    """
    B, N, C = X.shape
    device = X.device
    
    p2f_distances = []

    # Loop over each batch
    for b in range(B):
        batch_X = X[b].cpu().numpy()  # Shape: (N, C)
        batch_Y = Y[b].cpu().numpy()  # Shape: (M, C)

        # KD-Tree for efficient neighbor search
         # Initialize Open3D point cloud and KD-Tree for target point cloud Y
        target_pcd = open3d.geometry.PointCloud()
        target_pcd.points = open3d.utility.Vector3dVector(batch_Y)
        y_kd_tree = open3d.geometry.KDTreeFlann(target_pcd)  # KD-Tree for efficient neighbor search
        batch_p2f_distances = []

        for x in batch_X:
            _, idx, _ = y_kd_tree.search_knn_vector_3d(x, k)  # Get k nearest neighbors in Y
            neighbors = batch_Y[idx, :]  # Shape: (k, C)

            # Fit a plane using PCA
            pca = PCA(n_components=3)
            pca.fit(neighbors)
            normal = pca.components_[-1]  # Plane normal vector
            q = neighbors[0]  # A point on the plane

            # Calculate point-to-plane distance
            d = np.abs(np.dot((x - q), normal))
            batch_p2f_distances.append(d)

        p2f_distances.append(np.mean(batch_p2f_distances))

    # Calculate mean and RMS P2F distance
    p2f_mean = np.mean(p2f_distances)
    p2f_rms = np.sqrt(np.mean(np.square(p2f_distances)))

    return p2f_mean, p2f_rms

def evaluate(
        net,
        testloader,
        diffusion_hyperparams,
        print_every_n_steps=200,
        scale=1,
        compute_cd=True,
        return_all_metrics=False,
        R=4,
        npoints=2048,
        gamma=0.5,
        T=1000,
        step=30,
        mesh_path = "/mnt/SG10T/DataSet/PUGAN/test/mesh",
        p2f_root="/home/zhang/workspace/PUDM-main/pointnet2/test/pugan",
        save_dir = "./test/xys",
        save_xyz = True,            # pre dense point cloud
        save_sp=False,               # pre sparse point cloud
        save_z = False,             # input Gaussian noise
        save_condition = False,     # input sparse point cloud
        save_gt = False,            # true dense point cloud
        save_mesh = False,
        p2f = False,
):
    CD_meter = AverageMeter()
    HD_meter = AverageMeter()
    P2F_meter = AverageMeter()
    P2F_meter_rms = AverageMeter()
    total_len = len(testloader)

    total_meta = torch.rand(0).cuda().long()

    metrics = {
        'cd_distance': torch.rand(0).cuda(),
        'h_distance': torch.rand(0).cuda(),
        'cd_p': torch.rand(0).cuda(),
    }

    cd_module = chamfer_3DDist()

    total_time = 0
    cd_result = 0
    times = 0
    mesh_path = mesh_path
    save_path = save_dir
    p2f_root = p2f_root
    save_xyz = save_xyz
    save_z = save_z
    save_condition =save_condition
    save_gt =save_gt
    save_sp = save_sp
    save_mesh = save_mesh
    p2f = p2f
    print(f"**** {npoints} -----> {npoints * R} ****")
    for idx, data in enumerate(testloader):

        label = data['label'].cuda()
        condition = data['partial'].cuda()
        gt = data['complete'].cuda()

        batch,num_points,_ = gt.shape
        net.reset_cond_features()
        start = time.time()

        start_time = time.time()

        if (step < T):
            generated_data, condition_pre, z = sampling_ddim(
                net=net,
                size=(batch, num_points, 3),
                diffusion_hyperparams=diffusion_hyperparams,
                label=label,
                condition=condition,
                R=R,
                gamma=gamma,
                step=step
            )
        else:
            generated_data,condition_pre,z = sampling(
                net=net,
                size=(batch,num_points,3),
                diffusion_hyperparams=diffusion_hyperparams,
                print_every_n_steps=print_every_n_steps,
                label=label,
                condition=condition,
                R=R,
                gamma=gamma
            )

        end_time = time.time() - start_time
        times += end_time

        generation_time = time.time() - start
        total_time = total_time + generation_time
        generated_data = generated_data/scale
        gt = gt/scale
        torch.cuda.empty_cache()

        if compute_cd:
            cd_p, dist, _,_ = cd_module(generated_data, gt)
            dist = (cd_p + dist) / 2.0
            cd_loss = dist.mean().detach().cpu().item()
        else:
            dist = torch.zeros(generated_data.shape[0], device=generated_data.device, dtype=generated_data.dtype)
            cd_p = dist
            cd_loss = dist.mean().detach().cpu().item()

        cd_result += torch.sum(cd_p).item()

        # ---- h distance ----
        hd_cost = hausdorff_distance(generated_data,gt)
        hd_loss = hd_cost.mean().detach().cpu().item()
        # ---- h distance ----

        # ---- p2f ----
        p2f_loss = 0
        p2f_std = 0
        names = data['name']
        if(p2f):
            global_p2f = []
            for name in names:
                p2f_path = os.path.join(p2f_root,f"{name}_point2mesh_distance.xyz")
                if(os.path.exists(p2f_path)):
                    point2mesh_distance = np.loadtxt(p2f_path).astype(np.float32)
                    if point2mesh_distance.size == 0:
                        continue
                    point2mesh_distance = point2mesh_distance[:, 3]
                    global_p2f.append(point2mesh_distance)
            global_p2f = np.concatenate(global_p2f, axis=0)
            p2f_loss = np.nanmean(global_p2f)
            p2f_std = np.nanstd(global_p2f)
        # if(p2f):
        #     p2f_mean, p2f_rms = p2f_distance(generated_data, gt)
        # ---- p2f ----
        total_meta = torch.cat([total_meta, label])

        metrics['cd_distance'] = torch.cat([metrics['cd_distance'], dist])
        metrics['h_distance'] = torch.cat([metrics['h_distance'], hd_cost])
        metrics['cd_p'] = torch.cat([metrics['cd_p'], cd_p])

        CD_meter.update(cd_loss, n=batch)
        HD_meter.update(hd_loss, n=batch)
        P2F_meter.update(p2f_loss, n=batch)
        P2F_meter_rms.update(p2f_std, n=batch)

        print('progress [%d/%d] %.4f (%d samples) CD distance %.8f Hausdorff distance %.8f p2f %.8f p2f_rms %.8f this batch time %.2f total generation time %.2f' % (
            idx, total_len,
            idx/total_len,
            batch,
            CD_meter.avg,
            HD_meter.avg,
            P2F_meter.avg,
            P2F_meter_rms.avg,
            generation_time,
            total_time
        ), flush=True)


        if(save_xyz):
            # ---- save data ----
            generated_np = generated_data.detach().cpu().numpy()
            condition_pre_np = condition_pre.detach().cpu().numpy()
            z_np = z.detach().cpu().numpy()
            gt_np = gt.detach().cpu().numpy()
            condition_np = condition.detach().cpu().numpy()
            for i in range(len(generated_np)):
                name = names[i]
                #name = idx
                # ---- generated ----
                generated_points = generated_np[i]
                generated_pc = numpy_to_pc(generated_points)
                generated_path = os.path.join(save_path,f"{name}.xyz")
                open3d.io.write_point_cloud(filename=generated_path,pointcloud=generated_pc)
                print(f"---- saving generated dense point cloud: {generated_path} ----")
                # ---- generated ----

                # ---- mesh ----
                if (save_mesh):
                    mesh_source = os.path.join(mesh_path,f"{name}.off")
                    mesh_dist = os.path.join(save_path,f"{name}.off")
                    shutil.copy(mesh_source,mesh_dist)
                    print(f"---- saving mesh: {mesh_dist} ----")
                # ---- mesh ----

                # ---- pre condition ----
                if(save_sp):
                    condition_pre_points = condition_pre_np[i]
                    condition_pre_pc = numpy_to_pc(condition_pre_points)
                    condition_pre_path = os.path.join(save_path,f"{name}_sp.xyz")
                    open3d.io.write_point_cloud(filename=condition_pre_path,pointcloud=condition_pre_pc)
                    print(f"---- saving generated sparse point cloud: {condition_pre_path} ----")
                # ---- pre condition ----

                # ---- z ----
                if(save_z):
                    z_points = z_np[i]
                    z_pc = numpy_to_pc(z_points)
                    z_path = os.path.join(save_path,f"{name}_z.xyz")
                    open3d.io.write_point_cloud(filename=z_path,pointcloud=z_pc)
                    print(f"---- saving input Gaussian noise: {z_path} ----")
                # ---- z ----

                # ---- gt ----
                if(save_gt):
                    gt_points = gt_np[i]
                    gt_pc = numpy_to_pc(gt_points)
                    gt_path = os.path.join(save_path,f"{name}_gt.xyz")
                    open3d.io.write_point_cloud(filename=gt_path,pointcloud=gt_pc)
                    print(f"---- saving truth dense point cloud: {gt_path} ----")
                # ---- gt ----

                # ---- condition ----
                if(save_condition):
                    condition_points = condition_np[i]
                    condition_pc = numpy_to_pc(condition_points)
                    condition_path = os.path.join(save_path,f"{name}_condition.xyz")
                    open3d.io.write_point_cloud(filename=condition_path,pointcloud=condition_pc)
                    print(f"---- saving input sparse point cloud: {condition_path} ----")
                # ---- condition ----
            # ---- save data ----

    total_meta = total_meta.detach().cpu().numpy()
    print(f"Times : {times}")
    if return_all_metrics:
        return CD_meter.avg, HD_meter.avg, P2F_meter.avg, total_meta, metrics
    else:
        return CD_meter.avg, HD_meter.avg, P2F_meter.avg, total_meta, metrics['cd_distance']



