import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from torch.autograd import Function
from typing import *
from pytorch3d.ops import knn
import math

try:
    import pointnet2_ops._ext as _ext
except ImportError:
    from torch.utils.cpp_extension import load
    import glob
    import os.path as osp
    import os

    warnings.warn("Unable to load pointnet2_ops cpp extension. JIT Compiling.")

    _ext_src_root = osp.join(osp.dirname(__file__), "_ext-src")
    _ext_sources = glob.glob(osp.join(_ext_src_root, "src", "*.cpp")) + glob.glob(
        osp.join(_ext_src_root, "src", "*.cu")
    )
    _ext_headers = glob.glob(osp.join(_ext_src_root, "include", "*"))

    os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"
    _ext = load(
        "_ext",
        sources=_ext_sources,
        extra_include_paths=[osp.join(_ext_src_root, "include")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-Xfatbin", "-compress-all"],
        with_cuda=True,
    )

def get_neighbor_index(vertices: "(bs, vertice_num, 3)", neighbor_num: int):
    """
    Return: (bs, vertice_num, neighbor_num)
    """
    bs, v, _ = vertices.size()
    device = vertices.device
    inner = torch.bmm(vertices, vertices.transpose(1, 2))  # (bs, v, v)
    quadratic = torch.sum(vertices**2, dim=2)  # (bs, v)
    distance = inner * (-2) + quadratic.unsqueeze(1) + quadratic.unsqueeze(2)
    neighbor_index = torch.topk(
        distance, k=neighbor_num + 1, dim=-1, largest=False)[1]
    neighbor_index = neighbor_index[:, :, 1:]
    return neighbor_index


def get_nearest_index(target: "(bs, v1, 3)", source: "(bs, v2, 3)"):
    """
    Return: (bs, v1, 1)
    """
    inner = torch.bmm(target, source.transpose(1, 2))  # (bs, v1, v2)
    s_norm_2 = torch.sum(source**2, dim=2)  # (bs, v2)
    t_norm_2 = torch.sum(target**2, dim=2)  # (bs, v1)
    d_norm_2 = s_norm_2.unsqueeze(1) + t_norm_2.unsqueeze(2) - 2 * inner
    nearest_index = torch.topk(d_norm_2, k=1, dim=-1, largest=False)[1]
    return nearest_index


def indexing_neighbor(tensor: "(bs, vertice_num, dim)",
                      index: "(bs, vertice_num, neighbor_num)"):
    """
    Return: (bs, vertice_num, neighbor_num, dim)
    """
    bs, v, n = index.size()
    id_0 = torch.arange(bs).view(-1, 1, 1)
    tensor_indexed = tensor[id_0, index]
    return tensor_indexed


def get_neighbor_displacement(
        vertices: "(bs, vertice_num, 3)",
        neighbor_index: "(bs, vertice_num, neighbor_num)"):
    """
    Return: (bs, vertice_num, neighobr_num, 3)
    """
    neighbors = indexing_neighbor(vertices, neighbor_index)  # (bs, v, n, 3)
    neighbor_anchored = neighbors - vertices.unsqueeze(2)
    return neighbor_anchored

def count_to_mask(count, K):
    # counts is of shape (B, npoint)
    # its value range from 0 to K-1
    # return a mask of shape (B, npoint, K)
    mask = torch.arange(K, device=count.device, dtype=count.dtype)
    B, npoint = count.size()
    mask = mask.repeat(B, npoint).view(B, npoint,-1) # shape (B, npoint, K)
    mask = mask < count.unsqueeze(-1)
    return mask

def average_feature(feature, count, K):
    # feature is of shape (B, C , npoint, K) 
    # counts is of shape (B, npoint)
    # return an average feature of shape (B, C, npoint)
    if count == 'all':
        avg_feature = F.avg_pool2d(feature, kernel_size=[1, feature.size(3)])  # (B, mlp[-1], npoint, 1)
        avg_feature = avg_feature.squeeze(-1)  # (B, mlp[-1], npoint)
    else:
        count = torch.clamp(count, min=1)
        mask = count_to_mask(count, K)
        # mask is of shape (B, npoint, K)
        mask = mask.unsqueeze(1) # mask is of shape (B, 1, npoint, K)
        sum_feature = (feature * mask).sum(dim=-1) # (B, C, npoint)
        avg_feature = sum_feature / count.unsqueeze(1)
    return avg_feature

class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        # type: (Any, torch.Tensor, int) -> torch.Tensor
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance

        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set

        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        out = _ext.furthest_point_sampling(xyz, npoint)

        ctx.mark_non_differentiable(out)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        return ()


furthest_point_sample = FurthestPointSampling.apply


class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor

        idx : torch.Tensor
            (B, npoint) tensor of the features to gather

        Returns
        -------
        torch.Tensor
            (B, C, npoint) tensor
        """

        ctx.save_for_backward(idx, features)

        return _ext.gather_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        idx, features = ctx.saved_tensors
        N = features.size(2)

        grad_features = _ext.gather_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None


gather_operation = GatherOperation.apply


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown, known):
        # type: (Any, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
            Find the three nearest neighbors of unknown in known
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of known features
        known : torch.Tensor
            (B, m, 3) tensor of unknown features

        Returns
        -------
        dist : torch.Tensor
            (B, n, 3) l2 distance to the three nearest neighbors
        idx : torch.Tensor
            (B, n, 3) index of 3 nearest neighbors
        """
        dist2, idx = _ext.three_nn(unknown, known)
        dist = torch.sqrt(dist2)

        ctx.mark_non_differentiable(dist, idx)

        return dist, idx

    @staticmethod
    def backward(ctx, grad_dist, grad_idx):
        return ()


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features, idx, weight):
        # type(Any, torch.Tensor, torch.Tensor, torch.Tensor) -> Torch.Tensor
        r"""
            Performs weight linear interpolation on 3 features
        Parameters
        ----------
        features : torch.Tensor
            (B, c, m) Features descriptors to be interpolated from
        idx : torch.Tensor
            (B, n, 3) three nearest neighbors of the target features in features
        weight : torch.Tensor
            (B, n, 3) weights

        Returns
        -------
        torch.Tensor
            (B, c, n) tensor of the interpolated features
        """
        ctx.save_for_backward(idx, weight, features)

        return _ext.three_interpolate(features, idx, weight)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        grad_out : torch.Tensor
            (B, c, n) tensor with gradients of ouputs

        Returns
        -------
        grad_features : torch.Tensor
            (B, c, m) tensor with gradients of features

        None

        None
        """
        idx, weight, features = ctx.saved_tensors
        m = features.size(2)

        grad_features = _ext.three_interpolate_grad(
            grad_out.contiguous(), idx, weight, m
        )

        return grad_features, torch.zeros_like(idx), torch.zeros_like(weight)


three_interpolate = ThreeInterpolate.apply


class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor of features to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indicies of features to group with

        Returns
        -------
        torch.Tensor
            (B, C, npoint, nsample) tensor
        """
        ctx.save_for_backward(idx, features)

        return _ext.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""

        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) tensor of the gradients of the output from forward

        Returns
        -------
        torch.Tensor
            (B, C, N) gradient of the features
        None
        """
        idx, features = ctx.saved_tensors
        N = features.size(2)

        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)

        return grad_features, torch.zeros_like(idx)


grouping_operation = GroupingOperation.apply


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):
        # type: (Any, float, int, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        radius : float
            radius of the balls
        nsample : int
            maximum number of features in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        output, counts = _ext.ball_query(new_xyz, xyz, radius, nsample)

        ctx.mark_non_differentiable(output)

        return output, counts

    @staticmethod
    def backward(ctx, grad_out):
        return ()


ball_query = BallQuery.apply

class GruopFeature(nn.Module):
    def __init__(self, radius, nsample, use_xyz=True, 
                 support_num=None,
                 include_abs_coordinate=False,
                 include_center_coordinate=False,
                 neighbor_def='radius', mlp=None, mlp_pre =None):
        super().__init__()
        # arguments:
        self.in_channel = mlp_pre[-1]
        self.out_channel = mlp[0]
        self.support_num = support_num
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.include_abs_coordinate = include_abs_coordinate
        self.include_center_coordinate = include_center_coordinate
        self.neighbor_stats = None
        self.quantile = torch.tensor([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        self.neighbor_num_quantile = None
        self.neighbor_def = neighbor_def
        assert (self.neighbor_def == 'radius' or self.neighbor_def == 'nn')

        # parameters:
        self.relu = nn.ReLU(inplace=True)
        self.weights = nn.Parameter(
            torch.FloatTensor(self.in_channel, (support_num + 1) * self.out_channel))
        self.bias = nn.Parameter(
            torch.FloatTensor((support_num + 1) * self.out_channel))
        self.displacement = nn.Parameter(
            torch.FloatTensor(3, support_num * self.out_channel))
        
        # attention:
        bottleneck_channel = max(32, self.in_channel // 2)
        self.bottleneck_channel = bottleneck_channel

        self.conv_kv = nn.Conv2d(3, bottleneck_channel * 2, kernel_size=1)  # 输入为相对坐标
        self.conv_q = nn.Conv2d(3, bottleneck_channel, kernel_size=1)       # 输入为中心点的绝对坐标
        self.softmax = nn.Softmax(dim=-1)
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.out_channel * (self.support_num + 1))
        self.weights.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        self.displacement.data.uniform_(-stdv, stdv)

        # init attention:
        nn.init.xavier_uniform_(self.conv_kv.weight)
        nn.init.zeros_(self.conv_kv.bias)
        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.zeros_(self.conv_q.bias)

    def forward(self, xyz, vertices=None, feature_map=None,
                subset=True, record_neighbor_stats=False, 
                return_counts=False):
        """
        vertices: "(bs, vertice_num, 3)"
        feature_map: "(bs, vertice_num, in_channel)"
        """
        

        if self.neighbor_def == 'radius':
            neighbor_index = get_neighbor_index(vertices, self.nsample)
            bs, vertice_num, neighbor_num = neighbor_index.size()
            neighbor_displacement = get_neighbor_displacement(vertices, neighbor_index)
            
            normalized_xyz = neighbor_displacement.permute(0, 3, 1, 2)  # (bs, 3, vertice_num, neighbor_num)
            center_coords = vertices.permute(0, 2, 1).unsqueeze(-1)  # (bs, 3, vertice_num, 1)
            transformed_neighbor = self.conv_kv(normalized_xyz)  # (bs, 2 * bottleneck_channel, vertice_num, neighbor_num)
            transformed_new_point = self.conv_q(center_coords)   # (bs, bottleneck_channel, vertice_num, 1)

            transformed_feature1 = transformed_neighbor[:, :self.bottleneck_channel, :, :]  # Key: (bs, bottleneck_channel, vertice_num, neighbor_num)
            feature = transformed_neighbor[:, self.bottleneck_channel:, :, :]  # Value: (bs, bottleneck_channel, vertice_num, neighbor_num)

            # caculate theta
            # (bs, vertice_num, 1, bottleneck_channel) x (bs, vertice_num, bottleneck_channel, neighbor_num)
            theta = torch.matmul(transformed_new_point.permute(0, 2, 3, 1), transformed_feature1.permute(0, 2, 1, 3))
            # (bs, vertice_num, 1, neighbor_num)
            theta = theta / math.sqrt(self.bottleneck_channel)
            theta = self.softmax(theta)  # (bs, vertice_num, 1, neighbor_num)
            theta = theta.squeeze(2)  # (bs, vertice_num, neighbor_num)
            # theta = neighbor_displacement @ self.displacement
            # (bs, vertice_num, neighbor_num, support_num * out_channel)
            # theta = self.relu(theta)
            # theta = theta.contiguous().view(bs, vertice_num, neighbor_num, -1)
            # (bs, vertice_num, neighbor_num, support_num * out_channel)

            feature_out = feature_map @ self.weights + self.bias
            # (bs, vertice_num, (support_num + 1) * out_channel)
            # feature_center = feature_out[:, :, :self.out_channel]
            # (bs, vertice_num, out_channel)
            feature_support = feature_out[:, :, self.out_channel:]
            # (bs, vertice_num, support_num * out_channel)

            # Fuse together - max among product
            feature_support = indexing_neighbor(
                feature_support, neighbor_index
            )  
            # (bs, vertice_num, neighbor_num, support_num * out_channel)
            activation_support = theta.unsqueeze(-1) * feature_support  
            # (bs, vertice_num, neighbor_num, support_num * out_channel)
            activation_support = activation_support.view(
                bs, vertice_num, neighbor_num, self.support_num, self.out_channel)
            activation_support = torch.mean(activation_support, dim=3) 
            # (bs, vertice_num, neighbor_num, out_channel)
            activation_support = activation_support.permute(0, 3, 1, 2)
            # (bs, out_channel, vertice_num, neighbor_num)
        elif self.neighbor_def == 'nn':
            num_neighbors = min(self.nsample, xyz.shape[1])
            _, idx, _ = knn.knn_points(vertices, xyz, K=num_neighbors)
            # idx is of shape (B, npoint, K)
            idx = idx.int()
            B, npoint, K = idx.size()
            counts = torch.ones(B, npoint, device=vertices.device) * K
        else:
            raise Exception('Neighbor definition %s is not supported' % self.neighbor_def)

        if return_counts:
            if self.neighbor_def == 'nn':
                return activation_support, 'all'
            else:
                return activation_support, 'all'
        else:
            return activation_support

class QueryAndGroup(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz=True, include_abs_coordinate=False, 
                        include_center_coordinate=False, neighbor_def='radius'):
        # type: (QueryAndGroup, float, int, bool) -> None
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.include_abs_coordinate = include_abs_coordinate
        self.include_center_coordinate = include_center_coordinate
        self.neighbor_stats = None
        self.quantile = torch.tensor([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        self.neighbor_num_quantile = None
        self.neighbor_def = neighbor_def
        assert (self.neighbor_def == 'radius' or self.neighbor_def == 'nn')

    def forward(self, xyz, new_xyz, features=None, subset=True, record_neighbor_stats=False, return_counts=False):
        # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[Torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)
        subset : bool
            Indicate whether new_xyz is guaranteed to be a subset of xyz

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor if not include_abs_coordinate
            (B, 6 + C, npoint, nsample) tensor if include_abs_coordinate
        """
        if self.neighbor_def == 'radius':
            idx, counts = ball_query(self.radius, self.nsample, xyz, new_xyz)
        # counts contains the number of neighbors of every point in new_xyz
        # counts is of shape (B, npoint)
        # idx is of shape (B, npoint, K), counts is of shape (B, npoint)
        # idx[i,j,:] are k neighbors of new_xyz[i,j,:] in xyz
        # idx has repeated values, it repeats the first found neighbor when the number of neighbors are less than K
        # it doesn't matter which neighbor to repeat, because each neighbor is processed independently and maxpooling 
        # is applied among all neighbors at the last step of a pointnet
        # every point in new_xyz are guaranteed to have at least one neighbor if new_xyz is a subset of xyz
        # idx are initialized as zeros, if new_xyz have no neighbors in xyz, idx will be kept 0 as default
        elif self.neighbor_def == 'nn':
            num_neighbors = min(self.nsample, xyz.shape[1])
            _, idx, _ = knn.knn_points(new_xyz, xyz, K=num_neighbors)
            # idx is of shape (B, npoint, K)
            idx = idx.int()
            B, npoint, K = idx.size()
            counts = torch.ones(B, npoint, device=new_xyz.device) * K
        else:
            raise Exception('Neighbor definition %s is not supported' % self.neighbor_def)

        # pdb.set_trace()
        xyz_trans = xyz.transpose(1, 2).contiguous()
        abs_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, K), K=nsample
        if (not subset) and self.neighbor_def == 'radius':
            # in this case, new_xyz may not be a subset of xyz
            # we always assume for every element in new_xyz, it has at least one neighbor, which is it self
            # and it carrys a features of zeros of shape (C)
            # new_xyz (B, npoint, 3)
            # abs_xyz (B, 3, npoint, K)
            new_xyz_trans = new_xyz.transpose(1,2).unsqueeze(-1) # (B, 3, npoint, 1)
            have_neigh = (counts>0).float().unsqueeze(1).unsqueeze(-1).detach() # (B, 1, npoint, 1)
            no_neigh = (1-have_neigh)
            abs_xyz = have_neigh * abs_xyz + no_neigh*new_xyz_trans
            relative_xyz = abs_xyz - new_xyz_trans
        else:
            new_xyz_trans = new_xyz.transpose(1,2).unsqueeze(-1) # (B, 3, npoint, 1)
            relative_xyz = abs_xyz - new_xyz_trans

        if self.include_abs_coordinate:
            grouped_xyz = torch.cat([relative_xyz, abs_xyz], dim=1)
            # print('Use both relative position and absolute position')
        else:
            grouped_xyz = relative_xyz
            # print('Only use relative position')
        if self.include_center_coordinate:
            new_xyz_trans = new_xyz_trans.expand(-1,-1,-1,grouped_xyz.shape[3]) # (B, 3, npoint, K)
            grouped_xyz = torch.cat([grouped_xyz, new_xyz_trans], dim=1)


        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if not subset and self.neighbor_def == 'radius':
                # grouped_features (B, C, npoint, nsample)
                C = features.shape[1]
                default_feature = torch.zeros(C, device=features.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
                # default_feature (1, C, 1, 1)
                # have_neigh (B, 1, npoint, 1)
                grouped_features = have_neigh*grouped_features + no_neigh*default_feature

            if self.use_xyz:
                new_features = torch.cat([grouped_features, grouped_xyz], dim=1)  
                # (B, C + 3, npoint, nsample) or (B, C + 6, npoint, nsample) 
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        if record_neighbor_stats:
            with torch.no_grad():
                counts = counts.float()
                self.neighbor_stats = torch.stack([counts.min(), counts.mean(), counts.max()])
                self.neighbor_num_quantile = torch.quantile(counts, self.quantile.to(counts.device))
                self.neighbor_num_quantile = self.neighbor_num_quantile.long()

        # new_features is of shape (B, C + 3*f, npoint, K) 
        # counts is of shape (B, npoint)
        if return_counts:
            if self.neighbor_def == 'nn':
                return new_features, 'all'
            else:
                return new_features, counts
        else:
            return new_features


class GroupAll(nn.Module):
    r"""
    Groups all features

    Parameters
    ---------
    """

    def __init__(self, use_xyz=True):
        # type: (GroupAll, bool) -> None
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, features=None):
        # type: (GroupAll, torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            Ignored
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, C + 3, 1, N) tensor
        """

        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_features, grouped_xyz], dim=1
                )  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features


def group_knn(x, y, features_at_y, K, transpose=False):
    # we find k nearest neighbors for every point in x
    # x is of shape (B,N1,3), y is of shape (B,N2,3)
    # features_at_y are of shape (B,N2,C)
    if transpose:
        # features_at_y are of shape (B,C,N2)
        features_at_y_copy = features_at_y.transpose(1,2).contiguous()
    else:
        features_at_y_copy = features_at_y
    dist, idx, nn_abs_position = knn.knn_points(x, y, K=K, return_nn=True) # idx is of shape (B,N1,K), 
    x_neighbor_feas = knn.knn_gather(features_at_y_copy, idx) # (B, N1, K, C)
    x_repeat = x.unsqueeze(2).repeat(1,1,K,1) # (B,N1,K,3)
    nn_relative_position = nn_abs_position - x_repeat # (B,N1,K,3)
    dist = dist.unsqueeze(3) # (B,N1,K,1)
    dist_recip = 1.0 / (dist + 1e-8) # (B,N1,K,1)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm
    # x_neighbor_feas is of shape (B, N1, K, C)
    # dist is of shape (B,N1,K,1)
    # dist_recip is of shape (B,N1,K,1)
    # nn_abs_position is of shape (B,N1,K,3), positions of points in y
    # nn_relative_position is of shape (B,N1,K,3), positions of points in y
    # x_repeat is of shape (B,N1,K,3)
    new_features = torch.cat([x_neighbor_feas, dist, weight, nn_abs_position, nn_relative_position, x_repeat], dim=3)
    # new_features is of shape (B,N1,K,C+1+1+3+3+3) = (B,N1,K,C+11)
    if transpose:
        new_features = new_features.transpose(2,3).transpose(1,2) # (B,C+11,N1,K)
    return new_features

if __name__ == '__main__':
    import pdb
    N = 1024
    B = 10
    # C = 8
    # npoint = 512
    # radius = 0.2
    # K = 32
    # use_xyz = True
    # include_abs_coordinate = True

    # features = torch.rand(B,C,N).cuda()
    # xyz = (torch.rand(B,N,3).cuda()-0.5)*2
    # xyz_flipped = xyz.transpose(1, 2).contiguous() # shape (B,3,N)
    # idx = furthest_point_sample(xyz, npoint)
    # new_xyz = gather_operation(xyz_flipped, idx)
    # new_xyz = new_xyz.transpose(1, 2).contiguous()

    # # pdb.set_trace()
    # # new_xyz = (torch.rand(B,npoint,3).cuda()-0.5)*2
    # new_xyz = (torch.rand(B,N,3).cuda()-0.5)*2
    # q_group = QueryAndGroup(radius, K, use_xyz=use_xyz, include_abs_coordinate=include_abs_coordinate)
    # new_features = q_group(xyz, new_xyz, features) # shape (B,C+6,npoint,K)
    # q_group2 = QueryAndGroup(radius, K, use_xyz=use_xyz, include_abs_coordinate=include_abs_coordinate)
    # new_features2 = q_group2(xyz, new_xyz, features, subset=False) # shape (B,C+6,npoint,K)
    # x = torch.rand(B,N*2,3) # (B,N1,3)
    # y = torch.rand(B,N,3) # (B,N2,3)
    # features_at_y = torch.rand(B,N,128)
    # features_at_y = features_at_y.transpose(1,2)
    # K = 8
    # # find k nearest neighbors for every point in x
    # # the neighbors are found in y
    # # dist, idx, nn = knn.knn_points(x, y, K=K, return_nn=True)
    # # dist, nn_position, x_neighbor_feas = group_knn(x, y, features_at_y, K)
    # new_features = group_knn(x, y, features_at_y, K, transpose=True)

    xyz = torch.rand(B,N*2,3).cuda() # (B,N1,3)
    new_xyz = torch.rand(B,N,3).cuda() # (B,N2,3)
    features_at_y = torch.rand(B,128,2*N).cuda()
    grouper = QueryAndGroup(radius=0.2, nsample=32, use_xyz=True, include_abs_coordinate=True, include_center_coordinate=True, 
                            neighbor_def='nn')
    new_features = grouper(xyz, new_xyz, features=features_at_y, subset=True, record_neighbor_stats=False)
    # dist is of shape (B,N1,K), idx is of shape (B,N1,K), nn is of shape (B,N1,K,3)
    pdb.set_trace()