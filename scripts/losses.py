import torch
import pytorch3d.loss

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# implement some loss for binary voxel grids

	# creating an instance of BCEWithLogitsLoss()
	# BCE with logits is numerically more stable then sigmoid on BCELoss. 
	criterion = torch.nn.BCEWithLogitsLoss()
	# calculating the loss by passing predicted and target values.
	loss = criterion(voxel_src, voxel_tgt)
	return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# implement chamfer loss from scratch

	# torch.ops.knn_points return the tensor of size (b*n_points*K) giving the squared distance. 
	loss_src_tgt, idx1, b1 = pytorch3d.ops.knn_points(point_cloud_src, point_cloud_tgt, K=1)
	loss_tgt_src, idx2, b2 = pytorch3d.ops.knn_points(point_cloud_tgt, point_cloud_src, K=1)
	loss_chamfer = torch.sum(loss_src_tgt + loss_tgt_src)
	return loss_chamfer

def smoothness_loss(mesh_src): 
	# implement laplacian smoothening loss
	# using pytorch.loss.mesh_laplacian_smoothing
	loss_laplacian = pytorch3d.loss.mesh_laplacian_smoothing(mesh_src, method = "uniform")
	return loss_laplacian