"""
This file contain functions to visualize the point cloud, 
mesh or voxels as gifs. 

"""

import torch
from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
)
from pytorch3d.io import load_obj

from PIL import Image, ImageDraw
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import pytorch3d
import numpy as np

# import for rendering gif
import imageio

def get_device():
    """
    Checks if GPU is available and returns device accordingly.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device

def get_points_renderer(
    image_size=512, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer

def get_mesh_renderer(image_size=512, lights=None, device=None):
    """
    Returns a Pytorch3D Mesh Renderer.

    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer

def unproject_depth_image(image, mask, depth, camera):
    """
    Unprojects a depth image into a 3D point cloud.

    Args:
        image (torch.Tensor): A square image to unproject (S, S, 3).
        mask (torch.Tensor): A binary mask for the image (S, S).
        depth (torch.Tensor): The depth map of the image (S, S).
        camera: The Pytorch3D camera to render the image.
    
    Returns:
        points (torch.Tensor): The 3D points of the unprojected image (N, 3).
        rgba (torch.Tensor): The rgba color values corresponding to the unprojected
            points (N, 4).
    """
    device = camera.device
    assert image.shape[0] == image.shape[1], "Image must be square."
    image_shape = image.shape[0]
    ndc_pixel_coordinates = torch.linspace(1, -1, image_shape)
    Y, X = torch.meshgrid(ndc_pixel_coordinates, ndc_pixel_coordinates)
    xy_depth = torch.dstack([X, Y, depth])
    points = camera.unproject_points(
        xy_depth.to(device), in_ndc=False, from_ndc=False, world_coordinates=True,
    )
    points = points[mask > 0.5]
    rgb = image[mask > 0.5]
    rgb = rgb.to(device)

    # For some reason, the Pytorch3D compositor does not apply a background color
    # unless the pointcloud is RGBA.
    alpha = torch.ones_like(rgb)[..., :1]
    rgb = torch.cat([rgb, alpha], dim=1)

    return points, rgb

def render_gif_from_pc(points, output_path, colors=[1, 0.5, 0], image_size=256,background_color=(1, 1, 1),device=None, distance = 1):
    
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    device = get_device()

    verts = torch.Tensor(points).to(device)
    colors = torch.full(points.shape, 0.5).to(device)

    # import pdb; pdb.set_trace()
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=colors)
    
    # Place a point light in front of the object
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]],
                                            device=device)
    
    images = []
    for i in tqdm(range(0, 360, 10)):
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=distance,
                                                                 azim=i,
                                                                 device=device)

        # Prepare the camera:
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )
        rend = renderer(point_cloud, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)
        rend = (rend * 255).astype(np.uint8)
        images.append(rend)

    imageio.mimsave(output_path, images, duration = 60, loop = 0)

def render_gif_from_mesh(mesh, output_path, distance = 1.0, image_size=256, device = None):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()
    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)
    
    verts = torch.tensor(mesh.verts_list()[0])
    faces = torch.tensor(mesh.faces_list()[0])

    verts = verts.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)

    # defining texture: 

    # color1 = [1, 0.6, 0.1]
    # color2 = [0.055, 0.6, 0.91]
    # z_min = verts[:,:,2].min()
    # z_max = verts[:,:,2].max()
    # alpha = (verts[:, :, 2] - z_min) / (z_max - z_min)
    # color = alpha[:, :, None] * torch.tensor(color2).to(device) + (1 - alpha[:, :, None]) * torch.tensor(color1).to(device)

    textures = torch.full(verts.shape, 0.5).to(device)  # (1, N_v, 3)


    mesh = pytorch3d.structures.Meshes(
        verts=verts,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )

    mesh = mesh.to(device)
    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
    images = []
    for i in tqdm(range(0, 360,10)):

        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist =distance, 
                                                                azim = i, 
                                                                device=device)
        
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )

        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        # The .cpu moves the tensor to GPU (if needed).
        rend =  (rend * 255).astype(np.uint8)
        images.append(rend)
    imageio.mimsave(output_path, images, duration=65, loop=0)

def render_gif_from_voxel(voxel, output_path, distance = 3.0, image_size=256, device = None):

    if device is None:
        device = get_device()

    renderer = get_mesh_renderer(image_size=image_size)

    # cubify converts the given voxel to mesh surface and the 
    # threshold defines occupancy threshld to consider it a occupied position 
    cubemesh = pytorch3d.ops.cubify(voxel, thresh = 0.2)
    verts = torch.tensor(cubemesh.verts_list()[0])
    faces = torch.tensor(cubemesh.faces_list()[0])

    verts = verts.unsqueeze(0)
    faces = faces.unsqueeze(0)
    # defining texture: 
    # color1 = [1, 0.6, 0.1]
    # color2 = [0.055, 0.6, 0.91]
    # z_min = verts[:,:,2].min()
    # z_max = verts[:,:,2].max()
    # alpha = (verts[:, :, 2] - z_min) / (z_max - z_min)
    # color = alpha[:, :, None] * torch.tensor(color2).to(device) + (1 - alpha[:, :, None]) * torch.tensor(color1).to(device)
    textures = torch.full(verts.shape, 0.5).to(device)

    mesh = pytorch3d.structures.Meshes(
        verts = verts, 
        faces = faces,
        textures = pytorch3d.renderer.TexturesVertex(textures))

    mesh = mesh.to(device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    images = []
    for i in tqdm(range(0, 360,10)):

        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist =distance, 
                                                                azim = i, 
                                                                device=device)
        
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )

        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        # The .cpu moves the tensor to GPU (if needed).
        rend =  (rend * 255).astype(np.uint8)
        images.append(rend)
    imageio.mimsave(output_path, images, duration=65, loop=0)

