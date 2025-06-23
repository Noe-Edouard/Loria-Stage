import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from mpl_toolkits.mplot3d import Axes3D

def display_mip(images: list[np.ndarray], cmap='gray'):
    num_images = len(images)
    plt.figure(figsize=(16, 4 * num_images))

    for i, image in enumerate(images):
        if image.ndim != 3:
            raise ValueError(f"Image {i} must be 3D. Current shape: {image.shape}.")

        mip_x = np.max(image, axis=0)
        mip_y = np.max(image, axis=1)
        mip_z = np.max(image, axis=2)

        # X projection
        plt.subplot(num_images, 3, i * 3 + 1)
        plt.imshow(mip_x, cmap=cmap)
        plt.title(f'MIP X - Image {i+1}')
        plt.axis('off')

        # Y projection
        plt.subplot(num_images, 3, i * 3 + 2)
        plt.imshow(mip_y, cmap=cmap)
        plt.title(f'MIP Y - Image {i+1}')
        plt.axis('off')

        # Z projection
        plt.subplot(num_images, 3, i * 3 + 3)
        plt.imshow(mip_z, cmap=cmap)
        plt.title(f'MIP Z - Image {i+1}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def display_volume(volume, threshold=0, cmap='plasma'):
    # Select relevant points
    x, y, z = np.where(volume > threshold)
    
    # Display 3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, s=0.5, marker='.', edgecolors=None, c=volume[x, y, z], cmap=cmap)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D plot with threshold {threshold}')
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, label='Intensity')
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np

def display_animation(images: list[np.ndarray], interval=10):
    def get_frame(image, direction, index):
        if direction == 0:
            return image[index, :, :]
        elif direction == 1:
            return image[:, index, :]
        elif direction == 2:
            return image[:, :, index]

    num_images = len(images)
    directions = [0, 1, 2]
    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))

    if num_images == 1:
        axes = np.expand_dims(axes, 0)  # ensure axes is 2D

    num_frames = min([
        min(image.shape[dir] for dir in directions)
        for image in images
    ])

    plots = []
    titles = []

    for i, image in enumerate(images):
        row_plots = []
        row_titles = []
        for d in directions:
            ax = axes[i, d]
            im = ax.imshow(get_frame(image, d, 0), cmap='gray')
            title = ax.set_title(f"Image {i+1} - Axis {d} - Slice 1/{image.shape[d]}")
            row_plots.append(im)
            row_titles.append(title)
        plots.append(row_plots)
        titles.append(row_titles)

    def update(frame_idx):
        for i, image in enumerate(images):
            for d in directions:
                if frame_idx < image.shape[d]:
                    plots[i][d].set_array(get_frame(image, d, frame_idx))
                    titles[i][d].set_text(f"Image {i+1} - Axis {d} - Slice {frame_idx+1}/{image.shape[d]}")
        return sum(plots, []) + sum(titles, [])

    animation = ani.FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=False)
    plt.tight_layout()
    plt.show()



    