import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from mpl_toolkits.mplot3d import Axes3D


class Viewer():
    
    def __init__(self):
        pass
    

    def display_images(self, images: list[np.ndarray] | np.ndarray, titles: list[str] | str = None, cmap='gray'):
        
        if titles is None:
            titles = [f'Image {i+1}' for i in range(len(images))]
            
        if isinstance(images, np.ndarray):
            images = [images]
            
        
        num_images = len(images)
        plt.figure(figsize=(5 * num_images, 5))
        for i, img in enumerate(images):
            plt.subplot(1, num_images, i + 1)
            plt.imshow(img, cmap=cmap)
            plt.title(titles[i])
            plt.axis('off')
            plt.colorbar()
        plt.tight_layout()
        plt.show()

    
    def display_mip(self, images: list[np.ndarray] | np.ndarray, titles: list[str] | str = None, cmap='gray'):
        if isinstance(images, np.ndarray):
            images = [images]
        num_images = len(images)    
        
        if isinstance(titles, str):
            titles = [titles]
        if titles is None:
            titles = [f'Image {i+1}' for i in range(len(images))]
        elif len(titles) != num_images:
            raise ValueError("The number of images does not match the number of titles.")
        
        fig, axs = plt.subplots(
            num_images, 4, figsize=(15, 4 * num_images),
            gridspec_kw={'width_ratios': [0.08, 1, 1, 1.2], 'wspace': 0.05}
        )
        if num_images == 1:
            axs = np.expand_dims(axs, axis=0)

       
        for i, image in enumerate(images):
            if image.ndim != 3:
                raise ValueError(f"Image {i} must be 3D. Current shape: {image.shape}.")

            mip_x = np.max(image, axis=0)
            mip_y = np.max(image, axis=1)
            mip_z = np.max(image, axis=2)

            # Title column
            title_ax = axs[i, 0]
            title_ax.text(
                1, 0.5, titles[i], horizontalalignment='center',
                verticalalignment='center', rotation=90,fontsize=16
            )
            title_ax.axis('off')

            # MIP projections
            axs[i, 1].imshow(mip_x, cmap=cmap)
            axs[i, 1].set_title('MIP along x-axis')
            axs[i, 1].axis('off')

            axs[i, 2].imshow(mip_y, cmap=cmap)
            axs[i, 2].set_title('MIP along y-axis')
            axs[i, 2].axis('off')

            im = axs[i, 3].imshow(mip_z, cmap=cmap)
            axs[i, 3].set_title('MIP along z-axis')
            axs[i, 3].axis('off')

            # Add colorbar to the right of the last plot in the row
            cbar = fig.colorbar(im, ax=axs[i, 3], shrink=0.98, pad=0.1)
            cbar.ax.tick_params(labelsize=8)

        plt.tight_layout()
        plt.show()


    def display_volume(self, volume, threshold=0, cmap='plasma'):
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


    def display_slices(self, images: list[np.ndarray] | np.ndarray, titles: list[str] | str = None, interval=10, cmap='gray'):

        if isinstance(images, np.ndarray):
            images = [images]
        num_images = len(images)    

        if isinstance(titles, str):
            titles = [titles]
        if titles is None:
            titles = [f'Image {i+1}' for i in range(len(images))]
        elif len(titles) != num_images:
            raise ValueError("The number of images does not match the number of titles.")

        directions = [0, 1, 2]

        def get_frame(image, direction, index):
            if direction == 0:
                return image[index, :, :]
            elif direction == 1:
                return image[:, index, :]
            elif direction == 2:
                return image[:, :, index]

        fig, axes = plt.subplots(
            num_images, 4, figsize=(15, 4 * num_images),
            gridspec_kw={'width_ratios': [0.08, 1, 1, 1.2], 'wspace': 0.05}
        )

        if num_images == 1:
            axes = np.expand_dims(axes, axis=0)

        num_frames = min([
            min(image.shape[dir] for dir in directions)
            for image in images
        ])

        plots = []  # stores imshow objects
        subplot_titles = []  # ← Renommé pour éviter conflit

        for i, image in enumerate(images):
            title_ax = axes[i, 0]
            title_ax.text(1, 0.5, f"{titles[i]}", rotation=90,
                        verticalalignment='center', horizontalalignment='center',
                        fontsize=16)
            title_ax.axis('off')

            row_plots = []
            row_titles = []
            for d in directions:
                ax = axes[i, d + 1]
                im = ax.imshow(get_frame(image, d, 0), cmap=cmap)
                ax.set_title(f'Axis {d} - Slice 1/{image.shape[d]}')
                ax.axis('off')
                row_plots.append(im)
                row_titles.append(ax.title)
            plots.append(row_plots)
            subplot_titles.append(row_titles)

            cbar = fig.colorbar(row_plots[-1], ax=axes[i, 3], shrink=0.98, pad=0.1)
            cbar.ax.tick_params(labelsize=8)

        def update(frame_idx):
            for i, image in enumerate(images):
                for d in directions:
                    if frame_idx < image.shape[d]:
                        plots[i][d].set_array(get_frame(image, d, frame_idx))
                        subplot_titles[i][d].set_text(f'Axis {d} - Slice {frame_idx + 1}/{image.shape[d]}')
            return sum(plots, []) + sum(subplot_titles, [])

        animation = ani.FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=False)
        plt.tight_layout()
        plt.show()





    def display_histogram(self, images: list[np.ndarray] | np.ndarray, titles: list[str] = None, bins: int = 50, density: bool = False, color='dodgerblue'):
        if isinstance(images, np.ndarray):
            images = [images]

        num_images = len(images)

        if titles is None:
            titles = [f"Image {i+1}" for i in range(num_images)]
        elif len(titles) != num_images:
            raise ValueError("Number of titles must match number of images")

        cols = min(3, num_images)
        rows = (num_images + cols - 1) // cols

        fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axs = np.atleast_1d(axs).flatten()

        for i, image in enumerate(images):
            data = image.ravel()
            axs[i].hist(data, bins=bins, density=density, color=color, alpha=0.7)
            axs[i].set_title(titles[i])
            axs[i].set_xlabel('Intensity')
            axs[i].set_ylabel('Frequency' if not density else 'Density')
            axs[i].grid(True)

        # Supprimer axes vides s’il y en a
        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        plt.show()