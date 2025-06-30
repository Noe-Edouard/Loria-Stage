import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from pathlib import Path


class Viewer():
    def __init__(self):
        pass

    def _normalize_inputs(self, images, titles):
        if isinstance(images, np.ndarray):
            images = [images]
        num_images = len(images)
        if isinstance(titles, str):
            titles = [titles]
        if titles is None:
            titles = [f"Image {i+1}" for i in range(num_images)]
        elif len(titles) != num_images:
            raise ValueError("Number of titles must match number of images")
        return images, titles, num_images

    def is_binary_image(self, img: np.ndarray) -> bool:
        unique_vals = np.unique(img)
        return np.array_equal(unique_vals, [0, 1]) or np.array_equal(unique_vals, [0]) or np.array_equal(unique_vals, [1])

    def display_images(self, images: list[np.ndarray] | np.ndarray, titles: list[str] | str = None, cmap='gray'):
        images, titles, num_images = self._normalize_inputs(images, titles)
        fig = plt.figure(figsize=(5 * num_images, 5))
        for i, img in enumerate(images):
            plt.subplot(1, num_images, i + 1)
            plt.imshow(img, cmap=cmap)
            plt.title(titles[i])
            plt.axis('off')
            plt.colorbar()
        plt.tight_layout()
        plt.show()
        return fig

    def display_histogram(self, images: list[np.ndarray] | np.ndarray, titles: list[str] = None, bins: int = 50, density: bool = False, color='blue'):
        images, titles, num_images = self._normalize_inputs(images, titles)

        cols = min(3, num_images)
        rows = (num_images + cols - 1) // cols

        fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axs = np.atleast_1d(axs).flatten()

        for i, image in enumerate(images):
            axs[i].hist(image.ravel(), bins=bins, density=density, color=color, alpha=0.7)
            axs[i].set_title(titles[i])
            axs[i].set_xlabel('Intensity')
            axs[i].set_ylabel('Density' if density else 'Frequency')
            axs[i].grid(True)

        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        plt.show()
        return fig

    def display_mip(self, images: list[np.ndarray] | np.ndarray, titles: list[str] | str = None, cmap='gray'):
        images, titles, num_images = self._normalize_inputs(images, titles)

        fig, axs = plt.subplots(
            num_images, 4, figsize=(15, 4 * num_images),
            gridspec_kw={'width_ratios': [0.08, 1, 1, 1.3], 'wspace': 0.05}
        )
        axs = np.expand_dims(axs, axis=0) if num_images == 1 else axs

        for i, image in enumerate(images):
            if image.ndim != 3:
                raise ValueError(f"Image {i} must be 3D. Current shape: {image.shape}.")

            mips = [np.max(image, axis=ax) for ax in (0, 1, 2)]
            titles_mip = ['x-axis', 'y-axis', 'z-axis']

            axs[i, 0].text(1, 0.5, titles[i], rotation=90,
                           verticalalignment='center', horizontalalignment='center',
                           fontsize=16)
            axs[i, 0].axis('off')

            for j, (mip, label) in enumerate(zip(mips, titles_mip)):
                ax = axs[i, j + 1]
                im = ax.imshow(mip, cmap=cmap)
                ax.set_title(f'MIP along {label}')
                ax.axis('off')
            fig.colorbar(im, ax=axs[i, 3], shrink=0.98, pad=0.1).ax.tick_params(labelsize=8)

        plt.tight_layout()
        plt.show()
        return fig

    def display_volume(self, volume, threshold=0, cmap='plasma'):
        x, y, z = np.where(volume > threshold)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(x, y, z, s=0.5, marker='.', c=volume[x, y, z], cmap=cmap)

        ax.set(xlabel='X', ylabel='Y', zlabel='Z', title=f'3D plot with threshold {threshold}')
        fig.colorbar(sc, ax=ax, shrink=0.6, label='Intensity')
        plt.tight_layout()
        plt.show()
        return fig

    def display_slices(self, images: list[np.ndarray] | np.ndarray, titles: list[str] | str = None, interval=10, cmap='gray'):
        images, titles, num_images = self._normalize_inputs(images, titles)
        directions = [0, 1, 2]

        def get_frame(img, d, idx):
            return img[idx, :, :] if d == 0 else img[:, idx, :] if d == 1 else img[:, :, idx]

        fig, axes = plt.subplots(
            num_images, 4, figsize=(16, 4 * num_images),
            gridspec_kw={'width_ratios': [0.1, 1, 1, 1.2], 'wspace': 0.05}
        )
        axes = np.expand_dims(axes, axis=0) if num_images == 1 else axes

        num_frames = min(min(img.shape[d] for d in directions) for img in images)
        plots, subplot_titles = [], []

        for i, img in enumerate(images):
            is_bin = self.is_binary_image(img)
            cmap_used = 'binary' if is_bin else cmap

            axes[i, 0].text(1, 0.5, titles[i], rotation=90,
                            verticalalignment='center', horizontalalignment='center', fontsize=16)
            axes[i, 0].axis('off')

            row_plots, row_titles = [], []
            for d in directions:
                ax = axes[i, d + 1]
                frame = get_frame(img, d, 0)
                im = ax.imshow(frame, cmap=cmap_used, **({'vmin': 0, 'vmax': 1} if is_bin else {}))
                ax.set_title(f'Axis {d} - Slice 1/{img.shape[d]}')
                ax.axis('off')
                row_plots.append(im)
                row_titles.append(ax.title)

            plots.append(row_plots)
            subplot_titles.append(row_titles)

        cbar = fig.colorbar(row_plots[-1], ax=axes[:, 3], shrink=0.98, pad=0.1,
                            ticks=[0, 1] if is_bin else None)
        if is_bin:
            cbar.ax.set_yticklabels(['0', '1'])
        cbar.ax.tick_params(labelsize=8)

        def update(frame_idx):
            for i, img in enumerate(images):
                for d in directions:
                    if frame_idx < img.shape[d]:
                        plots[i][d].set_array(get_frame(img, d, frame_idx))
                        subplot_titles[i][d].set_text(f'Axis {d} - Slice {frame_idx + 1}/{img.shape[d]}')
            return sum(plots, []) + sum(subplot_titles, [])

        animation = ani.FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=False)
        plt.tight_layout()
        plt.show()
        return animation
