import os

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np


def show_collage(imgs, rows, cols, scores):
    fix, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(cols*5, rows*5), dpi=150)
    n = len(imgs)
    i = 0
    add_title_score = True if scores is not None else False
    for r in range(rows):
        for c in range(cols):
            ax[r, c].imshow(imgs[i])
            ax[r, c].set_xticks([])
            ax[r, c].set_yticks([])
            if add_title_score:
                title = "Image %i, score=%.3f" % (i, scores[i])
            else:
                title = "Image %i" % i
            ax[r, c].set_title(title, fontsize=14)                    
            i += 1
            if i == n:
                break
        if i == n:
            break
    plt.show()
    
    
def save_grid(fname, images, nrows, ncols, title, img_titles=None):
    """
    Should only be used when there are titles for each image, as it's very slow.
    Otherwise use 'fast_save_grid'.
    Note: '.jpg'
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20,20), dpi=150)
    add_img_title = img_titles is not None
    for idx, image in enumerate(images):
        row = idx // nrows
        col = idx % ncols
        axes[row, col].axis("off")
        axes[row, col].imshow(image, cmap="gray", aspect="auto")
        if add_img_title:
            axes[row, col].set_title(str(img_titles[idx]), fontsize=15)
    fig.suptitle(title, fontsize=20)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.subplots_adjust(wspace=.05, hspace=.2)
    plt.savefig(fname + '.jpg', bbox_inches='tight')
    plt.close()
    
    
def fast_make_grid(images, nrows, ncols, padding):
    _, h, w, _ = images.shape
    grid_h = nrows * h + (nrows - 1) * padding
    grid_w = ncols * w + (ncols - 1) * padding
    
    image_grid = np.zeros((grid_h, grid_w, 3), dtype=images.dtype)
    hp = h + padding
    wp = w + padding

    i = 0
    for r in range(nrows):
        for c in range(ncols):
            image_grid[hp*r : hp*(r + 1) - padding, wp*c : wp*(c + 1) - padding, :] = images[i]
            i += 1
                 
    return image_grid   


def plt_save_grid(fname, images, nrows, ncols, padding, title):
    """
    Should only be used for benchmarks.
    If you have title for each image then use 'save_grid' otherwise 'fast_save_grid'
    """
    img_grid = fast_make_grid(images, nrows=nrows, ncols=ncols, padding=padding)
    fig, ax = plt.subplots(figsize=(20,20), dpi=150)
    ax.imshow(img_grid)
    ax.axis("off")
    ax.set_title(title, fontsize=15)
    plt.savefig(fname + '.jpg', bbox_inches='tight', quality=80)
    plt.close()       


def add_title_background(img_array):
    h, w, _ = img_array.shape
    background = np.zeros([int(0.05 * h), w, 3], dtype=img_array.dtype)
    return np.vstack([background, img_array])


def convert_to_pil_image(images):
    """
    :param images: numpy array of dtype=uint8 in range [0, 255]
    :return: PIL image
    """
    return Image.fromarray(images)


def convert_to_pil_image_with_title(img_array, title):
    h, w, _ = img_array.shape
    img_array = add_title_background(img_array)
    img = convert_to_pil_image(img_array)

    # See function add_title_background
    font_size = int(0.025 * h)
    # Font can be stored in a folder with script
    font = ImageFont.truetype("arial.ttf", font_size)

    d = ImageDraw.Draw(img)
    # text_w, text_h = d.textsize(title)
    text_w_start_pos = (w - font.getsize(title)[0]) / 2
    d.text((text_w_start_pos, 0.01 * h), title, fill="white", font=font)
    return img


def fast_save_grid(out_dir, fname, images, nrows, ncols, padding, title, save_in_jpg=False):
    img_grid = fast_make_grid(images, nrows=nrows, ncols=ncols, padding=padding)
    if title is not None:
        img = convert_to_pil_image_with_title(img_grid, title)
    else:
        img = convert_to_pil_image(img_grid)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if save_in_jpg:
        img.save(os.path.join(out_dir, fname + '.jpg'), 'JPEG', quality=95, optimize=True)
    else:
        img.save(os.path.join(out_dir, fname + '.png'), 'PNG')
