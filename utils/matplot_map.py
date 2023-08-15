import matplotlib.pyplot as plt
import numpy as np

def plot_together(raw, true_img, predict_img, path):
    # Create subplots with 1 row and 3 columns
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the first image
    axs[0].imshow(raw, aspect='auto')
    axs[0].set_title('Raw Image')
    axs[0].axis('off')

    # Plot the second image
    axs[1].imshow(true_img, aspect='auto')
    axs[1].set_title('Label Image')
    axs[1].axis('off')

    # Plot the third image
    axs[2].imshow(predict_img, aspect='auto')
    axs[2].set_title('Predict Image')
    axs[2].axis('off')

    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    fig.savefig(path)  # 保存图像到文件


