# import matplotlib.pyplot as plt
# import numpy as np
#
# # 读取三张图像
# image1 = plt.imread(r'E:\Code_Github\hongwai_tance\SIRST\Infrared-Small-Target-Detection-master\NUDT-DNA-weights\AB-NUDT-DNA-twoflowadd.png')
# image2 = plt.imread(r'E:\Code_Github\hongwai_tance\SIRST\Infrared-Small-Target-Detection-master\NUDT-DNA-weights\NUDT55_DNA_oneflow.png')
# image3 = plt.imread(r'E:\Code_Github\hongwai_tance\SIRST\Infrared-Small-Target-Detection-master\NUDT-DNA-weights\NUDT-DNA-twoflownew.png')
#
# # 创建一个Figure对象和包含三个子图的Axes对象数组
# fig, axs = plt.subplots(1, 3, figsize=(12, 4))
#
# # 在每个子图中显示图像
# axs[0].imshow(image1)
# axs[0].axis('off')
# axs[1].imshow(image2)
# axs[1].axis('off')
# axs[2].imshow(image3)
# axs[2].axis('off')
#
# # 在每个子图后面添加colorbar
# cbar1 = fig.colorbar(axs[0].imshow(image1), ax=axs[0])
# cbar2 = fig.colorbar(axs[1].imshow(image2), ax=axs[1])
# cbar3 = fig.colorbar(axs[2].imshow(image3), ax=axs[2])
#
# # 设置colorbar的标签
# cbar1.set_label('Label 1')
# cbar2.set_label('Label 2')
# cbar3.set_label('Label 3')
#
# # 调整子图和colorbar的布局
# fig.tight_layout()
#
# # 保存图像
# plt.savefig('images_with_colorbar.png')


#####单张
import matplotlib.pyplot as plt

# # 读取图像
# image = plt.imread(r'E:\Code_Github\hongwai_tance\SIRST\Infrared-Small-Target-Detection-master\NUDT-DNA-weights\AB-NUDT-DNA-twoflowadd.png')
#
# # 创建一个Figure对象和一个Axes对象
# fig, ax = plt.subplots(figsize=(4, 6))
#
# # 在Axes对象中绘制示例图像
# im = ax.imshow(image, cmap='jet')
# ax.axis('off')
#
# # 添加colorbar
# cbar = fig.colorbar(im, ax=ax)
#
# # 保存colorbar
# plt.savefig('colorbar.png')

# ###color归一化
# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import numpy as np
#
# # 读取图像
# image = plt.imread(r'E:\Code_Github\hongwai_tance\SIRST\Infrared-Small-Target-Detection-master\NUDT-UIU-weights\AB-NUDT-UIU-twoflowadd.png')
#
# # 创建一个Figure对象和一个包含两行一列的GridSpec对象
# fig = plt.figure(figsize=(6, 8))
# gs = gridspec.GridSpec(2, 1, height_ratios=[8, 1])
#
# # 在第一行的子图中显示图像
# ax_image = fig.add_subplot(gs[0])
# ax_image.imshow(image, cmap='jet')
# ax_image.axis('off')
#
# # 在第二行的子图中显示colorbar
# ax_cbar = fig.add_subplot(gs[1])
# cbar = fig.colorbar(ax_image.imshow(image, cmap='jet'), cax=ax_cbar, orientation='horizontal')
#
# # 调整子图和colorbar的间距
# plt.subplots_adjust(hspace=0.01)
#
# # 保存图像
# plt.savefig('image_with_colorbar.png')

##################colorbar右边############

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# 读取图像
image = plt.imread(r'E:\Code_Github\hongwai_tance\SIRST\Infrared-Small-Target-Detection-master\NUDT-UIU-weights\AB-NUDT-UIU-twoflowadd.png')

# 创建一个Figure对象和一个包含一行两列的GridSpec对象
fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[8, 1])

# 在第一列的子图中显示图像
ax_image = fig.add_subplot(gs[0])
ax_image.imshow(image, cmap='jet')
ax_image.axis('off')

# 在第二列的子图中显示colorbar
ax_cbar = fig.add_subplot(gs[1])
cax = ax_cbar.inset_axes([0.2, 0.1, 0.05, 0.8])  # 调整colorbar的位置和尺寸
cbar = fig.colorbar(ax_image.imshow(image, cmap='jet'), cax=cax, fraction=0.05)

# 调整子图和colorbar的间距
plt.subplots_adjust(wspace=0.1)

# 保存图像
plt.savefig('image_with_colorbar.png')

#######################
# import cv2
# import matplotlib.pyplot as plt
#
# # 读取本地图片
# pred_np_normalized = cv2.imread(r'E:\Code_Github\hongwai_tance\SIRST\Infrared-Small-Target-Detection-master\NUDT-UIU-weights\AB-NUDT-UIU-twoflowadd.png', cv2.IMREAD_COLOR)
#
# # 转换颜色通道顺序 BGR -> RGB
# pred_np_rgb = cv2.cvtColor(pred_np_normalized, cv2.COLOR_BGR2RGB)
#
# # 创建图像和坐标轴
# fig, ax = plt.subplots()
#
# # 显示RGB图像
# cax = ax.imshow(pred_np_rgb)
#
# # 添加位于图像右侧的colorbar
# cbar = fig.colorbar(cax)
#
# # 设置colorbar的标签
# cbar.set_label('Pixel Values')
#
# # 保存图像
# plt.savefig('pred_rgb.png')
#
# # 关闭图像以释放资源
# plt.close(fig)