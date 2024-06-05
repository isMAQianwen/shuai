# Basic module
import matplotlib.pyplot as plt
# Torch and visulization
from torchvision      import transforms

# Metric, loss .etc
from model.utils import *
from model.loss import *
from model.load_param_data import load_param
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# Model
# from model.model_DNANet_improve3_chutuyong import  Res_CBAM_block
# #from model.model_ACM    import  ACM
# from model.model_DNANet_improve3_chutuyong import  DNANet
from model.model_DNANet_improve3_chutuyong import  Res_CBAM_block
#from model.model_ACM    import  ACM
from model.model_DNANet_improve3_chutuyong import  DNANet

def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Dense_Nested_Attention_Network_For_SIRST')
    # choose model
    parser.add_argument('--model', type=str, default='DNANet',
                        help='model name: DNANet,  ACM')

    # parameter for DNANet
    parser.add_argument('--channel_size', type=str, default='three',
                        help='one,  two,  three,  four')
    parser.add_argument('--backbone', type=str, default='resnet_18',
                        help='vgg10, resnet_10,  resnet_18,  resnet_34 ')
    parser.add_argument('--deep_supervision', type=str, default='True', help='True or False (model==DNANet), False(model==ACM)')

    # parameter for ACM
    parser.add_argument('--blocks', type=int, default=3, help='multiple block')
    parser.add_argument('--fuse_mode', type=str, default='AsymBi', help='fusion mode')

    # data and pre-process
    parser.add_argument('--img_demo_dir', type=str, default='img_demo',
                        help='img_demo')
    parser.add_argument('--img_demo_index', type=str,default='000019',
                        help='target1, target2, target3')
    parser.add_argument('--mode', type=str, default='TXT', help='mode name:  TXT, Ratio')
    parser.add_argument('--test_size', type=float, default='0.5', help='when --mode==Ratio')
    parser.add_argument('--suffix', type=str, default='.png')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='in_channel=3 for pre-process')
    parser.add_argument('--base_size', type=int, default=256,
                        help='base image size')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='crop image size')

    #  hyper params for training
    parser.add_argument('--test_batch_size', type=int, default=1,
                        metavar='N', help='input batch size for \
                        testing (default: 32)')

    # cuda and logging
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')

    args = parser.parse_args()

    # the parser
    return args

class Trainer(object):
    def __init__(self, args):

        # Initial
        self.args  = args
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)
        img_dir   = args.img_demo_dir+'/'+args.img_demo_index+args.suffix

        # Preprocess and load data
        input_transform = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        data            = DemoLoader (img_dir, base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        #data_reverse = DemoLoader (img_dir, base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        img             = data.img_preprocess()
        #img_reverse = data_reverse.img_preprocess()

        # Choose and load model (this paper is finished by one GPU)
        if args.model   == 'DNANet':
            model       = DNANet(num_classes=1,input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=args.deep_supervision)
        # elif args.model == 'ACM':
        #     model       = ACM   (args.in_channels, layers=[args.blocks] * 3, fuse_mode=args.fuse_mode, tiny=False, classes=1)
        model           = model.cuda()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model

        # Load Checkpoint
        #checkpoint      = torch.load('pretrain_DNANet_model.tar')
        checkpoint      = torch.load('mIoU__DNANet_NUDT-SIRST_epoch.pth.tar')
        self.model.load_state_dict(checkpoint['state_dict'])

        # Test
        self.model.eval()
        img = img.cuda()
        img = torch.unsqueeze(img,0)

        if args.deep_supervision == 'True':
            preds = self.model(img)
            pred  = preds[-1]
            pred = pred.to('cpu')
            pred_np = pred.squeeze().detach().numpy().astype(np.uint8)
            pred_npix = pred.squeeze().detach().numpy()
            height, width = pred_np.shape[-2],pred_np.shape[-1]
            ###制定区域得到像素值
            start_h, start_w = 11, 145
            end_h, end_w = 13, 148

            # 获取指定区域内的坐标和像素值
            coordinates = []
            pixel_values = []
            for h in range(start_h, end_h + 1):
                for w in range(start_w, end_w + 1):
                    pixel_value = pred_npix[h, w]
                    coordinates.append((h, w))
                    pixel_values.append(pixel_value)

            # 将坐标和像素值保存到文本文件
            with open("pixel_coordinates_values.txt", "w") as f:
                for coord, pixel_value in zip(coordinates, pixel_values):
                    f.write(f"Coordinate: {coord}, Pixel value: {pixel_value}\n")
#             ###下面的代码是得到像素区域的图像
#             region_pixels = pred_npix[start_h:end_h+1, start_w:end_w+1]
#             fig = plt.figure()
#             plt.imshow(region_pixels, cmap='gray')
#             plt.axis('off')
# ##下面显示像素点的像素值
#             for i in range(region_pixels.shape[0]):
#                 for j in range(region_pixels.shape[1]):
#                     plt.text(j,i,str(region_pixels[i,j]),color='red', fontsize=8,ha='center',va='center',fontfamily='Arial')
# ###上面显示像素点的像素值
#             plt.savefig('outimage.png', bbox_inches='tight',pad_inches=0)

            ###上面的代码是得到像素区域的图像

            ##3d
            region_pixels = pred_npix[start_h:end_h + 1, start_w:end_w + 1]
            x = np.arange(start_w, end_w + 1)
            y = np.arange(start_h, end_h + 1)
            x, y = np.meshgrid(x, y)

            # 将像素值展平为一维数组作为 z 坐标
            z = region_pixels.flatten()
            x = x.reshape(region_pixels.shape)
            y = y.reshape(region_pixels.shape)
            z = z.reshape(region_pixels.shape)
            # 创建三维图形对象
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # 绘制三维图形
            ax.plot_surface(x, y, z, cmap='viridis')

            # 设置坐标轴标签
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Pixel Value')

            # 显示图形
            plt.show()

            #上面3d
            ###
            #############底下这个代码用来确定最小坐标值和最大坐标值################
            '''coordinates = []
            pixel_values = []
            for h in range(height):
                for w in range(width):
                    pixel_value = pred_npix[h, w]
                    # pixel_value = tensor_pixel.item()
                    if pixel_value > 0:
                        coordinates.append((h, w))
                        pixel_values.append(pixel_value)
            with open("pixel_values.txt", "w") as f:
                for coord, value in zip(coordinates, pixel_values):
                    f.write(f"Coordinate: {coord}, Pixel value: {value}\n")'''
            #############上面这个代码用来确定最小坐标值和最大坐标值################
                # for h in range(height):
                #     for w in range(width):
                #         pixel_value = pred_npix[h,w]
                #         #pixel_value = tensor_pixel.item()
                #         if pixel_value > 0:
                #             coordinates.appens((h,w))
                            #f.write(f"Pixel value: {pixel_value},Coordinate: ({h}, {w})\n")
                           # f.write(f"{pixel_value}\n")
            image = Image.fromarray(pred_np)
            image.save('tensor.png')
        else:
            pred  = self.model(img)
        save_Pred_GT_visulize(pred, args.img_demo_dir, args.img_demo_index, args.suffix)






def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)





