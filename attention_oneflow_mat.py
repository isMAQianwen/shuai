# Basic module
import matplotlib.pyplot as plt
# Torch and visulization
from torchvision      import transforms
import scipy.io as sio
# Metric, loss .etc
from model.utils_oneflow_demo import *
from model.loss import *
from model.load_param_data import load_param
import cv2

# Model
# from model.model_DNANet_improve3_chutuyong import  Res_CBAM_block
# #from model.model_ACM    import  ACM
# from model.model_DNANet_improve3_chutuyong import  DNANet
# from model.UIU import  Res_CBAM_block
# #from model.model_ACM    import  ACM
from model.model_DNANet_oneflow import  Res_CBAM_block
from model.model_DNANet_oneflow import  DNANet
#from model.uiunet import UIUNET

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
    parser.add_argument('--demo_output_dir', type=str, default='demo_output',
                        help='img_demo')
    parser.add_argument('--img_demo_dir_reverse', type=str, default='img_demo_reverse',
                        help='img_demo')
    parser.add_argument('--img_demo_index', type=str,default='000720',
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
        yangben = [img_dir]

        # Preprocess and load data
        input_transform = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        data            = DemoLoader (yangben, base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        img_list             = data.img_preprocess()
        img = img_list[0]


        # Choose and load model (this paper is finished by one GPU)
        if args.model   == 'DNANet':
            model       = DNANet(num_classes=1,input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=args.deep_supervision)
            #model = UIUNET(3, 1)
        # elif args.model == 'ACM':
        #     model       = ACM   (args.in_channels, layers=[args.blocks] * 3, fuse_mode=args.fuse_mode, tiny=False, classes=1)
        model           = model.cuda()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model

        # Load Checkpoint
        #checkpoint      = torch.load('pretrain_DNANet_model.tar')
        device = torch.device('cuda:0')  # 指定你要加载模型的设备
        checkpoint      = torch.load(r'.\NUDT-DNA-weights\NUDT55_DNA_oneflow.tar',map_location=device)
        self.model.load_state_dict(checkpoint['state_dict'])

        # Test
        self.model.eval()
        img = img.cuda()
        img = torch.unsqueeze(img,0)

        if args.deep_supervision == 'True':
            preds = self.model(img)
            pred  = preds[-1]
            pred = pred.to('cpu')
            pred_np = pred.squeeze().detach().numpy()#.astype(np.uint8)

            filename = 'data.mat'
            varname = 'data'

            # 使用sio.savemat保存ndarray为.mat文件
            sio.savemat(filename, {varname: pred_np})
            pred_np_normalized = (pred_np - np.min(pred_np)) / (np.max(pred_np) - np.min(pred_np)) * 255

            # 转换为uint8类型的灰度图像
            pred_np_uint8 = pred_np_normalized.astype(np.uint8)

            # 应用Jet颜色映射
            pred_np_jet = cv2.applyColorMap(pred_np_uint8, cv2.COLORMAP_JET)

            # 保存为Jet格式的图像
            cv2.imwrite('pred_jet.png', pred_np_jet)

            # plt.axis('off')
            # plt.imshow(pred_np, cmap = 'jet')
            # # # #plt.tight_layout
            # # # #plt.show()
            # plt.savefig('632tensor.png',bbox_inches='tight')
            # image = Image.fromarray(pred_np)
            # image.save('tensor.png')
            # cmap = plt.get_cmap('jet')
            # colored_array = cmap(pred_np)
            # fig = plt.figure()
            # plt.axis('off')
            # plt.imshow(colored_array)
            # plt.savefig('tensor.png', dpi=300,bbox_inches='tight',pad_inches = 0)
        else:
            pred  = self.model(img)
        #save_Pred_GT_visulize(pred, args.demo_output_dir, args.img_demo_index, args.suffix)






def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)





