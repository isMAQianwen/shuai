# Basic module
from tqdm             import tqdm
from model.parse_args_test import  parse_args

# Torch and visulization
from torchvision      import transforms
from torch.utils.data import DataLoader


# Metric, loss .etc
from model.utils_two import *
from model.loss import *
from model.load_param_data import  load_dataset, load_param

# Model
from model.model_DNANet_twoflow_add import  Res_CBAM_block
from model.model_DNANet_twoflow_add import  DNANet
#from model.uiunet_twoadd import UIUNET

class Trainer(object):
    def __init__(self, args):

        # Initial
        self.args  = args
        self.save_prefix = '_'.join([args.model, args.dataset])
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

        # Read image index from TXT
        if args.mode    == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, val_img_ids, test_txt=load_dataset(args.root, args.dataset,args.split_method)

        # Preprocess and load data
        input_transform = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        testset         = TestSetLoader (dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False)

        # Choose and load model (this paper is finished by one GPU)
        if args.model   == 'DNANet':
            #model = UIUNET(3, 1)
            model       = DNANet(num_classes=1,input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=args.deep_supervision)
        model           = model.cuda()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model

        # Checkpoint
        device = torch.device('cuda:0')
        checkpoint             = torch.load('result/' + args.model_dir, map_location=device)
        visulization_path      = dataset_dir + '/' +'visulization_result_2024' + '/' + args.st_model + '_visulization_path'
        visulization_fuse_path = dataset_dir + '/' +'visulization_result_2024' + '/' + args.st_model + '_visulization_fuse'
        # # visulization_raw = dataset_dir + '/' +'visulization_result' + '/' + args.st_model + '_visulization_raw'
        # # visulization_reverse = dataset_dir + '/' + 'visulization_result' + '/' + args.st_model + '_visulization_reverse'
        # visulization_path      = dataset_dir + '/' +'NUDT_DNA_improved3' + '/' + args.st_model + '_visulization_path'
        # visulization_fuse_path = dataset_dir + '/' +'NUDT_DNA_improved3' + '/' + args.st_model + '_visulization_fuse'
        # visulization_raw = dataset_dir + '/' +'NUDT_DNA_improved3' + '/' + args.st_model + '_visulization_raw'
        # visulization_reverse = dataset_dir + '/' + 'NUDT_DNA_improved3' + '/' + args.st_model + '_visulization_reverse'

        make_visulization_dir(visulization_path, visulization_fuse_path)
        #make_visulization_dir(visulization_raw, visulization_reverse)

        # Load trained model
        self.model.load_state_dict(checkpoint['state_dict'])

        # Test
        self.model.eval()
        tbar = tqdm(self.test_data)
        with torch.no_grad():
            num = 0
            for i, ( data,datareverse, labels) in enumerate(tbar):
                data = data.cuda()
                datareverse = datareverse.cuda()
                labels = labels.cuda()
                if args.deep_supervision == 'True':
                    preds = self.model(data, datareverse)
                    pred =preds[-1]
                    # predreverse = preds[-1]
                    # predraw = preds[-2]

                else:
                    pred = self.model(data)
                save_Pred_GT(pred, labels,visulization_path, val_img_ids, num, args.suffix)
                # save_Pred_GT(predreverse, labels, visulization_reverse, val_img_ids, num, args.suffix)
                # save_Pred_GT(predraw, labels, visulization_raw, val_img_ids, num, args.suffix)
                num += 1

            #total_visulization_generation(dataset_dir, args.mode, test_txt, args.suffix, visulization_path, visulization_fuse_path)





def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)





