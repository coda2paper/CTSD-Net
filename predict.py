import torch.nn.functional as F
from RSCaMa.model.model_decoder import DecoderTransformer
from VideoX.SeqTrack.lib.models.seqtrack.vit import vit_base_patch16
from RSCaMa.utils_tool.utils import *
from model.model_encoder import Encoder as Resnet101
from RSCaMa.train_CC import CNN_ViT
import torch.optim
from torch.utils import data
import argparse
from PIL import Image
import cv2
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from data.LEVIR_CC.LEVIRCC import LEVIRCCDataset
# from model.model_encoder_attMamba import Encoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer
from utils_tool.utils import *
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchcam.utils import overlay_mask
from pytorch_grad_cam import GradCAM


    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        self.hook_handles = []

        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_data, target_time_step=None):
        # 前向传播
        output = self.model(input_data[0], input_data[1])

        if target_time_step is None:
            # 默认关注第一个生成词
            target_time_step = 0
        # 获取目标时间步的预测词索引
        target_probs = output[0, target_time_step]  # [hidden_dim]
        target_index = torch.argmax(target_probs).item()
        # 创建三维one-hot向量
        one_hot = torch.zeros_like(output)
        one_hot[0, target_time_step, target_index] = 1

        self.model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)

        # 计算权重
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        # 生成热力图
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(activations, dim=0).cpu().detach()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= torch.max(heatmap)

        return heatmap.numpy()

def get_last_conv(m):
    """
    Get the last conv layer in an Module.
    """
    convs = filter(lambda k: isinstance(k, torch.nn.Conv2d), m.modules())
    return list(convs)[-1]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def visualize_results(imgA, imgB, heatmap_A, heatmap_B, filename, caption):
    plt.figure(figsize=(24, 6))

    heatmap = cv2.resize(heatmap_A, (256, 256))
    heatmap = (heatmap * 255).astype("uint8")
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.imwrite(f'/workstation1/wrj/chg2cap_gradcam_pic/{filename}_A.png', heatmap_colored)

    heatmap = cv2.resize(heatmap_B, (256, 256))
    heatmap = (heatmap * 255).astype("uint8")
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.imwrite(f'/workstation1/wrj/chg2cap_gradcam_pic/{filename}_B.png', heatmap_colored)
    print()


def main(args):
    """
    Testing.
    """
    with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
        word_vocab = json.load(f)
    # Load checkpoint
    snapshot_full_path = args.checkpoint#os.path.join(args.savepath, args.checkpoint)
    checkpoint = torch.load(snapshot_full_path)

    args.result_path = os.path.join(args.result_path, os.path.basename(snapshot_full_path).replace('.pth', ''))
    if os.path.exists(args.result_path) == False:
        os.makedirs(args.result_path)
    else:
        print('result_path is existed!')

        for root, dirs, files in os.walk(args.result_path):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    encoder = vit_base_patch16(pretrained=True, pretrain_type='mae',
                                    search_size=256, template_size=256,
                                    search_number=1, template_number=1,
                                    drop_path_rate=0,
                                    use_checkpoint=False)
    # self.encoder = Encoder(args.network)
    # self.encoder.fine_tune(args.fine_tune_encoder)
    resnet101 = Resnet101('resnet101', fine_tune=False)
    cnn_vit = CNN_ViT()
    # encoder_trans = AttentiveEncoder(n_layers=args.n_layers,
    #                                       feature_size=[args.feat_size, args.feat_size, args.encoder_dim],
    #                                       heads=args.n_heads, dropout=args.dropout)
    decoder = DecoderTransformer(decoder_type=args.decoder_type,embed_dim=args.embed_dim,
                                      vocab_size=len(word_vocab), max_lengths=args.max_length,
                                      word_vocab=word_vocab, n_head=args.n_heads,
                                      n_layers=args.decoder_n_layers, dropout=args.dropout)


    encoder.load_state_dict(checkpoint['encoder_dict'], strict=True)
    resnet101.load_state_dict(checkpoint['resnet101_dict'], strict=True)
    cnn_vit.load_state_dict(checkpoint['cnn_vit_dict'], strict=True)
    # encoder_trans.load_state_dict(checkpoint['encoder_trans'])
    decoder.load_state_dict(checkpoint['decoder_dict'], strict=False)
    # Move to GPU, if available
    # cnn_vit.conv1.register_forward_hook(save_conv1_grad_hook)
    encoder.eval()
    encoder = encoder.cuda()
    resnet101.eval()
    resnet101 = resnet101.cuda()
    cnn_vit.eval()
    cnn_vit = cnn_vit.cuda()
    # encoder_trans.eval()
    # encoder_trans = encoder_trans.cuda()
    decoder.eval()
    decoder = decoder.cuda()

    target_layer = get_last_conv(cnn_vit)
    # target_layer = resnet101.layer4[-1]  
    gradcam = GradCAM(cnn_vit, target_layer)

    # target_layers = [layers]
    print('load model success!')

    # Custom dataloaders
    if args.data_name == 'LEVIR_CC':
        # LEVIR:
        nochange_list = ["the scene is the same as before ", "there is no difference ",
                         "the two scenes seem identical ", "no change has occurred ",
                         "almost nothing has changed "]
        test_loader = data.DataLoader(
                LEVIRCCDataset(args.network, args.data_folder, args.list_path, 'test', args.token_folder, word_vocab, args.max_length, args.allow_unk),
                batch_size=args.test_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)

    hypotheses = list()  # hypotheses (predictions)

    with torch.no_grad():
        # Batches
        for ind, batch_data in enumerate(
                tqdm(test_loader, desc='test_' + " EVALUATING AT BEAM SIZE " + str(1))):
            # Move to GPU, if available
            imgA = batch_data['imgA']
            imgB = batch_data['imgB']
            token_all = batch_data['token_all']
            token_all_len = batch_data['token_all_len']
            name = batch_data['name']
            imgA = imgA.cuda()
            imgB = imgB.cuda()
            token_all = token_all.squeeze(0).cuda()
            # Forward prop.
            if encoder is not None:
                feat = encoder((imgA, imgB))
                # feat1 = feat[:, :feat.shape[1] // 2, :]
                # feat2 = feat[:, feat.shape[1] // 2:, :]
                # feat = encoder_trans(feat1, feat2)
                imgA_list, imgB_list = resnet101(imgA, imgB)
                feat1 = feat[:, :feat.shape[1] // 2, :]
                feat2 = feat[:, feat.shape[1] // 2:, :]
                # feat1, feat2 = self.encoder(imgA, imgB)
                feat1 = cnn_vit(imgA_list, feat1)
                feat2 = cnn_vit(imgB_list, feat2)
                feat = torch.cat([feat1, feat2], 1)
            seq = decoder.sample(feat, k=1)

            # for captioning
            except_tokens = {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}
            img_token = token_all.tolist()
            # img_tokens = list(map(lambda c: [w for w in c if w not in except_tokens],
            #             img_token))  # remove <start> and pads
            # references.append(img_tokens)

            pred_seq = [w for w in seq if w not in except_tokens]
            hypotheses.append(pred_seq)

            pred_caption = ""
            for i in pred_seq:
                pred_caption += (list(word_vocab.keys())[i]) + " "
            name = name[0]
            visualize_results(
                imgA.cpu().numpy()[0],
                imgB.cpu().numpy()[0],
                heatmap_A,
                heatmap_B,
                name,
                pred_caption
            )
            # with open(os.path.join(args.result_path, name.split('.')[0] + f'_cap.txt'), 'w') as f:
            #     f.write('pred_caption: ' + pred_caption + '\n')
            # save_captions(pred_caption, ref_captions, hypotheses[-1], references[-1], name, args.result_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Captioning')

    # Data parameters
    parser.add_argument('--sys', default='linux', choices=('linux'), help='system')
    parser.add_argument('--data_folder', default='/workstation/Chg2Cap/data/LEVIR_CC/images',
                        help='folder with data files')
    parser.add_argument('--list_path', default='/workstation/Chg2Cap/data/LEVIR_CC/', help='path of the data lists')
    parser.add_argument('--token_folder', default='/workstation/Chg2Cap/data/LEVIR_CC/tokens/', help='folder with token files')
    parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    parser.add_argument('--max_length', type=int, default=42, help='path of the data lists')
    parser.add_argument('--allow_unk', type=int, default=1, help='if unknown token is allowed')
    parser.add_argument('--data_name', default="LEVIR_CC", help='base name shared by data files.')

    # Test
    parser.add_argument('--gpu_id', type=int, default=2, help='gpu id in the training.')
    # parser.add_argument('--checkpoint', default='/workstation1/wrj/Chg2cap/models_ckpt/transformer_decoder_2024-12-25-06-47-43/LEVIR_CC_bts_16_epo7_Bleu4_64249.pth', help='path to checkpoint, None if none.')
    parser.add_argument('--checkpoint', default='/workstation/wrj/Chg2Cap/models_ckpt/transformer_decoder_2024-11-27-16-15-57/LEVIR_CC_bts_16_epo12_Bleu4_64120.pth', help='path to checkpoint, None if none.')
    # Time: 223.178	BLEU-1: 0.84898	BLEU-2: 0.76240	BLEU-3: 0.68900	BLEU-4: 0.63000	Meteor: 0.39822	Rouge: 0.74650	Cider: 1.33622
    # parser.add_argument('--checkpoint', default='/workstation/wrj/Chg2Cap/models_ckpt/transformer_decoder_2024-11-27-11-39-11/LEVIR_CC_bts_16_epo4_Bleu4_64111.pth', help='path to checkpoint, None if none.')
    # Time: 224.417	BLEU-1: 0.84699	BLEU-2: 0.75972	BLEU-3: 0.68747	BLEU-4: 0.63012	Meteor: 0.39426	Rouge: 0.74023	Cider: 1.33030
    # parser.add_argument('--checkpoint', default='/workstation/wrj/Chg2Cap/models_ckpt/transformer_decoder_2024-11-25-05-44-41/LEVIR_CC_bts_16_epo3_Bleu4_64605.pth', help='path to checkpoint, None if none.')
    # Time: 229.610	BLEU-1: 0.84554	BLEU-2: 0.76114	BLEU-3: 0.68842	BLEU-4: 0.62846	Meteor: 0.39472	Rouge: 0.74002	Cider: 1.32016()
    parser.add_argument('--print_freq', type=int, default=100, help='print training/validation stats every __ batches')
    parser.add_argument('--test_batchsize', default=1, help='batch_size for validation')
    parser.add_argument('--workers', type=int, default=8,
                        help='for data-loading; right now, only 0 works with h5pys in windows.')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    # Validation
    parser.add_argument('--result_path', default="./predict_result/")

    # backbone parameters
    parser.add_argument('--decoder_type', default='transformer_decoder', help='mamba or gpt or transformer_decoder')
    parser.add_argument('--network', default='CLIP-ViT-B/32',help='define the backbone encoder to extract features')
    parser.add_argument('--encoder_dim', type=int, default=768, help='the dim of extracted features of backbone ')
    parser.add_argument('--feat_size', type=int, default=16, help='size of extracted features of backbone')
    # Model parameters
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers in AttentionEncoder.')
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=768, help='embedding dimension')
    args = parser.parse_args()
    if args.sys == 'linux':
        args.data_folder = '/workstation/Chg2Cap/data/LEVIR_CC/images'
        if os.path.exists(args.data_folder) == False:
            args.data_folder = '/workstation/Chg2Cap/data/LEVIR_CC/images'
            if os.path.exists(args.data_folder) == False:
                args.data_folder = '/workstation/Chg2Cap/data/LEVIR_CC/images'  # '/mnt/levir_datasets/LCY/Dataset/Levir-CC-dataset/images'
    print('list_path:', args.list_path)
    if args.network == 'CLIP-RN50':
        clip_emb_dim = 1024
        args.encoder_dim, args.feat_size = 2048, 7
    elif args.network == 'CLIP-RN101':
        clip_emb_dim = 512
        args.encoder_dim, args.feat_size = 2048, 7
    elif args.network == 'CLIP-RN50x4':
        clip_emb_dim = 640
        args.encoder_dim, args.feat_size = 2560, 9
    elif args.network == 'CLIP-RN50x16':
        clip_emb_dim = 768
        args.encoder_dim, args.feat_size = 3072, 12
    elif args.network == 'CLIP-ViT-B/16' or args.network == 'CLIP-ViT-L/16':
        clip_emb_dim = 512
        args.encoder_dim, args.feat_size = 768, 14
    elif args.network == 'CLIP-ViT-B/32' or args.network == 'CLIP-ViT-L/32':
        clip_emb_dim = 512
        args.encoder_dim, args.feat_size = 768, 16
    elif args.network == 'segformer-mit_b1':
        args.encoder_dim, args.feat_size = 512, 8

    args.embed_dim = args.encoder_dim
    main(args)

# 我的數據集
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Captioning')
#
#     # Data parameters
#     parser.add_argument('--sys', default='linux', choices=('linux'), help='system')
#     parser.add_argument('--data_folder', default='/workstation1/wrj/Chg2cap',
#                         help='folder with data files')
#     parser.add_argument('--list_path', default='/workstation1/wrj/Chg2cap/', help='path of the data lists')
#     parser.add_argument('--token_folder', default=None, help='folder with token files')
#     parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
#     parser.add_argument('--max_length', type=int, default=42, help='path of the data lists')
#     parser.add_argument('--allow_unk', type=int, default=1, help='if unknown token is allowed')
#     parser.add_argument('--data_name', default="LEVIR_CC", help='base name shared by data files.')
#
#     # Test
#     parser.add_argument('--gpu_id', type=int, default=2, help='gpu id in the training.')
#     # parser.add_argument('--checkpoint', default='/workstation1/wrj/Chg2cap/models_ckpt/transformer_decoder_2024-12-25-06-47-43/LEVIR_CC_bts_16_epo7_Bleu4_64249.pth', help='path to checkpoint, None if none.')
#     parser.add_argument('--checkpoint', default='/workstation/wrj/Chg2Cap/models_ckpt/transformer_decoder_2024-11-27-16-15-57/LEVIR_CC_bts_16_epo12_Bleu4_64120.pth', help='path to checkpoint, None if none.')
#     # Time: 223.178	BLEU-1: 0.84898	BLEU-2: 0.76240	BLEU-3: 0.68900	BLEU-4: 0.63000	Meteor: 0.39822	Rouge: 0.74650	Cider: 1.33622
#     # parser.add_argument('--checkpoint', default='/workstation/wrj/Chg2Cap/models_ckpt/transformer_decoder_2024-11-27-11-39-11/LEVIR_CC_bts_16_epo4_Bleu4_64111.pth', help='path to checkpoint, None if none.')
#     # Time: 224.417	BLEU-1: 0.84699	BLEU-2: 0.75972	BLEU-3: 0.68747	BLEU-4: 0.63012	Meteor: 0.39426	Rouge: 0.74023	Cider: 1.33030
#     # parser.add_argument('--checkpoint', default='/workstation/wrj/Chg2Cap/models_ckpt/transformer_decoder_2024-11-25-05-44-41/LEVIR_CC_bts_16_epo3_Bleu4_64605.pth', help='path to checkpoint, None if none.')
#     # Time: 229.610	BLEU-1: 0.84554	BLEU-2: 0.76114	BLEU-3: 0.68842	BLEU-4: 0.62846	Meteor: 0.39472	Rouge: 0.74002	Cider: 1.32016()
#     parser.add_argument('--print_freq', type=int, default=100, help='print training/validation stats every __ batches')
#     parser.add_argument('--test_batchsize', default=1, help='batch_size for validation')
#     parser.add_argument('--workers', type=int, default=8,
#                         help='for data-loading; right now, only 0 works with h5pys in windows.')
#     parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
#     # Validation
#     parser.add_argument('--result_path', default="/workstation1/wrj/Chg2cap/predict_result/")
#
#     # backbone parameters
#     parser.add_argument('--decoder_type', default='transformer_decoder', help='mamba or gpt or transformer_decoder')
#     parser.add_argument('--network', default='CLIP-ViT-B/32',help='define the backbone encoder to extract features')
#     parser.add_argument('--encoder_dim', type=int, default=768, help='the dim of extracted features of backbone ')
#     parser.add_argument('--feat_size', type=int, default=16, help='size of extracted features of backbone')
#     # Model parameters
#     parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
#     parser.add_argument('--n_layers', type=int, default=3, help='Number of layers in AttentionEncoder.')
#     parser.add_argument('--decoder_n_layers', type=int, default=1)
#     parser.add_argument('--embed_dim', type=int, default=768, help='embedding dimension')
#     args = parser.parse_args()
#     if args.sys == 'linux':
#         args.data_folder = '/workstation1/wrj/Chg2cap'
#         if os.path.exists(args.data_folder) == False:
#             args.data_folder = '/workstation/Chg2Cap/data/LEVIR_CC/images'
#             if os.path.exists(args.data_folder) == False:
#                 args.data_folder = '/workstation/Chg2Cap/data/LEVIR_CC/images'  # '/mnt/levir_datasets/LCY/Dataset/Levir-CC-dataset/images'
#     print('list_path:', args.list_path)
#     if args.network == 'CLIP-RN50':
#         clip_emb_dim = 1024
#         args.encoder_dim, args.feat_size = 2048, 7
#     elif args.network == 'CLIP-RN101':
#         clip_emb_dim = 512
#         args.encoder_dim, args.feat_size = 2048, 7
#     elif args.network == 'CLIP-RN50x4':
#         clip_emb_dim = 640
#         args.encoder_dim, args.feat_size = 2560, 9
#     elif args.network == 'CLIP-RN50x16':
#         clip_emb_dim = 768
#         args.encoder_dim, args.feat_size = 3072, 12
#     elif args.network == 'CLIP-ViT-B/16' or args.network == 'CLIP-ViT-L/16':
#         clip_emb_dim = 512
#         args.encoder_dim, args.feat_size = 768, 14
#     elif args.network == 'CLIP-ViT-B/32' or args.network == 'CLIP-ViT-L/32':
#         clip_emb_dim = 512
#         args.encoder_dim, args.feat_size = 768, 16
#     elif args.network == 'segformer-mit_b1':
#         args.encoder_dim, args.feat_size = 512, 8
#
#     args.embed_dim = args.encoder_dim
#     main(args)
