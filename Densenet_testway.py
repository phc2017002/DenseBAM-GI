import torch
import math
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy
import torch.utils.data as data
from data_iterator import dataIterator
from Attention_RNN_test_momentum_original_1 import AttnDecoderRNN
from Densenet_torchvision import densenet121
from PIL import Image
from numpy import *
import numpy as np
#from condensenet_v2 import CondenseNetV2
from Densenet_BAM_1 import DenseNet
import matplotlib.pyplot as plt
from skimage.transform import resize
import os
import cv2
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


torch.backends.cudnn.benchmark = False

def cmp_result(label,rec):
    dist_mat = numpy.zeros((len(label)+1, len(rec)+1),dtype='int32')
    dist_mat[0,:] = range(len(rec) + 1)
    dist_mat[:,0] = range(len(label) + 1)
    for i in range(1, len(label) + 1):
        for j in range(1, len(rec) + 1):
            hit_score = dist_mat[i-1, j-1] + (label[i-1] != rec[j-1])
            ins_score = dist_mat[i,j-1] + 1
            del_score = dist_mat[i-1, j] + 1
            dist_mat[i,j] = min(hit_score, ins_score, del_score)

    dist = dist_mat[len(label), len(rec)]
    return dist, len(label),hit_score,ins_score,del_score


def load_dict(dictFile):
    fp=open(dictFile)
    stuff=fp.readlines()
    fp.close()
    lexicon={}
    for l in stuff:
        w=l.strip().split()
        lexicon[w[0]]=int(w[1])

    print('total words/phones',len(lexicon))
    return lexicon

valid_datasets=['test_prof_lehal_binary.pkl', './test_prof_lehal.txt']
dictionaries=['./dictionary.txt']
batch_Imagesize=16
valid_batch_Imagesize=16
batch_size_t=1
maxlen=48
maxImagesize=300000
hidden_size = 256
gpu = [0]

#error = open('./model_test_Densenet_bam_attention_naive_gru_sequence_length_test_trail_2.txt',"w+")
error_0 = open('./model_attention_test_DenseBAM_GI_test_2.txt',"w+")

worddicts = load_dict(dictionaries[0])
worddicts_r = [None] * len(worddicts)
for kk, vv in worddicts.items():
        worddicts_r[vv] = kk

test,test_label = dataIterator(valid_datasets[0],valid_datasets[1],worddicts,batch_size=1,batch_Imagesize=batch_Imagesize,maxlen=maxlen,maxImagesize=maxImagesize)

class custom_dset(data.Dataset):
    def __init__(self,train,train_label):
        self.train = train
        self.train_label = train_label

    def __getitem__(self, index):
        train_setting = torch.from_numpy(numpy.array(self.train[index]))
        label_setting = torch.from_numpy(numpy.array(self.train_label[index])).type(torch.LongTensor)

        size = train_setting.size()
        train_setting = train_setting.view(1,size[2],size[3])
        label_setting = label_setting.view(-1)

        return train_setting,label_setting

    def __len__(self):
        return len(self.train)

off_image_test = custom_dset(test,test_label)
#print(off_image_train[10])


def imresize(im,sz):
    pil_im = Image.fromarray(im)
    return numpy.array(pil_im.resize(sz))


def collate_fn(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    img, label = zip(*batch)
    aa1 = 0
    bb1 = 0
    k = 0
    k1 = 0
    max_len = len(label[0])+1
    for j in range(len(img)):
        size = img[j].size()
        if size[1] > aa1:
            aa1 = size[1]
        if size[2] > bb1:
            bb1 = size[2]

    for ii in img:
        ii = ii.float()
        img_size_h = ii.size()[1]
        img_size_w = ii.size()[2]
        img_mask_sub_s = torch.ones(1,img_size_h,img_size_w).type(torch.FloatTensor)
        img_mask_sub_s = img_mask_sub_s*255.0
        img_mask_sub = torch.cat((ii,img_mask_sub_s),dim=0)
        padding_h = aa1-img_size_h
        padding_w = bb1-img_size_w
        m = torch.nn.ZeroPad2d((0,padding_w,0,padding_h))
        img_mask_sub_padding = m(img_mask_sub)
        img_mask_sub_padding = img_mask_sub_padding.unsqueeze(0)
        if k==0:
            img_padding_mask = img_mask_sub_padding
        else:
            img_padding_mask = torch.cat((img_padding_mask,img_mask_sub_padding),dim=0)
        k = k+1

    for ii1 in label:
        ii1 = ii1.long()
        ii1 = ii1.unsqueeze(0)
        ii1_len = ii1.size()[1]
        m = torch.nn.ZeroPad2d((0,max_len-ii1_len,0,0))
        ii1_padding = m(ii1)
        if k1 == 0:
            label_padding = ii1_padding
        else:
            label_padding = torch.cat((label_padding,ii1_padding),dim=0)
        k1 = k1+1

    img_padding_mask = img_padding_mask/255.0
    return img_padding_mask, label_padding

test_loader = torch.utils.data.DataLoader(
    dataset = off_image_test,
    batch_size = batch_size_t,
    shuffle = True,
    collate_fn = collate_fn
)

#error = open('with_dropout_and_l2_GRU_test_sin.txt','w+')


device = ('cuda:0')

growth_rate = 32
block_config = [6, 12, 24]
num_init_features = 64
bn_size = 6
drop_rate = 0
num_classes = 1000


encoder = DenseNet(growth_rate=32,num_layers=block_config)
#encoder = densenet121()
attn_decoder1 = AttnDecoderRNN(hidden_size,112,dropout_p=0.5)

encoder = torch.nn.DataParallel(encoder, device_ids=gpu)
attn_decoder1 = torch.nn.DataParallel(attn_decoder1, device_ids=gpu)
encoder = encoder.cuda().to(device)
attn_decoder1 = attn_decoder1.cuda().to(device)

print('--------------------herererer')

encoder.load_state_dict(torch.load('./model_test_densenet_1024_layers_bam_with_GRU_2014_0.00_0.05_batch_size_6_trail_2/encoder_lr0.00000_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl'))
attn_decoder1.load_state_dict(torch.load('./model_test_densenet_1024_layers_bam_with_naive_GRU_2014_0.00_0.05_batch_size_6_trail_2/attn_decoder_lr0.00000_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl'))

total_dist = 0
total_label = 0
total_line = 0
total_line_rec = 0
hit_all =0
ins_all =0
dls_all =0
wer_1 = 0
wer_2 = 0
wer_3 = 0
wer_4 = 0
wer_5 = 0
wer_6 = 0
wer_up=0
total_dist_1 = 0
total_label_1 = 0
total_line_1 = 0
total_line_rec_1 = 0
total_line_rec_2 = 0
total_line_rec_3 = 0

c_1_5 = 0
c_6_10 = 0
c_11_15 = 0
c_16_20 = 0
c_21_25 = 0
c_26_30 = 0
c_31_35 = 0
c_36_40 = 0
c_41_45 = 0
c_45_plus = 0

e_1_5 = 0
e_6_10 = 0
e_11_15 = 0
e_16_20 = 0
e_21_25 = 0
e_26_30 = 0
e_31_35 = 0
e_36_40 = 0
e_41_45 = 0
e_45_plus = 0

encoder.eval()
attn_decoder1.eval()

epoch = 5

loss_func = nn.CrossEntropyLoss()

'''
def visualize_attention(images, attention_weights, save_path='./model_visualize_attention'):
    batch_size = images.shape[0]

    for idx in range(batch_size):
        image = images[idx]

        # If the input image has an extra channel dimension, use only the first channel for visualization
        if image.ndim == 3 and image.shape[0] == 2:
            image = image[0]

        # Get the attention weights for the current image in the batch
        attention_weight = attention_weights[idx]

        # Normalize the attention weights
        attention_weight = (attention_weight - np.min(attention_weight)) / (np.max(attention_weight) - np.min(attention_weight))

        # Resize the attention weights to match the image size
        attention_weight_resized = np.resize(attention_weight, (image.shape[0], image.shape[1]))

        # Create a heatmap
        plt.imshow(image, cmap='gray')
        plt.imshow(attention_weight_resized, cmap='jet', alpha=0.5)
        plt.colorbar()

        if save_path:
            plt.savefig(f"{save_path}_{idx}")
        else:
            plt.show()

        # Clear the current figure to avoid overlapping plots
        plt.clf()
'''

def visualize_attention(image, attention_weights, output_path):
    
    attention_weights = attention_weights.detach().cpu().numpy()
    attention_map_sample = np.squeeze(attention_weights)

    attention_map_sample = (attention_map_sample - attention_map_sample.min()) / (attention_map_sample.max() - attention_map_sample.min())

    print("Attention map sample shape:", attention_map_sample.shape)

    attention_map_resized = resize(attention_map_sample, image.shape, order=1, preserve_range=True)

    print("Attention map resized shape:", attention_map_resized.shape)

    # Combine the image and attention map
    combined_image = image * attention_map_resized.squeeze()

    plt.imsave(output_path + '_original.png', image, cmap='gray')

    # Save the combined image
    plt.imsave(output_path + '_combined.png', combined_image, cmap='gray')

    
    # Plot the image
    plt.subplot(1, 3, 3)
    plt.imshow(combined_image, cmap='gray')
    plt.axis('off')
    plt.title('Combined Image')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.clf()
    

'''
def visualize_attention(image, attention_weights, num_heads=6, save_path=None):
    # Choose the first layer for visualization
    layer_index = 0
    layer_attention = attention_weights[layer_index]

    # Prepare the figure with a 2x3 grid for 6 attention maps
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for head_index in range(num_heads):
        # Get the attention map of the head at head_index
        attention_map = layer_attention[head_index]

        # Normalize the attention map
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

        # Overlay the attention map on the original image
        attention_overlay = np.where(attention_map > 0.5, 1, 0.5 * attention_map + 0.5)
        overlayed_image = image * attention_overlay

        # Plot the overlayed image
        row = head_index // 3
        col = head_index % 3
        axes[row, col].imshow(overlayed_image, cmap='gray')
        axes[row, col].axis('off')
        axes[row, col].set_title(f'Head {head_index + 1}')

        # Save the overlayed image if save_path is provided
        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.imsave(os.path.join(save_path, f'image_{i}_head_{head_index + 1}.png'), overlayed_image, cmap='gray')
'''


'''
with torch.set_grad_enabled(True):
    def generate_saliency_map(input_image,decoder_output,y_t,output_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # assuming you're using cuda
        #input_image = torch.float().to(device)
        print('input_image-----', input_image.shape)
        #input_image.requires_grad = True  # Ensure we calculate gradients for input
        # ... get decoder_output_t ...
        loss_t = loss_func(decoder_output, y_t)  # Assume batch size of 1
        encoder.zero_grad()  # Clear previous gradients
        loss_t.backward(retain_graph=True)  # Backward pass to calculate gradients
        saliency_map = input_image.grad.data  # Derivative of loss wrt input

        if not isinstance(saliency_map, np.ndarray):
            saliency_map = saliency_map.detach().cpu().numpy()

        # Use absolute value for saliency_map and normalize it to the range 0-1
        saliency_map = np.abs(saliency_map)
        saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map))

        # Save the image (with colormap applied)
        #plt.imsave("saliency_map.png", saliency_map, cmap=plt.cm.hot)
        input_image = input_image - torch.min(input_image).to(input_image.device)  # Ensure the image is normalized
        input_image = input_image / torch.max(input_image)
        heatmap = cv2.applyColorMap(np.uint8(255 * saliency_map[0,0]), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255

        print("Shape of heatmap:", heatmap.shape)
        print("Shape of input_image:", input_image.cpu().detach().shape)

        input_image_expanded = np.expand_dims(input_image[0, 0, :, :].cpu().detach(), axis=-1)
        cam = heatmap + np.float32(input_image_expanded)

        cam = cam / np.max(cam)


        plt.imsave(output_path + '_combined.png', np.uint8(255 * cam))
'''

for step_t, (x_t, y_t) in enumerate(test_loader):
    x_real_high = x_t.size()[2]
    x_real_width = x_t.size()[3]
    if x_t.size()[0]<batch_size_t:
        break
    
    print('x_t-------------',x_t.shape)
    print('step_t',step_t)
    
    if x_real_high == 150:
        h_mask_t = []
        w_mask_t = []
        for i in x_t:
        #h*w
            size_mask_t = i[1].size()
            s_w_t = str(i[1][0])
            s_h_t = str(i[1][:,1])
            w_t = s_w_t.count('1')
            h_t = s_h_t.count('1')
            h_comp_t = int(h_t/16)+1
            w_comp_t = int(w_t/16)+1
            h_mask_t.append(h_comp_t)
            w_mask_t.append(w_comp_t)  
        x_t = x_t.cuda().to(device)
        y_t = y_t.cuda().to(device)
        x_t.requires_grad = True
        output_highfeature_t = encoder(x_t,x_t)

        x_mean_t = torch.mean(output_highfeature_t)
        x_mean_t = float(x_mean_t)
        output_area_t1 = output_highfeature_t.size()
        output_area_t = output_area_t1[3]
        dense_input = output_area_t1[2]

        decoder_input_t = torch.LongTensor([111]*batch_size_t)
        decoder_input_t = decoder_input_t.cuda().to(device)

        v_test = torch.randn(batch_size_t, 1, 3*hidden_size)
        nn.init.xavier_uniform_(v_test)
    
        attention_weights_list = []

        decoder_hidden_t = torch.randn(batch_size_t, 1, hidden_size).cuda().to(device)
        #v_test = torch.randn(batch_size_t, 1, 3*hidden_size)
        #nn.init.xavier_uniform_(decoder_hidden_t)
        #nn.init.xavier_uniform_(v_test)
    
        decoder_hidden_t = decoder_hidden_t * x_mean_t
        decoder_hidden_t = torch.tanh(decoder_hidden_t)
    

        prediction = torch.zeros(batch_size_t,maxlen)
        #label = torch.zeros(batch_size_t,maxlen)
        prediction_sub = []
        label_sub = []
        label_real = []
        prediction_real = []

        decoder_attention_t = torch.zeros(batch_size_t,1,dense_input,output_area_t).cuda().to(device)
        attention_sum_t = torch.zeros(batch_size_t,1,dense_input,output_area_t).cuda().to(device)

        m = torch.nn.ZeroPad2d((0,maxlen-y_t.size()[1],0,0))
        y_t = m(y_t)
        for i in range(maxlen):
            decoder_output, decoder_hidden_t, v_test, decoder_attention_t, attention_sum_t = attn_decoder1(decoder_input_t,
                                                                                         decoder_hidden_t,
                                                                                         v_test,
                                                                                         output_highfeature_t,
                                                                                         output_area_t,
                                                                                         attention_sum_t,
                                                                                         decoder_attention_t,dense_input,batch_size_t,h_mask_t,w_mask_t,gpu,epoch)      

            image = x_t[0][0].cpu().detach().numpy()
            #print('image------',image.shape)
            attention_weights = decoder_attention_t  # Extract attention weights for the current example
            #print("Attention weights shape:", attention_weights.shape)
            output_path = f'./attention_map_1_DenseBAM-GI/test_attention_map_only_trail_6_{i}.png'

            visualize_attention(image, attention_weights, output_path)

            #generate_saliency_map(image,decoder_output,y_t,output_path)

            #y_t = y_t.unsqueeze(0)
            #print('Shape of y_t:', y_t.shape)
            flag_z = [0]*batch_size_t
            
            for k in range(batch_size_t):

                #print('decoder_output-----', decoder_output[k].shape)
                decoder_output_p = decoder_output[k].squeeze(0)
                #y_t[k] = y_t[k].unsqueeze(0)
                #print('y_t[k]-----', y_t[k].shape)
                #generate_saliency_map(x_t,decoder_output_p, y_t[k,i],output_path)

            topv,topi = torch.max(decoder_output,2)
            if torch.sum(topi)==0:
                break
            decoder_input_t = topi
            decoder_input_t = decoder_input_t.view(batch_size_t)
                #print(topi.size()) 16,1

                # prediction
            prediction[:,i] = decoder_input_t

        for i in range(batch_size_t):
            for j in range(maxlen):
                if int(prediction[i][j]) ==0:
                    break
                else:
                    prediction_sub.append(int(prediction[i][j]))
                    prediction_real.append(worddicts_r[int(prediction[i][j])])
                
                error_0.write('\n prediction is :' + str(j) + str(prediction_real))
            if len(prediction_sub)<maxlen:
                prediction_sub.append(0)
            



'''
    h_mask_t = []
    w_mask_t = []
    for i in x_t:
        #h*w
        size_mask_t = i[1].size()
        s_w_t = str(i[1][0])
        s_h_t = str(i[1][:,1])
        w_t = s_w_t.count('1')
        h_t = s_h_t.count('1')
        h_comp_t = int(h_t/16)+1
        w_comp_t = int(w_t/16)+1
        h_mask_t.append(h_comp_t)
        w_mask_t.append(w_comp_t)

    x_t = x_t.cuda().to(device)
    y_t = y_t.cuda().to(device)
    output_highfeature_t = encoder(x_t,x_t)

    x_mean_t = torch.mean(output_highfeature_t)
    x_mean_t = float(x_mean_t)
    output_area_t1 = output_highfeature_t.size()
    output_area_t = output_area_t1[3]
    dense_input = output_area_t1[2]

    decoder_input_t = torch.LongTensor([111]*batch_size_t)
    decoder_input_t = decoder_input_t.cuda().to(device)

    v_test = torch.randn(batch_size_t, 1, 3*hidden_size)
    nn.init.xavier_uniform_(v_test)
    
    attention_weights_list = []

    decoder_hidden_t = torch.randn(batch_size_t, 1, hidden_size).cuda().to(device)
    #v_test = torch.randn(batch_size_t, 1, 3*hidden_size)
    #nn.init.xavier_uniform_(decoder_hidden_t)
    #nn.init.xavier_uniform_(v_test)
    
    decoder_hidden_t = decoder_hidden_t * x_mean_t
    decoder_hidden_t = torch.tanh(decoder_hidden_t)
    

    prediction = torch.zeros(batch_size_t,maxlen)
    #label = torch.zeros(batch_size_t,maxlen)
    prediction_sub = []
    label_sub = []
    label_real = []
    prediction_real = []

    decoder_attention_t = torch.zeros(batch_size_t,1,dense_input,output_area_t).cuda().to(device)
    attention_sum_t = torch.zeros(batch_size_t,1,dense_input,output_area_t).cuda().to(device)

    m = torch.nn.ZeroPad2d((0,maxlen-y_t.size()[1],0,0))
    y_t = m(y_t)
    for i in range(maxlen):
        decoder_output, decoder_hidden_t, decoder_attention_t, attention_sum_t = attn_decoder1(decoder_input_t,
                                                                                         decoder_hidden_t,
                                                                                         v_test,
                                                                                         output_highfeature_t,
                                                                                         output_area_t,
                                                                                         attention_sum_t,
                                                                                         decoder_attention_t,dense_input,batch_size_t,h_mask_t,w_mask_t,gpu,epoch)
        print('decoder_attention_t------',decoder_attention_t.shape)
        for i in range(batch_size_t):
            image = x_t[i][0].cpu().numpy()
            attention_weights = decoder_attention_t[i]  # Extract attention weights for the current example
            output_path = f'./test_bam_trail_2_{i}.png'

            visualize_attention(image, attention_weights, output_path)

        topv,topi = torch.max(decoder_output,2)
        if torch.sum(topi)==0:
            break
        decoder_input_t = topi
        decoder_input_t = decoder_input_t.view(batch_size_t)
        #print(topi.size()) 16,1

        # prediction
        prediction[:,i] = decoder_input_t
    
    #attention_weights_list.append(decoder_attention_t.detach().cpu().numpy())

print('----------------------completed')

'''


# Use the visualize_attention function to visualize the attention weights



'''
    for i in range(batch_size_t):
        for j in range(maxlen):
            if int(prediction[i][j]) ==0:
                break
            else:
                prediction_sub.append(int(prediction[i][j]))
                prediction_real.append(worddicts_r[int(prediction[i][j])])
        if len(prediction_sub)<maxlen:
            prediction_sub.append(0)

        for k in range(y_t.size()[1]):
            if int(y_t[i][k]) ==0:
                break
            else:
                label_sub.append(int(y_t[i][k]))
                label_real.append(worddicts_r[int(y_t[i][k])])
        label_sub.append(0)

        dist, llen, hit, ins, dls = cmp_result(label_sub, prediction_sub)
        wer_step = float(dist) / llen

        total_dist += dist
        total_label += llen
        total_line += 1
        if dist == 0:
            total_line_rec = total_line_rec+ 1
        if dist <= 1:
            total_line_rec_1 = total_line_rec_1+ 1
        if dist <= 2:
            total_line_rec_2 = total_line_rec_2+ 1
        if dist <= 3:
            total_line_rec_3 = total_line_rec_3+ 1

        if wer_step == 0.0:
            if len(label_real) >=1 and len(label_real) <= 5:
                c_1_5 = c_1_5 + 1 
            elif len(label_real) >=6 and len(label_real) <= 10:
                c_6_10 = c_6_10 + 1
            elif len(label_real) >=11 and len(label_real) <= 15:
                c_11_15 = c_11_15 + 1
            elif len(label_real) >=16 and len(label_real) <= 20:
                c_16_20 = c_16_20 + 1
            elif len(label_real) >=21 and len(label_real) <= 25:
                c_21_25 = c_21_25 + 1
            elif len(label_real) >=26 and len(label_real) <= 30:
                c_26_30 = c_26_30 + 1
            elif len(label_real) >=31 and len(label_real) <= 35:
                c_31_35 = c_31_35 + 1
            elif len(label_real) >=36 and len(label_real) <= 40:
                c_36_40 = c_36_40 + 1
            elif len(label_real) >=41 and len(label_real) <= 45:
                c_41_45 = c_41_45 + 1
            else:
                c_45_plus = c_45_plus + 1
            error_0.write('\n prediction is :' + str(prediction_real))
            error_0.write('\n label is :' + str(label_real) + ' length is: ' + str(len(label_real)))
            error_0.write('\n the wer is :' + str(wer_step))
            error_0.write('\n dist, llen, hit, ins, dls :' + str(dist) + ',' + str(llen) + ',' + str(hit) + ',' + str(ins) + ',' + str(dls) + ',')
        else:
            if len(label_real) >=1 and len(label_real) <= 5:
                e_1_5 = e_1_5 + 1 
            elif len(label_real) >=6 and len(label_real) <= 10:
                e_6_10 = e_6_10 + 1
            elif len(label_real) >=11 and len(label_real) <= 15:
                e_11_15 = e_11_15 + 1
            elif len(label_real) >=16 and len(label_real) <= 20:
                e_16_20 = e_16_20 + 1
            elif len(label_real) >=21 and len(label_real) <= 25:
                e_21_25 = e_21_25 + 1
            elif len(label_real) >=26 and len(label_real) <= 30:
                e_26_30 = e_26_30 + 1
            elif len(label_real) >=31 and len(label_real) <= 35:
                e_31_35 = e_31_35 + 1
            elif len(label_real) >=36 and len(label_real) <= 40:
                e_36_40 = e_36_40 + 1
            elif len(label_real) >=41 and len(label_real) <= 45:
                e_41_45 = e_41_45 + 1
            else:
                e_45_plus = e_45_plus + 1
            error.write('\n prediction is :' + str(prediction_real))
            error.write('\n label is :' + str(label_real) + ' length is: ' + str(len(label_real)))
            error.write('\n the wer is :' + str(wer_step))
            error.write('\n dist, llen, hit, ins, dls :' + str(dist) + ',' + str(llen) + ',' + str(hit) + ',' + str(ins) + ',' + str(dls) + ',')
        


        label_sub = []
        prediction_sub = []
        label_real = []
        prediction_real = []


  
    # dist, llen, hit, ins, dls = cmp_result(label, prediction)
    # wer_step = float(dist) / llen
    # print('the wer is %.5f' % (wer_step))


    # if wer_step <= 0.1:
    #     wer_1 += 1
    # elif 0.1 < wer_step <= 0.2:
    #     wer_2 += 1
    # elif 0.2 < wer_step <= 0.3:
    #     wer_3 += 1
    # elif 0.3 < wer_step <= 0.4:
    #     wer_4 += 1
    # elif 0.4 < wer_step <= 0.5:
    #     wer_5 += 1
    # elif 0.5 < wer_step <= 0.6:
    #     wer_6 += 1
    # else:
    #     wer_up += 1

    # hit_all += hit
    # ins_all += ins
    # dls_all += dls
    # total_dist += dist
    # total_label += llen
    # total_line += 1
    # if dist == 0:
    #     total_line_rec += 1

#error.close()
print("total_line_rec:",total_line_rec)
error.write('\n total_line_rec is' + str(total_line_rec))
wer = float(total_dist) / total_label
sacc = float(total_line_rec) / total_line
sacc_1_per = float(total_line_rec_1) / total_line
sacc_2_per = float(total_line_rec_2) / total_line
sacc_3_per = float(total_line_rec_3) / total_line
print('wer is %.5f' % (wer))
print('sacc is %.5f ' % (sacc))
print('sacc_1_per is %.5f ' % (sacc_1_per))
print('sacc_2_per is %.5f ' % (sacc_2_per))
print('sacc_3_per is %.5f ' % (sacc_3_per))
error.write('\n wer is %.5f' % (wer))
error.write('\n sacc is %.5f ' % (sacc))
error.write('\n sacc_1_per is %.5f ' % (sacc_1_per))
error.write('\n sacc_2_per is %.5f ' % (sacc_2_per))
error.write('\n sacc_3_per is %.5f ' % (sacc_3_per))

error.write('\n e_1_5 is %.5f' % (e_1_5))
error.write('\n e_6_10 is %.5f' % (e_6_10))
error.write('\n e_11_15 is %.5f' % (e_11_15))
error.write('\n e_16_20 is %.5f' % (e_16_20))
error.write('\n e_21_25 is %.5f' % (e_21_25))
error.write('\n e_26_30 is %.5f' % (e_26_30))
error.write('\n e_31_35 is %.5f' % (e_31_35))
error.write('\n e_36_40 is %.5f' % (e_36_40))
error.write('\n e_41_45 is %.5f' % (e_41_45))
error.write('\n e_45_plus is %.5f' % (e_45_plus))

error_0.write('\n c_1_5 is %.5f' % (c_1_5))
error_0.write('\n c_6_10 is %.5f' % (c_6_10))
error_0.write('\n c_11_15 is %.5f' % (c_11_15))
error_0.write('\n c_16_20 is %.5f' % (c_16_20))
error_0.write('\n c_21_25 is %.5f' % (c_21_25))
error_0.write('\n c_26_30 is %.5f' % (c_26_30))
error_0.write('\n c_31_35 is %.5f' % (c_31_35))
error_0.write('\n c_36_40 is %.5f' % (c_36_40))
error_0.write('\n c_41_45 is %.5f' % (c_41_45))
error_0.write('\n c_45_plus is %.5f' % (c_45_plus))

# print('hit is %d' % (hit_all))
# print('ins is %d' % (ins_all))
# print('dls is %d' % (dls_all))
# print('wer loss is %.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f' % (wer_1, wer_2, wer_3, wer_4, wer_5, wer_6, wer_up))
error.close()
error_0.close()
'''

error_0.close()

#output_grad = torch.autograd.grad(torch.sum(output), input_data, retain_graph=True)[0]

# Compute the relevance scores using the lrp() method
#input_relevance, hidden_relevance = model.gru.lrp(input_data, hidden_state, output_grad)