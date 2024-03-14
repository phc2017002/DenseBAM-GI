import torch
import math
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy
import torch.utils.data as data
#from data_iterator import dataIterator
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
import sys
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


torch.backends.cudnn.benchmark = False

features={}

channels=1

all_equation = []


def dataIterator(feature_file,batch_size,batch_Imagesize,maxlen,maxImagesize):

    imageSize={}
    imagehigh={}
    imagewidth={}
    sentNum=0

    #print('feature_file-----', feature_file)
    
    

    #print('filename---------', filename)

    #image = cv2.imread(feature_file)

    key = feature_file.split('/')[-1].split('.')[0]
    #print('key-------',key)
    image_path = '/ssd_scratch/cvit/ani31101993/HMER/test_HMER/'
    #image_file = image_path + key
    im = cv2.imread(feature_file)
        
    #print('im  size------', im.size)
    gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #print('gray_image size------', gray_image.size)
        
    im = gray_image 
        
    mat = numpy.zeros([channels, im.shape[0], im.shape[1]], dtype='uint8')
    for channel in range(channels):
        #image_file = image_path + key
        im = cv2.imread(feature_file)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        mat[channel,:,:] = im
    sentNum = sentNum + 1
    features[key] = mat

    uid = feature_file.split('/')[-1].split('.')[0]
        
    imageSize[uid]=features[key].shape[1]*features[key].shape[2]
    imagehigh[uid]=features[key].shape[1]
    imagewidth[uid]=features[key].shape[2]

    imageSize= sorted(imageSize.items(), key=lambda d:d[1],reverse=True) # sorted by sentence length,  return a list with each triple element


    feature_batch=[]
    feature_total=[]
    uidList=[]

    batch_image_size=0
    biggest_image_size=0
    i=0
    for uid,size in imageSize:
        if size>biggest_image_size:
            biggest_image_size=size
        fea=features
        batch_image_size=biggest_image_size*(i+1)

        if size>maxImagesize:
            print('image', uid, 'size bigger than', maxImagesize, 'ignore')
            continue
            # print('image', uid, 'size bigger than', maxImagesize, 'ignore')

        else:
            uidList.append(uid)
            if batch_image_size>batch_Imagesize or i==batch_size: # a batch is full

                if feature_batch:
                    feature_total.append(feature_batch)

                i=0
                biggest_image_size=size
                feature_batch=[]
                feature_batch.append(fea)
                batch_image_size=biggest_image_size*(i+1)
                i+=1
            else:
                feature_batch.append(fea)
                i+=1

    # last
    feature_total.append(feature_batch)

    return feature_total

'''
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
'''

def load_dict(dictFile):
    fp=open(dictFile)
    stuff=fp.readlines()
    fp.close()
    lexicon={}
    for l in stuff:
        w=l.strip().split()
        lexicon[w[0]]=int(w[1])

    #print('total words/phones',len(lexicon))
    return lexicon

#img_path='/ssd_scratch/cvit/ani31101993/HMER/test_HMER/'
dictionaries=['./dictionary.txt']
batch_Imagesize=16
valid_batch_Imagesize=16
batch_size_t=1
maxlen=48
maxImagesize=300000
hidden_size = 256
gpu = [0]

if __name__ == "__main__":
    # Check if the img_path argument is provided
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <img_path>")
        sys.exit(1)
    
    # Get the img_path from command-line arguments
    img_path = sys.argv[1]

    # Call the dataIterator function with the img_path
    #dataIterator(img_path)

    error = open('./test_output.txt',"w+")

    worddicts = load_dict(dictionaries[0])
    worddicts_r = [None] * len(worddicts)
    for kk, vv in worddicts.items():
        worddicts_r[vv] = kk

    test = dataIterator(img_path,batch_size=1,batch_Imagesize=batch_Imagesize,maxlen=maxlen,maxImagesize=maxImagesize)

    class custom_dset(data.Dataset):
        def __init__(self,train):
            self.train = train
            #self.train_label = train_label

        def __getitem__(self, index):
            element_type = type(self.train[index][0])
            #print('type of self-train--------', element_type)

            data_dicts = self.train[index]

            all_values = [list(data_dict.values()) for data_dict in data_dicts]

            data_array = np.array(all_values)

            train_setting = torch.from_numpy(data_array)

            #train_setting = torch.from_numpy(numpy.array(self.train[index]))
            #label_setting = torch.from_numpy(numpy.array(self.train_label[index])).type(torch.LongTensor)

            size = train_setting.size()
            #print('size-------------', size)
            train_setting = train_setting.view(1,size[3],size[4])
            #label_setting = label_setting.view(-1)

            return train_setting

        def __len__(self):
            return len(self.train)

    off_image_test = custom_dset(test)
    #print(off_image_train[10])


    def imresize(im,sz):
        pil_im = Image.fromarray(im)
        return numpy.array(pil_im.resize(sz))


    def collate_fn(batch):
        #print('accessing the batch-------', batch[0].shape)
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        #batch.sort(key=lambda x: len(x[0]) if len(x) > 1 else 0, reverse=True)

        tensors = [item[0] for item in batch]

        img = zip(*batch)
        aa1 = 0
        bb1 = 0
        k = 0
        k1 = 0

        #max_len = len(label[0])+1

        #print('len(tensors)------', len(tensors))

        for j in range(len(tensors)):
            size = tensors[j].size()
            print('size-----',size)
            if size[0] > aa1:
                aa1 = size[0]
            if size[1] > bb1:
                bb1 = size[1]

        for ii in img:
            #print('length', len(ii))
            #ii = ii.float()
            ii = tuple(tensor.float() for tensor in ii)
            #print('ii.size()---',  ii[0].size())
            img_size_h = ii[0].size()[0]
            img_size_w = ii[0].size()[1]

            ii = tuple(tensor.unsqueeze(0) for tensor in ii)

            ij = list(ii)

            #ij[0] = ij[0].unsqueeze(0)
            #print('ii.size()--- after',  ij[0].size())

            img_mask_sub_s = torch.ones(1,img_size_h,img_size_w).type(torch.FloatTensor)
            img_mask_sub_s = img_mask_sub_s*255.0

            concatenated_tensors = [torch.cat((tensor, img_mask_sub_s), dim=0) for tensor in ii]
            img_mask_sub = torch.cat(concatenated_tensors, dim=0)

            #img_mask_sub = torch.cat((ii,img_mask_sub_s),dim=0)
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

        '''
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
        '''

        img_padding_mask = img_padding_mask/255.0
        return img_padding_mask

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

    #print('--------------------herererer')

    encoder.load_state_dict(torch.load('./Pretrained_model/encoder_lr0.00000_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl'))
    attn_decoder1.load_state_dict(torch.load('./Pretrained_model/attn_decoder_lr0.00000_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl'))

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

    #print('test_loader------------',len(test_loader))
    
    for step_t, x_t in enumerate(test_loader):
        #print('size x_t--------', x_t.size())
        x_real_high = x_t.size()[2]
        x_real_width = x_t.size()[3]
        if x_t.size()[0]<batch_size_t:
            break
        
        #print('x_t-------------',x_t.shape)
        #print('step_t',step_t)
        
        h_mask_t = []
        w_mask_t = []
        for i in x_t:
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
        #y_t = y_t.cuda().to(device)
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
        nn.init.xavier_uniform_(decoder_hidden_t)
        
        decoder_hidden_t = decoder_hidden_t * x_mean_t
        decoder_hidden_t = torch.tanh(decoder_hidden_t)
        
    
        prediction = torch.zeros(batch_size_t,maxlen)
        label = torch.zeros(batch_size_t,maxlen)
        prediction_sub = []
        label_sub = []
        label_real = []
        prediction_real = []
    
        decoder_attention_t = torch.zeros(batch_size_t,1,dense_input,output_area_t).cuda().to(device)
        attention_sum_t = torch.zeros(batch_size_t,1,dense_input,output_area_t).cuda().to(device)
    
        #m = torch.nn.ZeroPad2d((0,maxlen-y_t.size()[1],0,0))
        #y_t = m(y_t)
        #print('hereee----------')
        for i in range(maxlen):
            decoder_output, decoder_hidden_t, v_test, decoder_attention_t, attention_sum_t = attn_decoder1(decoder_input_t, decoder_hidden_t, v_test, output_highfeature_t, output_area_t, attention_sum_t, decoder_attention_t,dense_input,batch_size_t,h_mask_t,w_mask_t,gpu,epoch)      
    
          
    
            
            topv,topi = torch.max(decoder_output,2)
            if torch.sum(topi)==0:
                break
            decoder_input_t = topi
            decoder_input_t = decoder_input_t.view(batch_size_t)
            
            prediction[:,i] = decoder_input_t
        with open("predictions.txt", "a") as file:
            for i in range(batch_size_t):
                for j in range(maxlen):
                    if int(prediction[i][j]) ==0:
                        break
                    else:
                        prediction_sub.append(int(prediction[i][j]))
                        prediction_real.append(worddicts_r[int(prediction[i][j])])
    
            
                print('testing--------', ' '.join(prediction_real))
                file.write(' '.join(prediction_real) + "\n")
                #all_equation.append(' '.join(prediction_real))
            
    
            label_sub = []
            prediction_sub = []
            label_real = []
            prediction_real = []
        
    
