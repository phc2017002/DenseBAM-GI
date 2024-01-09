import torch
import math
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy
import torch.utils.data as data
from data_iterator import dataIterator
from Densenet_torchvision import densenet121
from Attention_RNN_test_momentum import AttnDecoderRNN
#from Resnet101 import resnet101
import random
#import matplotlib.pyplot as plt
from PIL import Image
import time

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
    return dist, len(label)

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

datasets=['./offline-train.pkl','./train_caption.txt']
valid_datasets=['./offline-test-2019.pkl', './test-caption-2019.txt']#./offline-test.pkl,./offline-test.pkl
valid_datasets_1=['./offline-test.pkl','./test_caption.txt']
dictionaries=['./dictionary.txt']
batch_Imagesize=500000
valid_batch_Imagesize=500000
# batch_size for training and testing
batch_size=2
batch_size_t=2
# the max (label length/Image size) in training and testing
# you can change 'maxlen','maxImagesize' by the size of your GPU
maxlen=48
maxImagesize= 120000
# hidden_size in RNN
hidden_size = 256
context = 256
# teacher_forcing_ratio 
teacher_forcing_ratio = 1
# change the gpu id 
gpu = [2]
# learning rate
lr_rate = 0.0001
# flag to remember when to change the learning rate
flag = 0
# exprate
exprate = 0
best_wer = 1.0
best_total_line_rec = 0
exprate_1 = 0
best_wer_1 =0
best_total_line_rec_1 = 0

error = open('./model_test_momentum_0.1_step_size_0.1_GRU_2019_batch_size_2_trail_1.txt',"w+")
error_1 = open('./model_test_momentum_0.1_step_size_0.1_with_GRU_2014_batch_size_2_trail_1.txt',"w+")  
# worddicts
worddicts = load_dict(dictionaries[0])
worddicts_r = [None] * len(worddicts)
for kk, vv in worddicts.items():
        worddicts_r[vv] = kk


print('1')
#load train data and test data
train,train_label = dataIterator(
                                    datasets[0], datasets[1],worddicts,batch_size=1,
                                    batch_Imagesize=batch_Imagesize,maxlen=maxlen,maxImagesize=maxImagesize
                                 )
len_train = len(train)

test,test_label = dataIterator(
                                    valid_datasets[0],valid_datasets[1],worddicts,batch_size=1,
                                    batch_Imagesize=batch_Imagesize,maxlen=maxlen,maxImagesize=maxImagesize
                                )
len_test = len(test)
#len_test_1 = len(train)
test_1,test_label_1 = dataIterator(
                                    valid_datasets_1[0],valid_datasets_1[1],worddicts,batch_size=1,
                                    batch_Imagesize=batch_Imagesize,maxlen=maxlen,maxImagesize=maxImagesize
                                )

len_test_1 = len(test_1)

print('2')


class custom_dset(data.Dataset):
    def __init__(self,train,train_label,batch_size):
        self.train = train
        self.train_label = train_label
        self.batch_size = batch_size

    def __getitem__(self, index):
        train_setting = torch.from_numpy(numpy.array(self.train[index]))
        label_setting = torch.from_numpy(numpy.array(self.train_label[index])).type(torch.LongTensor)

        size = train_setting.size()
        train_setting = train_setting.view(1,size[2],size[3])
        label_setting = label_setting.view(-1)
        return train_setting,label_setting

    def __len__(self):
        return len(self.train)


print('3')

#print('train',train)
#print('train_label',train_label)
off_image_train = custom_dset(train,train_label,batch_size)
off_image_test = custom_dset(test,test_label,batch_size)
off_image_test_1 = custom_dset(test_1,test_label_1,batch_size)

# collate_fn is writting for padding imgs in batch. 
# As images in my dataset are different size, so the padding is necessary.
# Padding images to the max image size in a mini-batch and cat a mask. 
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


print('4')


train_loader = torch.utils.data.DataLoader(
    dataset = off_image_train,
    batch_size = batch_size,
    shuffle = True,
    collate_fn = collate_fn,
    num_workers=2,
    )
test_loader = torch.utils.data.DataLoader(
    dataset = off_image_test,
    batch_size = batch_size_t,
    shuffle = True,
    collate_fn = collate_fn,
    num_workers=2,
)
test_loader_1 = torch.utils.data.DataLoader(
    dataset = off_image_test_1,
    batch_size = batch_size_t,
    shuffle = True,
    collate_fn = collate_fn,
    num_workers=2,
)

print('5')


def my_train(target_length,attn_decoder1,
             output_highfeature, output_area,y,criterion,encoder_optimizer1,decoder_optimizer1,x_mean,dense_input,h_mask,w_mask,gpu,
             decoder_input,decoder_hidden,decoder_v,attention_sum,decoder_attention):
    loss = 0

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    flag_z = [0]*batch_size

    if use_teacher_forcing:
        encoder_optimizer1.zero_grad()
        decoder_optimizer1.zero_grad()
        my_num = 0

        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_v, decoder_attention, attention_sum = attn_decoder1(decoder_input,
                                                                                             decoder_hidden,
                                                                                             decoder_v,
                                                                                             output_highfeature,
                                                                                             output_area,
                                                                                             attention_sum,
                                                                                             decoder_attention,
                                                                                             dense_input,batch_size,h_mask,w_mask,gpu)
            
            
            #print(decoder_output.size()) #(batch,1,112)
            #decoder_output = decoder_output.view(batch_size,1,112)
            y = y.unsqueeze(0)
            for i in range(batch_size):
                if int(y[0][i][di]) == 0:
                    flag_z[i] = flag_z[i]+1
                    if flag_z[i] > 1:
                        continue
                    else:
                        loss += criterion(decoder_output[i], y[:,i,di])
                else:
                    loss += criterion(decoder_output[i], y[:,i,di])

            if int(y[0][0][di]) == 0:
                break
            decoder_input = y[:,:,di]
            decoder_input = decoder_input.squeeze(0)
            y = y.squeeze(0)

        loss.backward()

        encoder_optimizer1.step()
        decoder_optimizer1.step()
        return loss.item()

    else:
        encoder_optimizer1.zero_grad()
        decoder_optimizer1.zero_grad()
        my_num = 0
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_v, decoder_attention,attention_sum= attn_decoder1(decoder_input, decoder_hidden, decoder_v,
                                                                                output_highfeature, output_area,
                                                                                attention_sum,decoder_attention,dense_input,batch_size,
                                                                                h_mask,w_mask,gpu)
            #print(decoder_output.size()) 1*10*112
            #print(y.size())  1*37
            #topi (b,1)
            topv,topi = torch.max(decoder_output,2)
            decoder_input = topi
            decoder_input = decoder_input.view(batch_size)

            y = y.unsqueeze(0)
            #print(y_t)

            # 1*bs*17
            #decoder_output = decoder_ouput(batch_size,1,112)
            for k in range(batch_size):
                if int(y[0][k][di]) == 0:
                    flag_z[k] = flag_z[k]+1
                    if flag_z[k] > 1:
                        continue
                    else:
                        loss += criterion(decoder_output[k], y[:,k,di])
                else:
                    loss += criterion(decoder_output[k], y[:,k,di])

            y = y.squeeze(0)
            # if int(topi[0]) == 0:
            #     break
        loss.backward()
        encoder_optimizer1.step()
        decoder_optimizer1.step()
        return loss.item()


print('6')



device = ('cuda:2')

#error = open("with_dropout_GRU.txt", "w+")

encoder = densenet121()

'''
pthfile = './densenet121-a639ec97.pth'
pretrained_dict = torch.load(pthfile) 
encoder_dict = encoder.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
encoder_dict.update(pretrained_dict)
encoder.load_state_dict(encoder_dict)
'''

attn_decoder1 = AttnDecoderRNN(hidden_size,112,dropout_p=0.5)

encoder = torch.nn.DataParallel(encoder,device_ids=gpu)
attn_decoder1 = torch.nn.DataParallel(attn_decoder1,device_ids=gpu)
encoder = encoder.cuda().to(device)
attn_decoder1 = attn_decoder1.cuda().to(device)

encoder.load_state_dict(torch.load('./model_test_momentum_RGRU_testing_batch_size_2_trail_9/encoder_lr0.00010_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl'))
attn_decoder1.load_state_dict(torch.load('./model_test_momentum_RGRU_testing_batch_size_2_trail_9/attn_decoder_lr0.00010_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl'))



children_of_child_counter = 0
children_of_child_child_counter = 0
children_of_child_child_child_counter = 0
child_counter = 0

'''
for child in encoder.children():
    if child_counter == 0:
        for children_of_child in child.children():
            children_of_child_counter += 1 
            if children_of_child_counter == 1:
                for children_of_child_child in children_of_child.children():
                    children_of_child_child_counter += 1
                    if children_of_child_child_counter < 9:
                        for param in children_of_child_child.parameters():
                            param.requires_grad = False
                        print(children_of_child_child_counter, 'was frozen')
                    elif children_of_child_child_counter == 9:
                        for children_of_child_child_child in children_of_child_child.children():
                            children_of_child_child_child_counter += 1
                            print(children_of_child_child_child_counter, 'was not frozen')
                    else:   
                        print(children_of_child_child_counter,"was not frozen")
'''
'''
for child in encoder.children():
    if child_counter == 0:
        for children_of_child in child.children():
            children_of_child_counter += 1 
            if children_of_child_counter == 1:
                for children_of_child_child in children_of_child.children():
                    children_of_child_child_counter += 1
                    if children_of_child_child_counter < 9:
                        for param in children_of_child_child.parameters():
                            param.requires_grad = False
                        print(children_of_child_child_counter, 'was frozen')
                    elif children_of_child_child_counter == 9:
                        for children_of_child_child_child in children_of_child_child.children():
                            children_of_child_child_child_counter += 1
                            if children_of_child_child_child_counter < 24:
                                for param in children_of_child_child_child.parameters():
                                    param.requires_grad = False
                                print(children_of_child_child_child_counter, 'was frozen')
                            else:
                                print(children_of_child_child_child_counter, 'was not frozen')
                    else:   
                        print(children_of_child_child_counter,"was not frozen")

'''
'''
for child in encoder.children():
    for children_of_child in child.children():
        print(children_of_child)

'''
'''
for child in encoder.children():
    if child_counter == 0:
        for children_of_child in child.children():
            if children_of_child_counter < 8:
                for param in children_of_child.parameters():
                    param.requires_grad = False
                print(children_of_child_counter," was frozen")
            elif children_of_child_counter == 8:
                for children_of_child_child in children_of_child.children():
                    if children_of_child_child_counter < 12:
                        for param in children_of_child_child.parameters():
                            param.requires_grad = False
                        print("was  frozen")
                    else:
                        print(children_of_child_child_counter,"was not frozen")
                    children_of_child_child_counter += 1
            else:
                print(children_of_child_counter,"was not frozen") 
            children_of_child_counter += 1

'''

criterion = nn.NLLLoss()
# encoder.load_state_dict(torch.load('model/encoder_lr0.00001_BN_te1_d05_SGD_bs8_mask_conv_bn_b.pkl'))
# attn_decoder1.load_state_dict(torch.load('model/attn_decoder_lr0.00001_BN_te1_d05_SGD_bs8_mask_conv_bn_b.pkl'))
decoder_input_init = torch.LongTensor([111]*batch_size)
decoder_hidden_init = torch.randn(batch_size, 1, hidden_size)
v_init = torch.randn(batch_size, 1, 3*hidden_size)
nn.init.xavier_uniform_(decoder_hidden_init)
nn.init.xavier_uniform_(v_init)

for epoch in range(300):
    encoder_optimizer1 = torch.optim.SGD(encoder.parameters(), lr=lr_rate,momentum=0.9,weight_decay=0.01)
    decoder_optimizer1 = torch.optim.SGD(attn_decoder1.parameters(), lr=lr_rate,momentum=0.9,weight_decay=0.01)

    #t0 = time.time()
    print('8.1')
    # # if using SGD optimizer
    # if epoch+1 == 50:
    #     lr_rate = lr_rate/10
    #     encoder_optimizer1 = torch.optim.SGD(encoder.parameters(), lr=lr_rate,momentum=0.9)
    #     decoder_optimizer1 = torch.optim.SGD(attn_decoder1.parameters(), lr=lr_rate,momentum=0.9)
    # if epoch+1 == 75:
    #     lr_rate = lr_rate/10
    #     encoder_optimizer1 = torch.optim.SGD(encoder.parameters(), lr=lr_rate,momentum=0.9)
    #     decoder_optimizer1 = torch.optim.SGD(attn_decoder1.parameters(), lr=lr_rate,momentum=0.9)

    print('8.2')

    running_loss=0
    whole_loss = 0

    print('8.3')

    encoder.train(mode=True)
    attn_decoder1.train(mode=True)

    #print('8.4')
    #print('just before')

    # this is the train
    for step,(x,y) in enumerate(train_loader):
        if x.size()[0]<batch_size:
            break   
        #print('inside train_loader')
        h_mask = []
        w_mask = []
        for i in x:
            #h*w
            size_mask = i[1].size()
            s_w = str(i[1][0])
            s_h = str(i[1][:,1])
            w = s_w.count('1')
            h = s_h.count('1')
            h_comp = int(h/16)+1
            w_comp = int(w/16)+1
            h_mask.append(h_comp)
            w_mask.append(w_comp)

        x = x.cuda().to(device)
        y = y.cuda().to(device)
        # out is CNN featuremaps
        output_highfeature = encoder(x)
        x_mean=[]
        for i in output_highfeature:
            x_mean.append(float(torch.mean(i)))
        # x_mean = torch.mean(output_highfeature)
        # x_mean = float(x_mean)
        for i in range(batch_size):
            decoder_hidden_init[i] = decoder_hidden_init[i]*x_mean[i]
            decoder_hidden_init[i] = torch.tanh(decoder_hidden_init[i])

        # dense_input is height and output_area is width which is bb
        output_area1 = output_highfeature.size()

        output_area = output_area1[3]
        dense_input = output_area1[2]
        target_length = y.size()[1]
        #print("target_length",target_length)
        attention_sum_init = torch.zeros(batch_size,1,dense_input,output_area).cuda().to(device)
        decoder_attention_init = torch.zeros(batch_size,1,dense_input,output_area).cuda().to(device)

        running_loss += my_train(target_length,attn_decoder1,output_highfeature,
                                output_area,y,criterion,encoder_optimizer1,decoder_optimizer1,x_mean,dense_input,h_mask,w_mask,gpu,
                                decoder_input_init,decoder_hidden_init,v_init,attention_sum_init,decoder_attention_init)

        #torch.cuda.synchronize()
        #t1 = time.time()
        #eta = t1-t0
        #if step % 20 == 19:
        #    print('\n Elapsed time :{:.2f} ms'.format((eta)*1000))
        if step % 20 == 19:
            pre = ((step+1)/len_train)*100*batch_size
            whole_loss += running_loss
            running_loss = running_loss/(batch_size*20)
            print('epoch is %d, lr rate is %.5f, te is %.3f, batch_size is %d, loading for %.3f%%, running_loss is %f' %(epoch,lr_rate,teacher_forcing_ratio, batch_size,pre,running_loss))
            error.write('\n epoch is %d, lr rate is %.5f, te is %.3f, batch_size is %d, loading for %.3f%%, running_loss is %f' %(epoch,lr_rate,teacher_forcing_ratio, batch_size,pre,running_loss))
            error_1.write('\n epoch is %d, lr rate is %.5f, te is %.3f, batch_size is %d, loading for %.3f%%, running_loss is %f' %(epoch,lr_rate,teacher_forcing_ratio, batch_size,pre,running_loss))
            # with open("training_data/running_loss_%.5f_pre_GN_te05_d02_all.txt" %(lr_rate),"a") as f:
            #     f.write("%s\n"%(str(running_loss)))
            running_loss = 0


    print('8.5')


    loss_all_out = whole_loss / len_train
    print("epoch is %d, the whole loss is %f" % (epoch, loss_all_out))
    # with open("training_data/whole_loss_%.5f_pre_GN_te05_d02_all.txt" % (lr_rate), "a") as f:
    #     f.write("%s\n" % (str(loss_all_out)))

    # this is the prediction and compute wer loss
    total_dist = 0
    total_label = 0
    total_line = 0
    total_line_rec = 0
    whole_loss_t = 0
    total_dist_1 = 0
    total_label_1 = 0
    total_line_1 = 0
    total_line_rec_1 = 0
    total_line_rec_2 = 0
    total_line_rec_3 = 0
    total_line_rec_11 = 0
    total_line_rec_22 = 0
    total_line_rec_33 = 0
    total_line_rec_t_1 = 0

    encoder.eval()
    attn_decoder1.eval()
    print('Now, begin testing!!')
    error.write('\n Now, begin testing!!')
    error_1.write('\n Now, begin testing!!')

    for step_t, (x_t, y_t) in enumerate(test_loader):
        x_real_high = x_t.size()[2]
        x_real_width = x_t.size()[3]
        if x_t.size()[0]<batch_size_t:
            break
        print('testing for %.3f%%'%(step_t*100*batch_size_t/len_test),end='\r')
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
        output_highfeature_t = encoder(x_t)

        x_mean_t = torch.mean(output_highfeature_t)
        x_mean_t = float(x_mean_t)
        output_area_t1 = output_highfeature_t.size()
        output_area_t = output_area_t1[3]
        dense_input = output_area_t1[2]

        decoder_input_t = torch.LongTensor([111]*batch_size_t)
        decoder_input_t = decoder_input_t
        decoder_hidden_t = torch.randn(batch_size_t, 1, hidden_size).cuda().to(device)
        v_test = torch.randn(batch_size_t, 1, 3*hidden_size)
        nn.init.xavier_uniform_(decoder_hidden_t)
        nn.init.xavier_uniform_(v_test)

        x_mean_t=[]
        for i in output_highfeature_t:
            x_mean_t.append(float(torch.mean(i)))
        # x_mean = torch.mean(output_highfeature)
        # x_mean = float(x_mean)
        for i in range(batch_size_t):
            decoder_hidden_t[i] = decoder_hidden_t[i]*x_mean_t[i]
            decoder_hidden_t[i] = torch.tanh(decoder_hidden_t[i])

        prediction = torch.zeros(batch_size_t,maxlen)
        #label = torch.zeros(batch_size_t,maxlen)
        prediction_sub = []
        label_sub = []
        decoder_attention_t = torch.zeros(batch_size_t,1,dense_input,output_area_t).cuda().to(device)
        attention_sum_t = torch.zeros(batch_size_t,1,dense_input,output_area_t).cuda().to(device)
        flag_z_t = [0]*batch_size_t
        loss_t = 0
        m = torch.nn.ZeroPad2d((0,maxlen-y_t.size()[1],0,0))
        y_t = m(y_t)
        for i in range(maxlen):
            decoder_output, decoder_hidden_t, v_test, decoder_attention_t, attention_sum_t = attn_decoder1(decoder_input_t,
                                                                                             decoder_hidden_t,
                                                                                             v_test,
                                                                                             output_highfeature_t,
                                                                                             output_area_t,
                                                                                             attention_sum_t,
                                                                                             decoder_attention_t,dense_input,batch_size_t,h_mask_t,w_mask_t,gpu)

            ### you can see the attention when testing

            # print('this is',i)
            # for i in range(batch_size_t):
            #     x_real = numpy.array(x_t[i][0].data.cpu())

            #     show = numpy.array(decoder_attention_t[i][0].data.cpu())
            #     show = imresize(show,(x_real_width,x_real_high))
            #     k_max = show.max()
            #     show = show/k_max

            #     show_x = x_real+show
            #     plt.imshow(show_x, interpolation='nearest', cmap='gray_r')
            #     plt.show()
            

            #decoder_output = decoder_output.view(batch_size_t,1,112)
            topv,topi = torch.max(decoder_output,2)
            # if torch.sum(y_t[0,:,i])==0:
            #     y_t = y_t.squeeze(0)
            #     break
            if torch.sum(topi)==0:
                break
            decoder_input_t = topi
            decoder_input_t = decoder_input_t.view(batch_size_t)

            # prediction
            prediction[:,i] = decoder_input_t

        for i in range(batch_size_t):
            for j in range(maxlen):
                if int(prediction[i][j]) ==0:
                    break
                else:
                    prediction_sub.append(int(prediction[i][j]))
            if len(prediction_sub)<maxlen:
                prediction_sub.append(0)

            for k in range(y_t.size()[1]):
                if int(y_t[i][k]) ==0:
                    break
                else:
                    label_sub.append(int(y_t[i][k]))
            label_sub.append(0)

            dist, llen = cmp_result(label_sub, prediction_sub)
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



            label_sub = []
            prediction_sub = []


    print('8.6')


    print('total_line_rec is',total_line_rec)
    error.write('\n total_line_rec is' + str(total_line_rec))
    wer = float(total_dist) / total_label
    sacc = float(total_line_rec) / total_line
    sacc_1_per = float(total_line_rec_1) / total_line
    sacc_2_per = float(total_line_rec_2) / total_line
    sacc_3_per = float(total_line_rec_3) / total_line
    print('wer is %.5f' % (wer))
    error.write('\n wer is %.5f' % (wer))
    error.write('\n sacc is %.5f ' % (sacc))
    error.write('\n sacc_1_per is %.5f ' % (sacc_1_per))
    error.write('\n sacc_2_per is %.5f ' % (sacc_2_per))
    error.write('\n sacc_3_per is %.5f ' % (sacc_3_per))
    print('sacc is %.5f ' % (sacc))
    print('sacc_1_per is %.5f ' % (sacc_1_per))
    print('sacc_2_per is %.5f ' % (sacc_2_per))
    print('sacc_3_per is %.5f ' % (sacc_3_per))
    # print('whole loss is %.5f'%(whole_loss_t/925))
    # with open("training_data/wer_%.5f_pre_GN_te05_d02_all.txt" % (lr_rate), "a") as f:
    #     f.write("%s\n" % (str(wer)))

    for step_t_1, (x_t_1, y_t_1) in enumerate(test_loader_1):
        x_real_high_1 = x_t_1.size()[2]
        x_real_width_1 = x_t_1.size()[3]
        if x_t_1.size()[0]<batch_size_t:
            break
        print('testing for %.3f%%'%(step_t_1*100*batch_size_t/len_test_1),end='\r')
        h_mask_t_1 = []
        w_mask_t_1 = []
        for i in x_t_1:
            #h*w
            size_mask_t_1 = i[1].size()
            s_w_t_1 = str(i[1][0])
            s_h_t_1 = str(i[1][:,1])
            w_t_1 = s_w_t_1.count('1')
            h_t_1 = s_h_t_1.count('1')
            h_comp_t_1 = int(h_t_1/16)+1
            w_comp_t_1 = int(w_t_1/16)+1
            h_mask_t_1.append(h_comp_t_1)
            w_mask_t_1.append(w_comp_t_1)

        x_t_1 = x_t_1.cuda().to(device)
        y_t_1 = y_t_1.cuda().to(device)
        output_highfeature_t_1 = encoder(x_t_1)

        x_mean_t_1 = torch.mean(output_highfeature_t_1)
        x_mean_t_1 = float(x_mean_t_1)
        output_area_t1_1 = output_highfeature_t_1.size()
        output_area_t_1 = output_area_t1_1[3]
        dense_input_1 = output_area_t1_1[2]

        decoder_input_t_1 = torch.LongTensor([111]*batch_size_t)
        #decoder_input_t_1 = decoder_input_t_1.cuda().to(device)
        v_test_1 = torch.randn(batch_size_t, 1, 3*hidden_size)
        decoder_hidden_t_1 = torch.randn(batch_size_t, 1, hidden_size).cuda().to(device)
        nn.init.xavier_uniform_(decoder_hidden_t_1)
        nn.init.xavier_uniform_(v_test_1)

        x_mean_t_1=[]
        for i in output_highfeature_t_1:
            x_mean_t_1.append(float(torch.mean(i)))
        # x_mean = torch.mean(output_highfeature)
        # x_mean = float(x_mean)
        for i in range(batch_size_t):
            decoder_hidden_t_1[i] = decoder_hidden_t_1[i]*x_mean_t_1[i]
            decoder_hidden_t_1[i] = torch.tanh(decoder_hidden_t_1[i])

        prediction_1 = torch.zeros(batch_size_t,maxlen)
        #label = torch.zeros(batch_size_t,maxlen)
        prediction_sub_1 = []
        label_sub_1 = []
        decoder_attention_t_1 = torch.zeros(batch_size_t,1,dense_input_1,output_area_t_1).cuda().to(device)
        attention_sum_t_1 = torch.zeros(batch_size_t,1,dense_input_1,output_area_t_1).cuda().to(device)
        flag_z_t_1 = [0]*batch_size_t
        loss_t_1 = 0
        m = torch.nn.ZeroPad2d((0,maxlen-y_t_1.size()[1],0,0))
        y_t_1 = m(y_t_1)
        for i in range(maxlen):
            decoder_output_1, decoder_hidden_t_1, v_test_1, decoder_attention_t_1, attention_sum_t_1 = attn_decoder1(decoder_input_t_1,
                                                                                             decoder_hidden_t_1,
                                                                                             v_test_1,
                                                                                             output_highfeature_t_1,
                                                                                             output_area_t_1,
                                                                                             attention_sum_t_1,
                                                                                             decoder_attention_t_1,dense_input_1,batch_size_t,h_mask_t_1,w_mask_t_1,gpu)

            ### you can see the attention when testing

            # print('this is',i)
            # for i in range(batch_size_t):
            #     x_real = numpy.array(x_t[i][0].data.cpu())

            #     show = numpy.array(decoder_attention_t[i][0].data.cpu())
            #     show = imresize(show,(x_real_width,x_real_high))
            #     k_max = show.max()
            #     show = show/k_max

            #     show_x = x_real+show
            #     plt.imshow(show_x, interpolation='nearest', cmap='gray_r')
            #     plt.show()
            

            #decoder_output = decoder_output.view(batch_size_t,1,112)
            topv,topi = torch.max(decoder_output_1,2)
            # if torch.sum(y_t[0,:,i])==0:
            #     y_t = y_t.squeeze(0)
            #     break
            if torch.sum(topi)==0:
                break
            decoder_input_t_1 = topi
            decoder_input_t_1 = decoder_input_t_1.view(batch_size_t)

            # prediction
            prediction_1[:,i] = decoder_input_t_1

        for i in range(batch_size_t):
            for j in range(maxlen):
                if int(prediction_1[i][j]) ==0:
                    break
                else:
                    prediction_sub_1.append(int(prediction_1[i][j]))
            if len(prediction_sub_1)<maxlen:
                prediction_sub_1.append(0)

            for k in range(y_t_1.size()[1]):
                if int(y_t_1[i][k]) ==0:
                    break
                else:
                    label_sub_1.append(int(y_t_1[i][k]))
            label_sub_1.append(0)

            dist_1, llen_1 = cmp_result(label_sub_1, prediction_sub_1)
            total_dist_1 += dist_1
            total_label_1 += llen_1
            total_line_1 += 1
            if dist_1 == 0:
                total_line_rec_t_1 = total_line_rec_t_1+ 1
            if dist_1 <= 1:
                total_line_rec_11 = total_line_rec_11+ 1
            if dist_1 <= 2:
                total_line_rec_22 = total_line_rec_22+ 1
            if dist_1 <= 3:
                total_line_rec_33 = total_line_rec_33+ 1


            label_sub_1 = []
            prediction_sub_1 = []


    print('8.6')


    
    print('total_line_rec is',total_line_rec_t_1)
    error_1.write('\n total_line_rec is' + str(total_line_rec_t_1))
    wer_1 = float(total_dist_1) / total_label_1
    sacc_1 = float(total_line_rec_t_1) / total_line_1
    sacc_1_1_per = float(total_line_rec_11) / total_line_1
    sacc_1_2_per = float(total_line_rec_22) / total_line_1
    sacc_1_3_per = float(total_line_rec_33) / total_line_1
    print('wer is %.5f' % (wer_1))
    error_1.write('\n wer is %.5f' % (wer_1))
    error_1.write('\n sacc_1 is %.5f ' % (sacc_1))
    error_1.write('\n sacc_1_1_per is %.5f ' % (sacc_1_1_per))
    error_1.write('\n sacc_1_2_per is %.5f ' % (sacc_1_2_per))
    error_1.write('\n sacc_1_3_per is %.5f ' % (sacc_1_3_per))
    print('sacc_1 is %.5f ' % (sacc_1))
    print('sacc_1_per is %.5f ' % (sacc_1_per))
    if (sacc > exprate):
        exprate = sacc
        best_wer = wer
        best_total_line_rec = total_line_rec
        print(exprate)
        error.write("\n saving the model....")
        error.write('\n encoder_lr%.5f_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl' %(lr_rate))
        print("saving the model....")
        print('encoder_lr%.5f_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl' %(lr_rate))
        torch.save(encoder.state_dict(), './model_test_momentum_0.1_step_size_0.1_GRU_2019_with_batch_size_2_trail_1/encoder_lr%.5f_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl'%(lr_rate))
        torch.save(attn_decoder1.state_dict(), './model_test_momentum_0.1_step_size_0.1_GRU_2019_with_batch_size_2_trail_1/attn_decoder_lr%.5f_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl'%(lr_rate))
        print("done")
        flag = 0
    else:
        flag = flag+1
        error.write('\n the best is %f' % (exprate))
        error.write('\n the best wer is %f' % (best_wer))
        error.write('\n highest total no of line recocognized till now is %f' % (best_total_line_rec))
        print('the best is %f' % (exprate))
        print('the best wer is %f' % (best_wer))
        print('highest total no of line recocognized till now is %f' % (best_total_line_rec))
        print('the loss is bigger than before,so do not save the model')

    if (sacc_1 > exprate_1):
        exprate_1 = sacc_1
        best_wer_1 = wer_1
        best_total_line_rec_1 = total_line_rec_1
        print('new best in training %f' % (exprate_1))
        error_1.write("\n saving the model....")
        error_1.write('\n encoder_lr%.5f_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl' %(lr_rate))
    else:
        error_1.write('\n the best is %f' % (exprate_1))
        error_1.write('\n the best wer is %f' % (best_wer_1))
        error_1.write('\n highest total no of line recocognized till now is %f' % (best_total_line_rec_1))
        print('the best in training is %f' % (exprate_1))
        print('the best WER in training is %f' % (best_wer_1))

    if flag == 10 :
        lr_rate = lr_rate*0.1
        flag = 0


error.close()


def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    run()




print('9')
