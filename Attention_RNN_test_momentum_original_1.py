import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from grucell_pytorch_in_vr_test_momentum_added import GRUCell


device = ('cuda:0')

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(self.dropout_p)

        self.embedding = nn.Embedding(self.output_size, 256)
        #self.gru = nn.GRUCell(684, 256)
        self.gru = GRUCell(1024, self.hidden_size, 0.0, 0.05)#Dynamic
        self.gru1 = GRUCell(256, self.hidden_size, 0.0, 0.05)
        self.out = nn.Linear(128, self.output_size)
        self.hidden = nn.Linear(self.hidden_size, 256)
        self.emb = nn.Linear(256, 128)
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.conv_et = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)
        self.conv_tan = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.hidden2 = nn.Linear(self.hidden_size, 128)
        self.emb2 = nn.Linear(256, 128)
        self.ua = nn.Linear(1024, 256)#Dynamic
        self.uf = nn.Linear(1, 256)
        self.v = nn.Linear(256, 1)
        self.wc = nn.Linear(1024, 128)#Dynamic
        self.bn = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self,input_a, hidden, v, encoder_outputs,bb,attention_sum,decoder_attention,dense_input,batch_size,h_mask,w_mask,gpu,epoch):

        # batch_gpu must be an int object
        batch_gpu = int(batch_size/len(gpu))
        #batch_gpu = 1
        #et_mask = torch.zeros(batch_gpu,dense_input,bb)
        et_mask = torch.zeros(batch_gpu,dense_input,bb).cuda()
        #et_mask = torch.zeros(batch_gpu,dense_input,bb).cuda().to(device)
        
        if et_mask.device == torch.device('cuda:0'):
            for i in range(batch_gpu):
                et_mask[i][:h_mask[i],:w_mask[i]]=1

        if et_mask.device == torch.device('cuda:1'):
            for i in range(batch_gpu):
                et_mask[i][:h_mask[i+1*batch_gpu],:w_mask[i+1*batch_gpu]]=1

        if et_mask.device == torch.device('cuda:3'):
            for i in range(batch_gpu):
                et_mask[i][:h_mask[i+2*batch_gpu],:w_mask[i+2*batch_gpu]]=1

        if et_mask.device == torch.device('cuda:2'):
            for i in range(batch_gpu):
                et_mask[i][:h_mask[i+3*batch_gpu],:w_mask[i+3*batch_gpu]]=1
        

        et_mask_4 = et_mask.unsqueeze(1)

        # embedding the word from 1 to 256(total 112 words)
        embedded = self.embedding(input_a).view(batch_gpu,256)
        embedded = self.dropout(embedded)
        hidden = hidden.view(batch_gpu,self.hidden_size)
        v = v.view(batch_gpu,3*self.hidden_size)

        st, vt = self.gru1(embedded,hidden,v)
        hidden1 = self.hidden(st)
        hidden1 = hidden1.view(batch_gpu,1,1,256)

        #att_gate = nn.GRUCell(hidden,en)
        # encoder_outputs from (batch,1024,height,width) => (batch,height,width,1024)
        encoder_outputs_trans = torch.transpose(encoder_outputs,1,2)
        encoder_outputs_trans = torch.transpose(encoder_outputs_trans,2,3)

        # encoder_outputs_trans (batch,height,width,1024) attention_sum_trans (batch,height,width,1) hidden1 (batch,1,1,256)
        decoder_attention = self.conv1(decoder_attention)
        attention_sum = attention_sum + decoder_attention
        attention_sum_trans = torch.transpose(attention_sum,1,2)
        attention_sum_trans = torch.transpose(attention_sum_trans,2,3)

        # encoder_outputs1 (batch,height,width,256) attention_sum1 (batch,height,width,256)
        #print('encoder_outputs_trans',encoder_outputs_trans.shape)
        
        #print('encoder_outputs_trans.shape', encoder_outputs_trans.shape)

        encoder_outputs1 = self.ua(encoder_outputs_trans)
        attention_sum1 = self.uf(attention_sum_trans)

        #print('hidden1.shape',hidden1.shape)
        #print('encoder_outputs1.shape',encoder_outputs1.shape)
        #print('attention_sum1.shape',attention_sum1.shape)


        et = hidden1 + encoder_outputs1 + attention_sum1
        et_trans = torch.transpose(et,2,3)
        et_trans = torch.transpose(et_trans,1,2)
        et_trans = self.conv_tan(et_trans)
        et_trans = et_trans*et_mask_4
        et_trans = self.bn1(et_trans)
        et_trans = torch.tanh(et_trans)
        et_trans = torch.transpose(et_trans,1,2)
        et_trans = torch.transpose(et_trans,2,3)

        #print('et_trans.shape',et_trans.shape)

        et = self.v(et_trans) #4,9,34,1
        et = et.squeeze(3)
        # et = torch.transpose(et,2,3)
        # et = torch.transpose(et,1,2)
        # et = self.conv_et(et)
        # et = et*et_mask_4
        # et = self.bn(et)
        # et = self.relu(et) 
        # et = et.squeeze(1)

        # et_div_all is attention alpha
        et_div_all = torch.zeros(batch_gpu,1,dense_input,bb)
        et_div_all = et_div_all.cuda()
        #et_div_all = et_div_all.cuda().to(device)
        #et_div_all = et_div_all

        et_exp = torch.exp(et)
        et_exp = et_exp*et_mask
        et_sum = torch.sum(et_exp,dim=1)
        et_sum = torch.sum(et_sum,dim=1)

        #print('et_sum.shape',et_sum.shape)

        for i in range(batch_gpu):
            et_div = et_exp[i]/(et_sum[i]+1e-8)
            et_div = et_div.unsqueeze(0)
            et_div_all[i] = et_div

        #print('encoder_outputs :',encoder_outputs.shape)
        #print('et_div_all :',et_div_all.shape)
        # ct is context vector (batch,128)
        
        ct = et_div_all*encoder_outputs
        
        #print("ct :",ct.shape)
        
        ct = ct.sum(dim=2)
        ct = ct.sum(dim=2)

        #print("ct after suming:",ct.shape)

        # the next hidden after gru
        # batch,hidden_size
        hidden_next_a, vt_next_a = self.gru(ct,st,vt)
        hidden_next = hidden_next_a.view(batch_gpu, 1, self.hidden_size)
        vt_next = vt_next_a.view(batch_gpu, 1, 3*self.hidden_size)

        # compute the output (batch,128)
        hidden2 = self.hidden2(hidden_next_a)
        embedded2 = self.emb2(embedded)
        ct2 = self.wc(ct)

        #output
        # output = F.log_softmax(self.out(hidden2+embedded2+ct2), dim=1)
        output = F.log_softmax(self.out(self.dropout(hidden2+embedded2+ct2)), dim=1)
        output = output.unsqueeze(1)

        return output, hidden_next, vt_next, et_div_all, attention_sum

    def initHidden(self,batch_size):
        result = Variable(torch.randn(batch_size, 1, self.hidden_size))
        #return result.cuda().to(device)
        return result.cuda()
        #return result
