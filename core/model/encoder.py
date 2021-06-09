import os
from posixpath import join
from sentence_transformers import SentenceTransformer
import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .modules import upBlock


def batch_to_cuda(batch):
    for key in batch:
        if isinstance(batch[key],torch.Tensor):
            batch[key] = batch[key].cuda()
    return batch

def words_pooling(words_embs, sum_mask, mode='MEAN'):
    if mode == 'MEAN':
        sum_embeddings = torch.sum(words_embs, 1)
        sent_emb = sum_embeddings / sum_mask 
    else:
        raise NotImplementedError()

    return sent_emb

class DECODER(nn.Module):
    def __init__(self, ndf):
        super(DECODER, self).__init__()
        self.ndf = ndf 
        self.block0 = upBlock(ndf*16,ndf*8)
        self.block1 = upBlock(ndf*8,ndf*4)
        self.block2 = upBlock(ndf*4,ndf*2)
        self.block3 = upBlock(ndf*2,ndf*1)
        self.conv = nn.Conv2d(ndf,3,3,1,1)

    def forward(self, out):
        out = self.block0(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.conv(out)
        return out



class SBERT_FT_ENCODER(nn.Module):
    def __init__(self, cfg):
        super(SBERT_FT_ENCODER, self).__init__()
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.bert_norm = cfg.TEXT.BERT_NORM
        self.pooling_mode = cfg.TEXT.POOLING_MODE
        self.max_seq_length = cfg.TEXT.MAX_LENGTH

        joint_ft = cfg.TEXT.JOINT_FT
        text_dim = cfg.TEXT.EMBEDDING_DIM
        ft_dim = cfg.TEXT.FT_DIM

        self._define_modules(joint_ft=joint_ft, ft_dim=ft_dim, text_dim = text_dim)

    def _define_modules(self, joint_ft = False, ft_dim = 256, text_dim = 256):
        self.bert = SentenceTransformer('stsb-roberta-base')
        for param in self.bert.parameters():
            param.requires_grad = False 
        self.bert.eval()
        self.bert.max_seq_length = self.max_seq_length

        self.proj_sent = nn.Linear(text_dim, ft_dim) if joint_ft else nn.Identity()
        self.proj_word = nn.Conv1d(text_dim, ft_dim, 1, 1, 0) if joint_ft else nn.Identity()
        
        print('SBERT ENCODER')

    #@torch.no_grad()
    def forward(self, sents, sent_lens):
        
        with torch.no_grad():
            self.bert.eval()
            sorted_sent_lens, sorted_idx = sent_lens.sort(descending=True)
            sorted_sents = [sents[idx] for idx in sorted_idx]

            features = self.bert.tokenize(sorted_sents)
            features = batch_to_cuda(features)

            output_features = self.bert(features)

            embeddings = output_features['token_embeddings']

            attn_mask = output_features['attention_mask']
            attn_mask_expanded = attn_mask.unsqueeze(-1).expand(embeddings.size()).float()

            embeddings = embeddings * attn_mask_expanded

            words_embs = embeddings[sorted_idx.argsort()]
            attn_mask = attn_mask[sorted_idx.argsort()]

            sum_mask = attn_mask.unsqueeze(-1).sum(1)
            mask = (attn_mask == 0)

            sent_embs = words_pooling(words_embs = words_embs, sum_mask = sum_mask, mode = self.pooling_mode)

            if self.bert_norm:
                sent_embs = F.normalize(sent_embs, p=2, dim=1)

            words_embs = words_embs.transpose(1,2) # [bs, text_dim, T]

        psent_embs = self.proj_sent(sent_embs)
        pwords_embs = self.proj_word(words_embs)

        return pwords_embs, psent_embs, mask, sent_embs

class SBERT_ENCODER(nn.Module):
    def __init__(self, cfg):
        super(SBERT_ENCODER, self).__init__()
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.bert_norm = cfg.TEXT.BERT_NORM
        self.pooling_mode = cfg.TEXT.POOLING_MODE
        self.max_seq_length = cfg.TEXT.MAX_LENGTH

        joint_ft = cfg.TEXT.JOINT_FT
        text_dim = cfg.TEXT.EMBEDDING_DIM
        nef = cfg.TRAIN.NEF

        self._define_modules(joint_ft=joint_ft, text_dim=text_dim, nef = nef)


    def _define_modules(self, joint_ft = False, text_dim = 256, nef = 256):
        self.bert = SentenceTransformer('stsb-roberta-base')
        for param in self.bert.parameters():
            param.requires_grad = False 
        self.bert.eval()
        self.bert.max_seq_length = self.max_seq_length

        self.proj_sent = nn.Linear(text_dim, nef) if joint_ft else nn.Identity()
        self.proj_word = nn.Conv1d(text_dim, nef, 1, 1, 0) if joint_ft else nn.Identity()
        
        print('SBERT ENCODER')

    @torch.no_grad()
    def forward(self, sents, sent_lens):
        sorted_sent_lens, sorted_idx = sent_lens.sort(descending=True)
        sorted_sents = [sents[idx] for idx in sorted_idx]

        features = self.bert.tokenize(sorted_sents)
        features = batch_to_cuda(features)

        output_features = self.bert(features)

        embeddings = output_features['token_embeddings']

        attn_mask = output_features['attention_mask']
        attn_mask_expanded = attn_mask.unsqueeze(-1).expand(embeddings.size()).float()

        embeddings = embeddings * attn_mask_expanded

        words_embs = embeddings[sorted_idx.argsort()]
        attn_mask = attn_mask[sorted_idx.argsort()]

        sum_mask = attn_mask.unsqueeze(-1).sum(1)
        mask = (attn_mask == 0)

        sent_embs = words_pooling(words_embs = words_embs, sum_mask = sum_mask, mode = self.pooling_mode)

        if self.bert_norm:
            sent_embs = F.normalize(sent_embs, p=2, dim=1)

        words_embs = words_embs.transpose(1,2) # [bs, text_dim, T]

        sent_embs = self.proj_sent(sent_embs)
        words_embs = self.proj_word(words_embs)

        return words_embs, sent_embs, mask


class RNN_ENCODER(nn.Module):
    def __init__(self,cfg):
        super(RNN_ENCODER,self).__init__()
        self.n_steps = cfg.TEXT.MAX_LENGTH
        self.ntoken = cfg.TEXT.VOCA_SIZE
        self.ninput = 300
        self.drop_prob = 0.5
        self.nlayers = 1
        self.bidirectional = True 
        self.rnn_type = cfg.TEXT.RNN_TYPE
        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.nhidden = cfg.TEXT.EMBEDDING_DIM // self.num_directions

        self._define_modules()
        
        #self._init_weights()

    def _define_modules(self):
        self.encoder = nn.Embedding(self.ntoken,self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.ninput,self.nhidden,self.nlayers,batch_first=True,
                                dropout=self.drop_prob,
                                bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput,self.nhidden,self.nlayers,batch_first=True,
                                dropout=self.drop_prob,
                                bidirectional = self.bidirectional)
        else:
            raise NotImplementedError()

        print('Use rnn encoder')

    def _init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange,initrange)
    
    def _init_hidden(self,batch_size):
        weight = next(self.parameters()).data 
        if self.rnn_type =='LSTM':
            return(nn.Parameter(weight.new(self.nlayers*self.num_directions,batch_size,self.nhidden).zero_()),
                    nn.Parameter(weight.new(self.nlayers*self.num_directions,batch_size,self.nhidden).zero_()))
        else:
            return nn.Parameter(weight.new(self.nlayers * self.num_directions,batch_size,self.nhidden).zero_())

    def forward(self,caps,cap_lens, **kwargs):

        caps = caps.cuda()
        cap_lens = cap_lens.cuda()

        sorted_cap_lens, sorted_idx = cap_lens.sort(descending=True)
        sorted_caps = caps[sorted_idx]
        sorted_cap_lens = sorted_cap_lens.tolist()

        batch_size = sorted_caps.size(0)
        hiddens = self._init_hidden(batch_size)

        sorted_embs = self.drop(self.encoder(sorted_caps))

        sorted_embs = pack_padded_sequence(sorted_embs, sorted_cap_lens, batch_first = True)
        sorted_outputs, sorted_hiddens = self.rnn(sorted_embs, hiddens)

        sorted_outputs = pad_packed_sequence(sorted_outputs, batch_first= True, total_length = self.n_steps)[0]
        #sorted_outputs = pad_packed_sequence(sorted_outputs, batch_first= True)[0]

        sorted_words_embs = sorted_outputs.transpose(1,2)

        if self.rnn_type == 'LSTM':
            sorted_sent_embs = sorted_hiddens[0].transpose(0,1).contiguous()
        else:
            sorted_sent_embs = sorted_hiddens.transpose(0,1).contiguous()
        
        sorted_sent_embs = sorted_sent_embs.view(-1,self.nhidden * self.num_directions)

        mask = (caps == 0) 

        words_embs = sorted_words_embs[sorted_idx.argsort()]
        sent_embs = sorted_sent_embs[sorted_idx.argsort()]

        return words_embs, sent_embs, mask