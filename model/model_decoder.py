import importlib
import torch, os
from torch import nn
import math
from torch.nn.init import xavier_uniform_
import copy
from VideoX.SeqTrack.lib.utils.pos_embed import get_sinusoid_encoding_table
from VideoX.SeqTrack.lib.models.seqtrack.decoder import DecoderEmbeddings,TransformerDecoderLayer, TransformerDecoder
from VideoX.SeqTrack.lib.models.seqtrack.seqtrack import MLP
from torch import Tensor
from typing import Optional

from torch.nn import functional as F
from transformers import MambaConfig, MambaModel, GPT2Config, GPT2Model



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        self.embedding_1D = nn.Embedding(52, int(d_model))

    def forward(self, x):
        # fixed
        x = x + self.pe[:x.size(0), :]
        # learnable
        # x = x + self.embedding_1D(torch.arange(52).cuda()).unsqueeze(1).repeat(1,x.size(1),  1)
        return self.dropout(x)


class Mesh_TransformerDecoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Mesh_TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(int(d_model), nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

        self.fc_alpha1 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha2 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha3 = nn.Linear(d_model + d_model, d_model)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_alpha1.weight)
        nn.init.xavier_uniform_(self.fc_alpha2.weight)
        nn.init.xavier_uniform_(self.fc_alpha3.weight)
        nn.init.constant_(self.fc_alpha1.bias, 0)
        nn.init.constant_(self.fc_alpha2.bias, 0)
        nn.init.constant_(self.fc_alpha3.bias, 0)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        self_att_tgt = self.norm1(tgt + self._sa_block(tgt, tgt_mask, tgt_key_padding_mask))
        # # cross self-attention
        enc_att, att_weight = self._mha_block(self_att_tgt,
                                              memory, memory_mask,
                                              memory_key_padding_mask)

        x = self.norm2(self_att_tgt + enc_att)
        x = self.norm3(x + self._ff_block(x))
        return x + tgt
        # return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x, att_weight = self.multihead_attn(x, mem, mem,
                                            attn_mask=attn_mask,
                                            key_padding_mask=key_padding_mask,
                                            need_weights=True)
        return self.dropout2(x), att_weight

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class StackTransformer(nn.Module):
    r"""StackTransformer is a stack of N decoder layers

    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(StackTransformer, self).__init__()
        self.layers = torch.nn.modules.transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class DecoderTransformer(nn.Module):
    """
    Decoder with Transformer.
    """

    def __init__(self, decoder_type, embed_dim, vocab_size, max_lengths, word_vocab, n_head, n_layers, dropout):
        """
        :param n_head: the number of heads in Transformer
        :param n_layers: the number of layers of Transformer
        """
        super(DecoderTransformer, self).__init__()

        # n_layers = 1
        print("decoder_n_layers=", n_layers)

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.max_lengths = max_lengths
        self.word_vocab = word_vocab
        self.dropout = dropout
        # embedding layer
        self.vocab_embedding = nn.Embedding(vocab_size, self.embed_dim)  # vocaburaly embedding
        # Mamba
        self.decoder_type = decoder_type  # 'gpt' # 'mamba' or 'gpt' or 'transformer_decoder'
        print("decoder_type=", self.decoder_type)
        if self.decoder_type == 'mamba':
            # from model.mamba_block import CaMambaModel
            config_1 = MambaConfig(num_hidden_layers=1, hidden_size=self.embed_dim)
            self.Mamba = nn.ModuleList([])
            for i in range(n_layers):
                self.Mamba.append(MambaModel(config_1))
            # assert n_layers==1
        elif self.decoder_type == 'gpt':
            config_2 = GPT2Config(n_layer=1, n_embd=self.embed_dim)
            self.GPT = nn.ModuleList([])  # GPT2Model(config_2)
            for i in range(n_layers):
                self.GPT.append(GPT2Model(config_2))
        else:
            # Transformer layer
            decoder_layer = Mesh_TransformerDecoderLayer(self.embed_dim, n_head, dim_feedforward=self.embed_dim * 4,
                                                         dropout=self.dropout)
            self.transformer = StackTransformer(decoder_layer, n_layers)

        self.position_encoding = PositionalEncoding(self.embed_dim, max_len=max_lengths)

        # Linear layer to find scores over vocabulary
        self.wdc = nn.Linear(self.embed_dim, vocab_size)
        self.dropout = nn.Dropout(p=self.dropout)
        ###### SeqTrack
        num_patches = 512  # 49   # 两个特征图拼起来就是512，只用一个特征图就是256
        hidden_dim = 768
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        pos_embed = get_sinusoid_encoding_table(num_patches, self.pos_embed.shape[-1], cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        d_model = 768
        nhead = 8
        dim_feedforward = 1024
        activation = 'relu'
        normalize_before = False
        num_decoder_layers = 6
        return_intermediate_dec = False
        max_position_embeddings = 42
        self.embedding = DecoderEmbeddings(501, d_model, max_position_embeddings, dropout)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.body = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        self._reset_parameters()
        self.st_vocab_embed = MLP(self.embed_dim, self.embed_dim, vocab_size, 3)
        ######
        self.init_weights()  # initialize some layers with the uniform distribution

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence
        """
        self.vocab_embedding.weight.data.uniform_(-0.1, 0.1)

        self.wdc.bias.data.fill_(0)
        self.wdc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, encoded_captions, caption_lengths):
        """
        :param x1, x2: encoded images, a tensor of dimension (batch_size, channel, enc_image_size, enc_image_size)
        :param encoded_captions: a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: a tensor of dimension (batch_size)
        """
        batch, channel = x.size(0), x.size(1)
        x = x.view(batch, channel, -1).permute(1, 0, 2)
        # x = x.view(batch, channel, -1).permute(2, 0, 1)

        word_length = encoded_captions.size(1)      # 42(一行最多42个单词）
        mask = torch.triu(torch.ones(word_length, word_length) * float('-inf'), diagonal=1)
        mask = mask.cuda()
        tgt_pad_mask = (encoded_captions == self.word_vocab['<NULL>']) | (encoded_captions == self.word_vocab['<END>'])

        word_emb = self.vocab_embedding(encoded_captions)  # (batch, length, embed_dim)可以加layernorm和dropout（DecoderEmbeddings）
        word_emb = word_emb.transpose(1, 0)  # (length, batch, embed_dim)

        # word_emb = self.position_encoding(word_emb)  # (length, batch, embed_dim)
        ###########
        tgt = self.embedding(encoded_captions).permute(1, 0, 2)
        query_embed = self.embedding.position_embeddings.weight.unsqueeze(1)
        query_embed = query_embed.repeat(1, batch, 1)
        pos_embed = self.pos_embed.permute(1, 0, 2).expand(-1, batch, -1)
        pred = self.body(tgt, x, pos=pos_embed, query_pos=query_embed[:len(tgt)],tgt_mask=mask, memory_mask=None)
        pred = self.st_vocab_embed(pred.transpose(1, 2)).squeeze(0)

        # if self.decoder_type == 'mamba' or self.decoder_type == 'gpt':
        #     img_emb = x.permute(1, 0, 2)
        #     img_emb_len = img_emb.size(1)
        #     text = word_emb.permute(1, 0, 2)
        #     prefix = torch.cat((img_emb, text), dim=1)
        #     if self.decoder_type == 'mamba':
        #         # pred = self.Mamba(inputs_embeds=prefix).last_hidden_state #+ prefix
        #         # pred = pred[:, img_emb_len:, :]
        #         for i in range(len(self.Mamba)):
        #             prefix = self.Mamba[i](inputs_embeds=prefix).last_hidden_state  # + prefix
        #         pred = prefix[:, img_emb_len:, :]
        #     else:
        #         # pred = self.GPT(inputs_embeds=prefix).last_hidden_state #+ prefix
        #         # pred = pred[:, img_emb_len:, :]
        #         for i in range(len(self.GPT)):
        #             prefix = self.GPT[i](inputs_embeds=prefix).last_hidden_state  # + prefix
        #         pred = prefix[:, img_emb_len:, :]
        #
        #     pred = pred.permute(1, 0, 2)
        # else:
        #     # word_emb：42,1,768，x：49,1,768，mask：42,42，tgt_pad_mask：1,42
        #     pred = self.transformer(word_emb, x, tgt_mask=mask,
        #                             tgt_key_padding_mask=tgt_pad_mask)  # (length, batch, embed_dim)
        # pred = self.wdc(self.dropout(pred))  # (length42, batch, vocab_size501)
        # pred = pred.permute(1, 0, 2)

        # Sort input data by decreasing lengths
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoded_captions = encoded_captions[sort_ind]
        pred = pred[sort_ind]
        decode_lengths = (caption_lengths - 1).tolist()
        # encoded_caption = torch.cat((encoded_captions, torch.zeros([batch, 1], dtype = int).cuda()), dim=1)
        # decode_lengths = (caption_lengths).tolist()

        return pred, encoded_captions, decode_lengths, sort_ind

    def sample(self, x, k=1):
        """
        :param x: encoded images, a tensor of dimension (batch_size, channel, enc_image_size* enc_image_size)
        """
        batch, channel = x.size(0), x.size(1)
        x = x.view(batch, channel, -1).permute(1, 0, 2)  # (hw, batch_size, embed_dim)
        # x = x.view(batch, channel, -1).permute(2, 0, 1)  # (hw, batch_size, embed_dim)

        tgt = torch.zeros(batch, self.max_lengths).to(torch.int64).cuda()  # (batch_size, self.max_lengths)

        mask = torch.triu(torch.ones(self.max_lengths, self.max_lengths) * float('-inf'), diagonal=1)
        mask = mask.cuda()
        tgt[:, 0] = torch.LongTensor([self.word_vocab['<START>']] * batch).cuda()  # (batch_size, 1)
        seqs = torch.LongTensor([[self.word_vocab['<START>']]] * batch).cuda()  # (batch_size, 1)
        # Weight = torch.zeros(1, self.max_lengths, x.size(0)).cuda()
        ##########
        query_embed = self.embedding.position_embeddings.weight.unsqueeze(1)
        query_embed = query_embed.repeat(1, batch, 1)
        pos_embed = self.pos_embed.permute(1, 0, 2).expand(-1, batch, -1)

        for step in range(self.max_lengths):
            tgt_pad_mask = (tgt == self.word_vocab['<NULL>'])
            word_emb = self.vocab_embedding(tgt)
            word_emb = word_emb.transpose(1, 0)  # (length, batch, embed_dim)
            # word_emb = self.position_encoding(word_emb)
            #######
            emb = self.embedding(tgt).permute(1, 0, 2)
            pred = self.body(emb, x, pos=pos_embed, query_pos=query_embed[:len(emb)], tgt_mask=mask, memory_mask=None)
            pred = self.st_vocab_embed(pred.transpose(1, 2)).squeeze(0)

            # if self.decoder_type == 'mamba' or self.decoder_type == 'gpt':
            #     img_emb = x.permute(1, 0, 2)
            #     img_emb_len = img_emb.size(1)
            #     text = word_emb.permute(1, 0, 2)
            #     prefix = torch.cat((img_emb, text), dim=1)
            #     if self.decoder_type == 'mamba':
            #         # pred = self.Mamba(inputs_embeds=prefix).last_hidden_state #+ prefix
            #         # pred = pred[:, img_emb_len:, :]
            #         for i in range(len(self.Mamba)):
            #             prefix = self.Mamba[i](inputs_embeds=prefix).last_hidden_state  # + prefix
            #         pred = prefix[:, img_emb_len:, :]
            #     else:
            #         # pred = self.GPT(inputs_embeds=prefix).last_hidden_state #+ prefix
            #         # pred = pred[:, img_emb_len:, :]
            #         for i in range(len(self.GPT)):
            #             prefix = self.GPT[i](inputs_embeds=prefix).last_hidden_state  # + prefix
            #         pred = prefix[:, img_emb_len:, :]
            #     pred = pred.permute(1, 0, 2)
            # else:
            #     pred = self.transformer(word_emb, x, tgt_mask=mask, tgt_key_padding_mask=tgt_pad_mask)
            #
            # pred = self.wdc(self.dropout(pred))  # (length, batch, vocab_size)
            # scores = pred.permute(1, 0, 2)  # (batch, length, vocab_size)
            scores = pred  # (batch, length, vocab_size)
            scores = scores[:, step, :].squeeze(1)  # [batch, 1, vocab_size] -> [batch, vocab_size]
            predicted_id = torch.argmax(scores, axis=-1)
            seqs = torch.cat([seqs, predicted_id.unsqueeze(1)], dim=-1)
            # Weight = torch.cat([Weight, weight], dim = 0)
            if predicted_id == self.word_vocab['<END>']:
                break
            if step < (self.max_lengths - 1):  # except <END> node
                tgt[:, step + 1] = predicted_id
        seqs = seqs.squeeze(0)
        seqs = seqs.tolist()

        # feature=x.clone()
        # Weight1=Weight.clone()
        return seqs

    def sample_beam(self, x, k=1):
        """
        :param x: encoded images, a tensor of dimension (batch_size, channel, enc_image_size*enc_image_size)
        :param max_lengths: maximum length of the generated captions
        :param k: beam_size
        """
        batch, channel, L = x.shape
        assert batch == 1, "batch size must be 1"
        x = x.view(batch, channel, -1).unsqueeze(0).expand(k, -1, -1, -1).reshape(batch * k, channel, L).permute(2, 0,
                                                                                                                 1)  # (h*w, batch, embed_dim)

        tgt = torch.zeros(k * batch, self.max_lengths).to(torch.int64).cuda()  # (batch_size*k, self.max_lengths)

        mask = (torch.triu(torch.ones(self.max_lengths, self.max_lengths)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.cuda()
        tgt[:, 0] = torch.LongTensor([self.word_vocab['<START>']] * batch * k).cuda()  # (batch_size*k, 1)
        seqs = torch.LongTensor([[self.word_vocab['<START>']]] * batch * k).cuda()
        top_k_scores = torch.zeros(k * batch, 1).cuda()
        complete_seqs = []
        complete_seqs_scores = []
        for step in range(self.max_lengths):
            word_emb = self.vocab_embedding(tgt)
            word_emb = word_emb.transpose(1, 0)
            word_emb = self.position_encoding(word_emb)
            if self.decoder_type == 'mamba' or self.decoder_type == 'gpt':
                img_emb = x.permute(1, 0, 2)
                img_emb_len = img_emb.size(1)
                text = word_emb.permute(1, 0, 2)
                prefix = torch.cat((img_emb, text), dim=1)
                if self.decoder_type == 'mamba':
                    # pred = self.Mamba(inputs_embeds=prefix).last_hidden_state #+ prefix
                    # pred = pred[:, img_emb_len:, :]
                    for i in range(len(self.Mamba)):
                        prefix = self.Mamba[i](inputs_embeds=prefix).last_hidden_state  # + prefix
                    pred = prefix[:, img_emb_len:, :]
                else:
                    # pred = self.GPT(inputs_embeds=prefix).last_hidden_state #+ prefix
                    # pred = pred[:, img_emb_len:, :]
                    for i in range(len(self.GPT)):
                        prefix = self.GPT[i](inputs_embeds=prefix).last_hidden_state  # + prefix
                    pred = prefix[:, img_emb_len:, :]
                pred = pred.permute(1, 0, 2)
            else:
                pred = self.transformer(word_emb, x, tgt_mask=mask)
            pred = self.wdc(self.dropout(pred))  # (length, batch, vocab_size)
            scores = pred.permute(1, 0, 2)  # (batch, length, vocab_size)
            scores = scores[:, step, :].squeeze(1)  # [batch, 1, vocab_size] -> [batch, vocab_size]
            scores = F.log_softmax(scores, dim=1)
            scores = top_k_scores.expand_as(scores) + scores
            if step == 0:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            # prev_word_inds = top_k_words // vocab_size  # (s)
            prev_word_inds = torch.div(top_k_words, self.vocab_size, rounding_mode='floor')
            next_word_inds = top_k_words % self.vocab_size  # (s)
            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != self.word_vocab['<END>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            x = x[:, prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            tgt = tgt[incomplete_inds]
            if step < self.max_lengths - 1:
                tgt[:, :step + 2] = seqs

        if complete_seqs == []:
            complete_seqs.extend(seqs[incomplete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[incomplete_inds])
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        return seq

    def fine_tune(self, fine_tune=True):
        for p in self.parameters():
            p.requires_grad = fine_tune
