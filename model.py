import torch
import torch.nn as nn
import torch.nn.functional as F

"""main architecture for hierarchical gated recurrent unit with masked residual attention mechanism"""


class Decoder_GRU_masked_residual_attention(nn.Module):
    def __init__(self, pretrained_layers, in_feature=840, decoder_embedding_size=1024, num_layers=8):
        super(Decoder_GRU_masked_residual_attention, self).__init__()

        self.pretrained = pretrained_layers

        N = 4
        self.d_model = 840
        self.fc = nn.Linear(in_feature, self.d_model)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.layers = nn.ModuleList([nn.GRU(self.d_model, self.d_model, num_layers=num_layers,
                                            batch_first=True, dropout=0.2, bidirectional=False) for _ in range(N)])
        self.fc1 = nn.Linear(self.d_model, decoder_embedding_size)

        self.activation = nn.ReLU()
        self.out_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.residual_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.dropout1 = nn.Dropout(0.3)

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted):
        input = F.relu(self.fc(input_embeddings_batch))
        input = self.dropout(input)
        input = self.layer_norm(input)
        out = input

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        extended_attention_mask = input_masks_batch.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        attention_mask = extended_attention_mask.squeeze(1)
        attention_mask_ = attention_mask.permute(0, 2, 1)

        i = -1
        for l in self.layers:
            residual = out
            out, _ = l(out)

            i += 1
            if i == 0:
                continue
            if i == 3:
                continue

            fusion_att = F.relu(torch.matmul(input_embeddings_batch, residual.transpose(-1, -2)))
            fusion_att = fusion_att + attention_mask + attention_mask_
            fusion_att = nn.Softmax(dim=-1)(fusion_att)
            fusion_att = self.dropout1(fusion_att)

            fusion_data = F.relu(torch.matmul(fusion_att, out))
            out = fusion_data + out

        encoded_embedding = self.fc1(out)
        out = self.pretrained(inputs_embeds=encoded_embedding, attention_mask=input_masks_batch, return_dict=True,
                              labels=target_ids_batch_converted)
        return out


""" crippled decoding model w/o HGRU-MRAM"""


class Decoder_Naive(nn.Module):
    def __init__(self, pretrained_layers, in_feature=840, decoder_embedding_size=1024, num_layers=8):
        super(Decoder_Naive, self).__init__()
        self.pretrained = pretrained_layers
        self.fc1 = nn.Linear(in_feature, decoder_embedding_size)

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted):
        """input_mask: 1 is not masked, 0 is masked"""
        """input_masks_invert: 1 is masked, 0 is not masked"""
        encoded_embedding = F.relu(self.fc1(input_embeddings_batch))
        out = self.pretrained(inputs_embeds=encoded_embedding, attention_mask=input_masks_batch, return_dict=True,
                              labels=target_ids_batch_converted)
        return out
