# coding= utf-8
import torch
from transformers import BertTokenizer, T5Config, T5ForConditionalGeneration
from functools import partial
import torch.nn as nn


class T5GenerationArgs:
    def __init__(self):
        # Prefixtuning Args
        self.num_token = 250
        self.prefix_dropout = 0.2
        self.mid_dim = 768  # choose the BertTokenizer Embedding 768
        self.using_encoder_key_values = True
        self.using_decoder_key_values = True


class PrefixtuningTemplate(nn.Module):
    def __init__(self, args, config, plm):
        super().__init__()
        self.args = args
        self.config = config
        self.plm = plm
        if isinstance(config, T5Config):
            self.n_layer = config.num_layers
            self.d_model = config.d_model
            self.n_head = config.num_heads
            self.n_embed = self.d_model // self.n_head
            assert self.n_embed * self.n_head == self.d_model
        else:
            raise NotImplementedError
        self.num_token = args.num_token
        self.input_tokens = nn.Parameter(torch.arange(self.num_token).long(), requires_grad=False)
        self.using_encoder_key_values = args.using_encoder_key_values
        self.using_decoder_key_values = args.using_decoder_key_values
        if args.using_encoder_key_values:
            self.wte = nn.Embedding(self.num_token, self.d_model)
            self.mlp = nn.Sequential(
                nn.Linear(self.d_model, args.mid_dim),
                nn.Tanh(),
                nn.Linear(args.mid_dim, self.n_layer * 2 * self.d_model)
            )
        if args.using_decoder_key_values:
            self.decoder_wte = nn.Embedding(self.num_token, self.d_model)
            self.decoder_mlp = nn.Sequential(
                nn.Linear(self.d_model, args.mid_dim),
                nn.Tanh(),
                nn.Linear(args.mid_dim, self.n_layer * 2 * self.d_model)
            )
        self.past_key_values = [None, None]
        self.dropout = nn.Dropout(args.prefix_dropout)
        self.plm_modified = False

    def forward(self, **kwargs):
        self.process_batch(**kwargs)
        return self.plm(**kwargs)

    def generate(self, **kwargs):
        self.process_batch(**kwargs)
        return self.plm.generate(**kwargs)

    def process_batch(self, **batch):
        bsz = batch['input_ids'].shape[0]
        if self.using_encoder_key_values:
            # l * e
            input_embeds = self.wte(self.input_tokens)
            # bsz * l * e
            past_key_value = self.mlp(input_embeds)
        past_key_value = past_key_value.view(1, self.num_token, self.n_layer * 2, self.n_head, self.n_embed)
        past_key_value = self.dropout(past_key_value)
        past_key_value = past_key_value.permute([2, 0, 3, 1, 4]).split(2)
        self.past_key_values[0] = past_key_value
        if self.using_decoder_key_values:
            # l * e
            input_embeds = self.wte(self.input_tokens)
            # bsz * l * e
            past_key_value = self.mlp(input_embeds).unsqueeze(0)
        past_key_value = past_key_value.view(1, self.num_token, self.n_layer * 2, self.n_head, self.n_embed)
        past_key_value = self.dropout(past_key_value)
        past_key_value = past_key_value.permute([2, 0, 3, 1, 4]).split(2)
        self.past_key_values[1] = past_key_value

    def modify_plm(self):
        if self.plm_modified:
            return
        model = self.plm
        args = self.args

        if isinstance(model, T5ForConditionalGeneration):
            if args.using_encoder_key_values:
                backup_encoder_forward_functions = []
                for i, layer_module in enumerate(model.encoder.block):
                    backup_encoder_forward_functions.append(layer_module.layer[0].forward)

                    def modified_encoder_forward(*args, **kwargs):
                        layer_id = kwargs.pop('layer_id')
                        batch_size = args[0].shape[0]
                        device = args[0].device
                        if kwargs['past_key_value'] is None:
                            kwargs['past_key_value'] = expand_to_batchsize_for_layer(self.past_key_values[0],
                                                                                     batch_size, layer_id).to(device)
                        if kwargs['attention_mask'] is not None:
                            am = kwargs[
                                'attention_mask']  # Should check the format of the attention_mask when moving to a new plm.
                            kwargs['attention_mask'] = torch.cat(
                                [-torch.zeros((*am.shape[:-1], self.num_token), dtype=am.dtype, device=am.device), am],
                                dim=-1)
                        return backup_encoder_forward_functions[layer_id](*args, **kwargs)

                    layer_module.layer[0].forward = partial(modified_encoder_forward, layer_id=i)
            if self.using_decoder_key_values:
                backup_decoder_self_attn_forward_functions = []
                for i, layer_module in enumerate(model.decoder.block):
                    backup_decoder_self_attn_forward_functions.append(layer_module.layer[0].forward)

                    def modified_decoder_self_attn_forward(*args, **kwargs):
                        batch_size = args[0].shape[0]
                        layer_id = kwargs.pop('layer_id')
                        device = args[0].device
                        if kwargs['past_key_value'] is None:
                            kwargs['past_key_value'] = expand_to_batchsize_for_layer(self.past_key_values[1],
                                                                                     batch_size, layer_id).to(device)
                        if kwargs['past_key_value'][0].size(-2) + args[0].size(-2) == kwargs['attention_mask'].size(-1):
                            pass
                        elif kwargs['past_key_value'][0].size(-2) + args[0].size(-2) == kwargs['attention_mask'].size(
                                -1) + self.num_token:
                            am = kwargs['attention_mask']
                            kwargs['attention_mask'] = torch.cat(
                                [torch.zeros((*am.shape[:-1], self.num_token), dtype=am.dtype, device=am.device), am],
                                dim=-1)
                        else:
                            raise RuntimeError("Size not match: past length: {}, inputlength:{},\
                                attention mask length {}".format(kwargs['past_key_value'][0].size(-2),
                                                                 args[0].size(-2), kwargs['attention_mask'].size(-1)))

                        return backup_decoder_self_attn_forward_functions[layer_id](*args, **kwargs)

                    layer_module.layer[0].forward = partial(modified_decoder_self_attn_forward, layer_id=i)
        else:
            raise NotImplementedError
        for p in self.plm.parameters():
            p.requires_grad = False
        self.plm_modified = True


def expand_to_batchsize_for_layer(tup, batch_size, layer_id):
    return tup[layer_id].expand(-1, batch_size, -1, -1, -1)
