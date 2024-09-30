from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn

def model_choose(model_name):
    # only support the following models
    if  "Llama" or "Mistral" in model_name: 
        model, tokenizer = load_model(model_name)
        num_layers=model.config.num_hidden_layers
        wrapper = Llamawrapper(model, tokenizer,num_layers)
    
    if  "gpt2" in model_name:
        model, tokenizer = load_model(model_name)
        num_layers=model.config.n_layer
        wrapper = GPT2Wrapper(model, tokenizer,num_layers)

    if  model_name =="gpt-j-6b" :   
        model, tokenizer = load_model(model_name)
        num_layers=model.config.n_layer
        wrapper = GPTJWrapper(model, tokenizer,num_layers)

    return wrapper

def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device

def load_model(model_name):
    device= get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16,attn_implementation="eager").to(device)
    return model, tokenizer

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class BloomIdentityLayer(nn.Module):
    def __init__(self):
        super(BloomIdentityLayer, self).__init__()
    def forward(self, x, y):
        return x+y #bloom expects the MLP to handle the residual connection


class ModelWrapper(nn.Module):

    def __init__(self, model, tokenizer,num_layers):
        super().__init__()
        self.model = model.eval()
        self.model.activations_ = {}
        self.model.activation_logitslens_ = []
        self.tokenizer = tokenizer
        self.device = get_device()
        self.num_layers = num_layers
        self.hooks  = []
        self.layer_pasts = {}
        

    def tokenize(self, s):
        return self.tokenizer.encode(s, return_tensors='pt').to(self.device)

    def list_decode(self, inpids):
        return [self.tokenizer.decode(s) for s in inpids]

    def layer_decode(self, hidden_states):
        raise Exception("Layer decode has to be implemented!")
    
    def get_layers_attentions(self,tokens, **kwargs):
        outputs = self.model(input_ids=tokens,  output_attentions=True, **kwargs)
        attentions=outputs.attentions
        return attentions

    def get_layers(self, tokens, **kwargs):
        outputs = self.model(input_ids=tokens, output_hidden_states=True, **kwargs)
        hidden_states, true_logits = outputs.hidden_states, outputs.logits
        logits = self.layer_decode(hidden_states)
        #logits[-1] = true_logits.squeeze(0)[-1].unsqueeze(-1) #we used to just replace the last logits because we were applying ln_f twice
        return torch.stack(logits).squeeze(-1)#, true_logits.squeeze(0)

    def get_layers_logitslens(self, tokens, **kwargs):
        outputs = self.model(input_ids=tokens, output_hidden_states=True, **kwargs)
        hidden_states=outputs.hidden_states
        hidden_states_final_layer=hidden_states[-1][0, -1, :]
        final_logits=torch.matmul(self.model.lm_head.weight.detach(), hidden_states_final_layer)
        logits = self.layer_decode_logitslens(self.model.activation_logitslens_)
        logits.append(final_logits)
        #logits[-1] = true_logits.squeeze(0)[-1].unsqueeze(-1) #we used to just replace the last logits because we were applying ln_f twice
        return torch.stack(logits)#, true_logits.squeeze(0)
    
    def get_logit(self, tokens, **kwargs):
        outputs = self.model(input_ids=tokens, output_hidden_states=True, **kwargs)
        hidden_states=outputs.hidden_states
        hidden_states_final_layer=hidden_states[-1][0, -1, :]
        logit =torch.matmul(self.model.lm_head.weight.detach(), hidden_states_final_layer)
        return logit

    def get_layers_logitslens_label_position(self, tokens, label_index, **kwargs):
        outputs = self.model(input_ids=tokens, output_hidden_states=True, **kwargs)
        hidden_states=outputs.hidden_states
        hidden_states_final_layer=hidden_states[-1][0, label_index, :]
        final_logits=torch.matmul(self.model.lm_head.weight.detach(), hidden_states_final_layer)
        logits = self.layer_decode_logitslens(self.model.activation_logitslens_)
        logits.append(final_logits)
        #logits[-1] = true_logits.squeeze(0)[-1].unsqueeze(-1) #we used to just replace the last logits because we were applying ln_f twice
        return torch.stack(logits)#, true_logits.squeeze(0)

    def get_layers_w_attns(self, tokens, **kwargs):
        outputs = self.model(input_ids=tokens, output_hidden_states=True, output_attentions=True, **kwargs)
        hidden_states, true_logits = outputs.hidden_states, outputs.logits
        logits = self.layer_decode(hidden_states)
        #logits[-1] = true_logits.squeeze(0)[-1].unsqueeze(-1)
        return torch.stack(logits).squeeze(-1), outputs.attentions#, true_logits.squeeze(0)

    def rr_per_layer(self, logits, answer, debug=False):
        #reciprocal rank of the answer at each layer
        answer_id = self.tokenizer.encode(answer)[0]
        if debug:
            print("Answer id", answer_id, answer)

        rrs = []
        for i,layer in enumerate(logits):
            soft = F.softmax(layer,dim=-1)
            sorted_probs = soft.argsort(descending=True)
            rank = float(np.where(sorted_probs.cpu().numpy()==answer_id)[0][0])
            rrs.append(1/(rank+1))

        return np.array(rrs)
    
    def logprob_per_layer(self, logits, answer, debug=False):
        #logitprob of the answer at each layer
        answer_id = self.tokenizer.encode(answer)[0]
        logprobs = []
        for i,layer in enumerate(logits):
            soft = F.softmax(layer,dim=-1)
            logprobs.append(torch.log(soft[answer_id]).item())

        return np.array(logprobs)

    def prob_of_answer(self, logits, answer, debug=False):
        answer_id = self.tokenizer.encode(answer)[0]
        if debug:
            print("Answer id", answer_id, answer)
        answer_probs = []
        first_top = -1
        mrrs = []
        for i,layer in enumerate(logits):
            soft = F.softmax(layer,dim=-1)
            answer_prob = soft[answer_id].item()
            sorted_probs = soft.argsort(descending=True)
            if debug:
                print(f"{i}::", answer_prob)
            answer_probs.append(answer_prob)
        #is_top_at_end = sorted_probs[0] == answer_id
        return np.array(answer_probs)

    def print_top(self, logits, k=30):
        for i,layer in enumerate(logits):
            print(f"{i}", self.tokenizer.decode(F.softmax(layer,dim=-1).argsort(descending=True)[:k]) )

    def print_top_last_layer(self, logits,predictions):
        layer=logits[-1]
        predictions.append( self.tokenizer.decode(F.softmax(layer,dim=-1).argsort(descending=True)[:1]) )

    def topk_per_layer(self, logits, k=10):
        topk = []
        for i,layer in enumerate(logits):
            topk.append([self.tokenizer.decode(s) for s in F.softmax(layer,dim=-1).argsort(descending=True)[:k]])
        return topk

    def get_activation(self, name):
        #https://github.com/mega002/lm-debugger/blob/01ba7413b3c671af08bc1c315e9cc64f9f4abee2/flask_server/req_res_oop.py#L57
        def hook(module, input, output):
            if "in_sln" in name:
                num_tokens = list(input[0].size())[1]
                self.model.activations_[name] = input[0][:, num_tokens - 1].detach()
            elif "mlp" in name or "attn" in name or "m_coef" in name:
                if "attn" in name:
                    num_tokens = list(output[0].size())[1]
                    self.model.activations_[name] = output[0][:, num_tokens - 1].detach()
                    self.model.activations_['in_'+name] = input[0][:, num_tokens - 1].detach()
                elif "mlp" in name:
                    num_tokens = list(output[0].size())[0]  # [num_tokens, 3072] for values;
                    self.model.activations_[name] = output[0][num_tokens - 1].detach()
                elif "m_coef" in name:
                    num_tokens = list(input[0].size())[1]  # (batch, sequence, hidden_state)
                    self.model.activations_[name] = input[0][:, num_tokens - 1].detach()
            elif "residual" in name or "embedding" in name:
                num_tokens = list(input[0].size())[1]  # (batch, sequence, hidden_state)
                if name == "layer_residual_" + str(self.num_layers-1):
                    self.model.activations_[name] = self.model.activations_[
                                                        "intermediate_residual_" + str(final_layer)] + \
                                                    self.model.activations_["mlp_" + str(final_layer)]

                else:
                    if 'out' in name:
                        self.model.activations_[name] = output[0][num_tokens-1].detach()
                    else:
                        self.model.activations_[name] = input[0][:,
                                                            num_tokens - 1].detach()

        return hook

    def get_activation_logitslens(self):
        #https://github.com/mega002/lm-debugger/blob/01ba7413b3c671af08bc1c315e9cc64f9f4abee2/flask_server/req_res_oop.py#L57
        def hook(module, input, output):
            
            self.model.activation_logitslens_.append(input[0] [0, -1].detach())
            

        return hook
    
    def get_activation_logitslens_gptj(self):
        #https://github.com/mega002/lm-debugger/blob/01ba7413b3c671af08bc1c315e9cc64f9f4abee2/flask_server/req_res_oop.py#L57
        def hook(module, input, output):
            
            self.model.activation_logitslens_.append(output[0] [0, -1].detach())
            

        return hook
    
    
    def get_activation_logitslens_label_position(self,label_index):
        #https://github.com/mega002/lm-debugger/blob/01ba7413b3c671af08bc1c315e9cc64f9f4abee2/flask_server/req_res_oop.py#L57
        def hook(module, input, output):
            
            self.model.activation_logitslens_.append(input[0] [0, label_index].detach())
            

        return hook

    def reset_activations(self):
        self.model.activations_ = {}

        
class GPTJWrapper(ModelWrapper):
    def add_hooks_logitslens(self):
        for i in range(self.num_layers):
            #intermediate residual between
            #print('saving hook') 
            self.model.transformer.h[i].ln_1.register_forward_hook(self.get_activation_logitslens())
            
            self.model.transformer.h[i].attn.register_forward_hook(self.get_activation_logitslens_gptj())

    def get_layers_logitslens(self, tokens, **kwargs):
        outputs = self.model(input_ids=tokens, output_hidden_states=True, **kwargs)
        hidden_states=outputs.hidden_states
        hidden_states_final_layer=hidden_states[-1][0, -1, :]
        final_logits=torch.matmul(self.model.lm_head.weight.detach(), hidden_states_final_layer)
        for i in range(1,len(self.model.activation_logitslens_),2):
            self.model.activation_logitslens_[i] = self.model.activation_logitslens_[i] + self.model.activation_logitslens_[i-1]

        logits = self.layer_decode_logitslens(self.model.activation_logitslens_)
        logits.append(final_logits)
        #logits[-1] = true_logits.squeeze(0)[-1].unsqueeze(-1) #we used to just replace the last logits because we were applying ln_f twice
        return torch.stack(logits)#, true_logits.squeeze(0)
    
    def layer_decode_logitslens(self,  activation_logitslens_):
        logits = []
        for i,h in enumerate(activation_logitslens_):
            normed = self.model.transformer.ln_f(h)
            l = torch.matmul(self.model.lm_head.weight.detach(), normed)
            logits.append(l)
        return logits

class Llamawrapper(ModelWrapper):
    def add_hooks_logitslens(self):
        for i in range(self.num_layers):
            #intermediate residual between
            #print('saving hook') 
            self.model.model.layers[i].input_layernorm.register_forward_hook(self.get_activation_logitslens())
            
            self.model.model.layers[i].post_attention_layernorm.register_forward_hook(self.get_activation_logitslens())
        
            #print(self.model.activations_)
    def repeat_kv(self,hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def layer_decode_logitslens(self,  activation_logitslens_):
        logits = []
        for i,h in enumerate(activation_logitslens_):
            normed = self.model.model.norm(h)
            l = torch.matmul(self.model.lm_head.weight.detach(), normed)
            logits.append(l)
        return logits
    
    def layer_decode_hidden_state(self,hidden_states, attention_weights,token_index):
        # Create an empty list to store the hidden states of the label tokens for each layer.
        logits = []

        for layer_index in range(self.num_layers):     
            # Get the block, attention, and normalization layers of the model.
            block = self.model.model.layers[layer_index]
            attn = block.self_attn
            input_layernorm = block.input_layernorm
            ln_f = self.model.model.norm 
            
            # Obtain the hidden states of the layer.
            hidden_states_layer = hidden_states[layer_index]

            # Get the value vectors.
            normed = input_layernorm (hidden_states_layer)
            value = attn.v_proj(normed)
            bsz, q_len, _ = normed.size()
            value = value.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)

            # Get the attention weights of the layer.
            attn_weights = attention_weights[layer_index]

            # Get the attention weights of the label token.
            attn_weights_index = attn_weights[:, :, -1, token_index].unsqueeze(-1)

            # Get the value vectors of the label token.
            value_index = value[:, :, token_index, :]

            # Calculate the hidden state of the label token.
            hidden_state = attn_weights_index * value_index
            hidden_state = hidden_state.view(attn.num_heads * attn.head_dim)
            hidden_state = attn.o_proj(hidden_state.unsqueeze(0))
            hidden_state = ln_f(hidden_state)

            # Calculate the logits of the label token.
            logits_hidden_state = torch.matmul(self.model.lm_head.weight.detach(), hidden_state.T).squeeze(-1)

            # Append the logits of the label token to the list.
            logits.append(logits_hidden_state)

        return logits
    
    def get_layers_logits(self, input_ids, token_index,**kwargs):
        # get the hidden states and attention weights of all layers
        outputs = self.model(input_ids=input_ids,  
                            output_hidden_states=True,
                            output_attentions=True,
                            **kwargs)
        hidden_states=outputs.hidden_states
        attn_weights = outputs.attentions

        # get the logits of the label token for each layer
        logits = self.layer_decode_hidden_state(hidden_states,attn_weights,token_index)
        
        return torch.stack(logits)
    
    def rr_per_layer(self, logits, task_token):
        # tokenize the task token
        task_token_id = self.tokenizer.encode(task_token)[1]

        # create an empty list to store the reciprocal ranks of the task token for each layer
        rrs = []

        for i, layer in enumerate(logits):
            # calculate the softmax probabilities
            probs = F.softmax(layer,dim=-1)

            # sort the probabilities in descending order
            sorted_probs = probs.argsort(descending=True)

            # calculate the rank of the task token
            rank = float(np.where(sorted_probs.cpu().numpy()==task_token_id)[0][0])

            # Append the reciprocal rank of the task token to the list
            rrs.append(1/(rank+1))

        return np.array(rrs)


class GPT2Wrapper(ModelWrapper):

    def layer_decode(self, hidden_states):
        logits = []
        for i,h in enumerate(hidden_states):
            h=h[:, -1, :] #(batch, num tokens, embedding size) take the last token
            if i == len(hidden_states)-1:
                normed = h #ln_f would already have been applied
            else:
                normed = self.model.transformer.ln_f(h)
            l = torch.matmul(self.model.lm_head.weight, normed.T)
            logits.append(l)
        return logits

    def layer_decode_logitslens(self,  activation_logitslens_):
        logits = []
        for i,h in enumerate(activation_logitslens_):
            normed = self.model.transformer.ln_f(h)
            l = torch.matmul(self.model.lm_head.weight.detach(), normed)
            logits.append(l)
        return logits


    def add_hooks(self):
        for i in range(self.num_layers):
            #intermediate residual between
            #print('saving hook') 

            self.model.transformer.h[i].mlp.register_forward_hook(self.get_activation('mlp_'+str(i)))
            #print(self.model.activations_)

    def add_hooks_logitslens(self):
        for i in range(self.num_layers):
            #intermediate residual between
            #print('saving hook') 
            self.model.transformer.h[i].ln_1.register_forward_hook(self.get_activation_logitslens())
            
            self.model.transformer.h[i].ln_2.register_forward_hook(self.get_activation_logitslens())
        
            #print(self.model.activations_)

    def add_hooks_logitslens_label_position(self, label_index):
        for i in range(self.num_layers):
            #intermediate residual between
            #print('saving hook') 
            self.model.transformer.h[i].ln_1.register_forward_hook(self.get_activation_logitslens_label_position(label_index))
            
            self.model.transformer.h[i].ln_2.register_forward_hook(self.get_activation_logitslens_label_position(label_index))
        
            #print(self.model.activations_)
   
    def get_pre_wo_activation(self, name):
        #wo refers to the output matrix in attention layers. The last linear layer in the attention calculation

        def hook(module, input, output):
            #use_cache=True (default) and output_attentions=True have to have been passed to the forward for this to work
            _, past_key_value, attn_weights = output
            value = past_key_value[1]
            pre_wo_attn = torch.matmul(attn_weights, value)    
            self.model.activations_[name]=pre_wo_attn

        return hook

    def get_past_layer(self, name):
        #wo refers to the output matrix in attention layers. The last linear layer in the attention calculation

        def hook(module, input, output):
            #use_cache=True (default) and output_attentions=True have to have been passed to the forward for this to work
            #print(len(output), output, name)
            _, past_key_value, attn_weights = output  
            self.layer_pasts[name]=past_key_value

        return hook

    def add_mid_attn_hooks(self):
        for i in range(self.num_layers):
            self.hooks.append(self.model.transformer.h[i].attn.register_forward_hook(self.get_pre_wo_activation('mid_attn_'+str(i))))

            self.hooks.append(self.model.transformer.h[i].attn.register_forward_hook(self.get_past_layer('past_layer_'+str(i))))

    def rm_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def reset_activations():
        self.activations_ = {}
        self.last_pasts = {}
           


class BloomWrapper(ModelWrapper):
    def layer_decode(self, hidden_states):
        logits = []
        for i,h in enumerate(hidden_states):
            h=h[:, -1, :] #(batch, num tokens, embedding size) take the last token
            if i == len(hidden_states)-1:
                normed = h #ln_f would already have been applied
            else:
                normed = self.model.transformer.ln_f(h)
            
            l = torch.matmul(self.model.lm_head.weight, normed.T)
            logits.append(l)
        return logits

    def add_hooks(self):
        for i in range(self.num_layers):
            #intermediate residual between
            #print('saving hook')
            #self.hooks.append(self.model.transformer.h[i].ln_1.register_forward_hook(self.get_activation(f'in_sln_{i}')))
            self.hooks.append(self.model.transformer.h[i].self_attention.register_forward_hook(self.get_activation('attn_'+str(i))))
            self.hooks.append(self.model.transformer.h[i].mlp.register_forward_hook(self.get_activation("intermediate_residual_" + str(i))))
            self.hooks.append(self.model.transformer.h[i].mlp.register_forward_hook(self.get_activation('mlp_'+str(i))))

class BloomPetalsWrapper(BloomWrapper):
    def get_layers(self, tokens, **kwargs):
        outputs = self.model(input_ids=tokens, output_hidden_states=True, **kwargs)
        hidden_states, true_logits = outputs.hidden_states, outputs.logits #hidden states will be none unfortunately.
        logits = [true_logits.squeeze(0)[-1].unsqueeze(-1),] #no real reason for this weirdness
        return torch.stack(logits).squeeze(-1)#, true_logits.squeeze(0)

    #note: attention and mlp outputs have residual already added in bloom. Need to subtract input from output to get effect
    #see here: https://github.com/huggingface/transformers/blob/983e40ac3b2af68fd6c927dce09324d54d023e54/src/transformers/models/bloom/modeling_bloom.py#L212
