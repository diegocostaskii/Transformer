import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config

class StackedGPT2Model(nn.Module):
    def __init__(self, config, fixed_additional_tensor = 50, additional_tensor_size=100, stackLayers = 5, vocab_size = 50304, block_size = 1024,dropout = 0.2 ):
        super(StackedGPT2Model, self).__init__()
        print(config)
        self.device = config['device']
        self.stackLayers = stackLayers
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, config['n_embd']),
            wpe = nn.Embedding(block_size, config['n_embd']),
            drop = nn.Dropout(dropout)
        ))
        self.additional_tensor_size = additional_tensor_size
        # 创建五个 GPT2LMHeadModel 实例
        self.models = nn.ModuleList([GPT2LMHeadModel.from_pretrained(config['init_from']).to(self.device) for _ in range(stackLayers)])
        
        # 定义一个定长的张量，形状为 (1, additional_tensor_size) 并固定其中一部分
        self.additional_tensorList = []
        for i in range(stackLayers):
            self.additional_tensorList.append(nn.Parameter(torch.randn(config['batch_size'], additional_tensor_size, config['n_embd'])).to(self.device))

        self.fixed_additional_tensorList = []
        for i in range(stackLayers):
            self.fixed_additional_tensorList.append(self.additional_tensorList[i][:,:fixed_additional_tensor,:])

    
    def forward(self, input_ids, attention_mask, targets=None):
        # 在每一层中循环输入
        All_logits = []
        idx = input_ids
        b, t = idx.size()
        assert t <= 1024, f"Cannot forward sequence of length {t}, block size is only {1024}"
        pos = torch.arange(0, t, dtype=torch.long, device=self.device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        for i, model in enumerate(self.models):
            # 拼接定长张量
            # 先将 additional_tensor 扩展到 (batch_size, additional_tensor_size)
            # B,T = x.size()
            # print('xsize',B,T)
            add_T = self.additional_tensorList[i]
            x = torch.cat((x,self.fixed_additional_tensorList[i], add_T[:, 50:,:]),1)
            # 通过当前的 GPT2LMHeadModel 进行前向传播
            outputs = model(inputs_embeds=x, attention_mask=attention_mask, output_hidden_states= True)
            # 获取 logits 和 loss
            hidden_states = outputs.hidden_states[0]  # (batch_size, sequence_length, vocab_size)
            
            logits = outputs.logits
            logits = logits[:, :-self.additional_tensor_size, :]
            All_logits.append(logits)

            x = hidden_states[:, :-self.additional_tensor_size, :]

            # print('gpt2model No.'+ str(i) +' finished')

              
        
        return All_logits
    
    def generate(self, input_ids, attention_mask, max_length=256, temperature=1.0, top_k=40):
        """
        input_ids: 初始输入文本的 token ids
        attention_mask: 注意力掩码
        max_length: 最多生成多少个新 token
        temperature: 采样温度
        top_k: 限制最高概率的前 k 个 token 进行采样
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]

        # 以 '\n' 作为起始 token
        newline_token_id = self.tokenizer.encode("\n")[0]
        decode_text_ids = [torch.full((batch_size, 1), newline_token_id, dtype=torch.long, device=device) for _ in range(self.num_iterations)]
        
        # 获取thinking结果
        neuron_matrixes = self.forward(input_ids, attention_mask)
        
        # 生成循环
        skip_i = []
        for _ in range(max_length):
            for i in range(self.num_iterations):
                if i in skip_i:
                    continue
                logits = self.decode(neuron_matrixes[i], decode_text_ids[i])
                logits = logits[:, -1, :]  # 取最后一个 token 的 logits

                # 采样：Top-K + Temperature
                if top_k > 0:
                    values, indices = torch.topk(logits, top_k)
                    probs = torch.softmax(values / temperature, dim=-1)
                    next_token = torch.gather(indices, 1, torch.multinomial(probs, num_samples=1))
                else:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                
                decode_text_ids[i] = torch.cat([decode_text_ids[i], next_token], dim=1)  # 追加新 token

                # 如果 batch 内的所有样本都生成了 `eos_token_id`，提前终止
                if torch.all(next_token == self.tokenizer.eos_token_id):
                    skip_i.append(i)

        return decode_text_ids  # 返回完整生成的 token 序列, num_iterations长度的list, 每一个元素batch x dec_len