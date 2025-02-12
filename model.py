import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

# Define our model
class ModelM(nn.Module):
    def __init__(self, tokenizer, init_from=('gpt2', 'gpt2', 'gpt2'), neuron_dim=100, num_iterations=5):
        super().__init__()
        self.tokenizer = tokenizer

        # Initialize modules
        self.encoder = GPT2LMHeadModel.from_pretrained(init_from[0])  # Encoder: GPT-2 (Frozen Parameters)
        self.think_model = GPT2LMHeadModel.from_pretrained(init_from[1])  # Thinking Module: GPT-2 (Trainable Parameters)
        self.decoder = GPT2LMHeadModel.from_pretrained(init_from[2])  # Decoding Module: GPT-2 (Trainable Parameters)
        
        # Extra Learnable Matrix (100 x embedding_dim)
        self.neuron_dim = neuron_dim
        self.neuron_matrix = nn.Parameter(torch.randn(neuron_dim, self.think_model.config.n_embd))
        
        self.num_iterations = num_iterations

    def forward(self, input_ids, attention_mask):
        """
        input_ids: (batch, seq_len)
        attention_mask: (batch, seq_len), where 1 = real token, 0 = padding
        decode_text_ids: list of (batch, dec_len)
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Encoder Forward Pass (with padding considered)
        with torch.no_grad():
            encoder_outputs = self.encoder.transformer(input_ids, attention_mask=attention_mask).last_hidden_state
        
        # Expand Neuron Matrix to Match Batch Size
        neuron_matrix = self.neuron_matrix.to(device).unsqueeze(0).expand(batch_size, -1, -1)  # (batch, 100, dim)

        # Update Attention Mask
        extra_attention_mask = torch.ones((batch_size, self.neuron_dim), dtype=torch.long, device=device)  # (batch, 100)
        updated_attention_mask = torch.cat([attention_mask, extra_attention_mask], dim=1)  # (batch, seq_len + 100)

        neuron_matrixes = []
        
        for i in range(self.num_iterations):
            # Concatenate Encoder Representation with Extra Matrix
            thinking_input = torch.cat([encoder_outputs, neuron_matrix], dim=1)  # (batch, seq_len + 100, dim)
            # Pass to Model
            output = self.think_model.transformer(inputs_embeds=thinking_input, attention_mask=updated_attention_mask).last_hidden_state  # (batch, seq_len + 100, dim)
            
            # Extract the Extra Matrix Part for Next Iteration
            neuron_matrix = output[:, -self.neuron_dim:, :]  # (batch, 100, dim)
            neuron_matrixes.append(neuron_matrix)
        return neuron_matrixes

    def decode(self, neuron_matrix, decode_text_ids=None):
        device = neuron_matrix.device  
        if decode_text_ids is not None:
            # 计算 decode_text_ids 的 attention_mask
            decode_text_mask = (decode_text_ids != self.tokenizer.eos_token_id).long().to(device)  # (batch, decode_len)
            
            # 用 decoder 计算 decode_text_ids 的表征
            decode_text_embeds = self.decoder.transformer.wte(decode_text_ids)  # (batch, decode_len, dim)
            position_ids = torch.arange(0, decode_text_ids.size(-1), dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, decode_text_ids.size(-1))
            position_embeds = self.decoder.transformer.wpe(position_ids)
            decode_text_embeds = decode_text_embeds + position_embeds

            # 拼接 neuron_matrix 和 decode_text_embeds
            decoder_input_embeds = torch.cat([neuron_matrix, decode_text_embeds], dim=1)  # (batch, 100 + decode_len, dim)
            
            # 扩展 decode_text_mask
            extra_decode_mask = torch.ones((neuron_matrix.size(0), self.neuron_dim), dtype=torch.long, device=device)  # (batch, 100)
            decoder_attention_mask = torch.cat([extra_decode_mask, decode_text_mask], dim=1)  # (batch, 100 + decode_len)

        else:
            # 只使用 neuron_matrix 作为 decoder 输入
            decoder_input_embeds = neuron_matrix  # (batch, 100, dim)
            decoder_attention_mask = torch.ones((neuron_matrix.size(0), self.neuron_dim), dtype=torch.long, device=device)  # (batch, 100)
        
        decoder_outputs = self.decoder(inputs_embeds=decoder_input_embeds, attention_mask=decoder_attention_mask)
        logits = decoder_outputs.logits[:, self.neuron_dim:, :]  # (batch, 100 + decode_len, vocab_size) -> (batch, decode_len, vocab_size)
        return logits
    
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
