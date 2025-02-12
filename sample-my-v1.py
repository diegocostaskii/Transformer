"""
Sample from a trained model
"""
import os
import torch
import json
from transformers import GPT2Tokenizer
from model import ModelM

# -----------------------------------------------------------------------------
init_from = ('resume', 'resume', 'resume')  # 'resume' 或者指定预训练的 GPT-2 模型 (例如 'gpt2-xl')
checkpoint_name = 'ckpt_20250211_165724.pt'
out_dir = 'out'  # 如果 init_from 是 'resume'，这里会用到
start = "\n"  # 或 "<|endoftext|>" 或其他自定义起始文本
num_iterations = 5
num_samples = 10  # 生成的样本数
max_new_tokens = 256  # 每个样本生成的最大token数量
temperature = 0.8  # 控制生成随机性的温度
top_k = 200  # top_k 策略
seed = 1337
device = 'cuda:3'  # 设备设置
compile = False  # 是否使用 PyTorch 2.0 编译加速
# 读取输入文件并进行编码
input_file = '/data/jyliu/transformer_project/nanoGPT-master/data/gsm8k/test.jsonl'
output_file = '/data/jyliu/transformer_project/our_model/test_results/test_output_' + checkpoint_name[:-3] + '.json'
batch_size = 64
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # 允许TF32加速
torch.backends.cudnn.allow_tf32 = True  # 允许cudnn加速
device_type = 'cuda' if 'cuda' in device else 'cpu'  # 设备类型
device = torch.device(device)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model
if init_from[0] == 'resume':
    ckpt_path = os.path.join(out_dir, checkpoint_name)
    checkpoint = torch.load(ckpt_path, map_location=device)
    encoder_name, think_name, decoder_name = checkpoint['encoder_name'], checkpoint['think_name'], checkpoint['decoder_name']
    model = ModelM(tokenizer, init_from=(encoder_name, think_name, decoder_name), neuron_dim=100, num_iterations=num_iterations)
    model.load_state_dict(checkpoint['model'])
elif init_from[0].startswith('gpt2'):
    model = ModelM(tokenizer, init_from=init_from, neuron_dim=100, num_iterations=num_iterations)
    
model.eval()
model = model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

def load_jsonl(file):
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def write_jsonl(res, outfile):
    with open(outfile, 'w', encoding='utf-8') as f:
        for d in res:
            f.writelines(json.dumps(d, ensure_ascii=False))
            f.writelines('\n')

def process_batches(input_file, output_file, model, tokenizer, device, batch_size=4, max_new_tokens=500, temperature=1.0, top_k=40):
    inputs_data = load_jsonl(input_file)
    eos_str = tokenizer.decode([tokenizer.eos_token_id])  # EOS token
    inputs = [d['question'] for d in inputs_data]  # 不需要再拼接'\n'了, 在模型的generate函数中已处理
    outputs = []
    
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i + batch_size]
        batch_tokens = [torch.tensor(tokenizer.encode(text), dtype=torch.long).to(device) for text in batch]
        batch_tokens = [t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.long, device=device) for t in batch_tokens]
        x = torch.nn.utils.rnn.pad_sequence(batch_tokens, batch_first=True, padding_value=tokenizer.eos_token_id)

        # 生成 attention mask 来标识填充部分
        attention_mask = (x != tokenizer.eos_token_id).long()  # 填充部分的 attention_mask 设为 0
        
        # 使用GPT2LMHeadModel进行生成
        with torch.no_grad():
            outputs_batch = model.generate(
                input_ids=x,
                attention_mask=attention_mask,  # 将 attention_mask 加入
                max_length=x.shape[1] + max_new_tokens,  # 生成的最大长度
                temperature=temperature,
                top_k=top_k
            )
            
            output_i = [[] for _ in range(batch_size)]
            for i in range(len(outputs_batch)):
                for idx, output in enumerate(outputs_batch[i]):
                    output_text = tokenizer.decode(output)
                    output_text = output_text.split(eos_str)[0]  # 以EOS符号截断
                    output_i[idx].append(output_text)
            outputs += output_i

    write_jsonl(outputs, output_file)
    print(f"Processed outputs saved to {output_file}")

process_batches(input_file, output_file, model, tokenizer, device, batch_size, max_new_tokens, temperature, top_k)