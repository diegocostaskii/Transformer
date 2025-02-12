import json
import torch
from torch.utils.data import Dataset

# Data Loader
class ProblemAnswerDataset(Dataset):
    def __init__(self, file_path, tokenizer, num_splits=5, max_length=1024):
        """
        Args:
            file_path (str): Path to the dataset (JSONL file with {"problem": ..., "answer": ...}).
            tokenizer: A tokenizer (e.g., GPT tokenizer) for tokenizing input text.
            max_length (int): Maximum sequence length.
            eos_token_id (int): End-of-sequence token ID.
        """
        self.data = self.load_jsonl(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eos_token_id = tokenizer.eos_token_id
        self.num_splits=num_splits
    
    def load_jsonl(self, file):
        data = []
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def split_answer(self, answer):
        sentences = answer.split('\n')
        sentences = sentences[:-2] + [sentences[-2]+'\n'+sentences[-1]]  # 总结数值结果不算一步
        num_sentences = len(sentences)

        # 计算每个 chunk 的大致大小
        avg_chunk_size = num_sentences / self.num_splits
        splits = []
        start_idx = 0

        for i in range(self.num_splits):
            # 计算当前 chunk 的结束索引
            end_idx = round((i + 1) * avg_chunk_size)  # 四舍五入取整，确保所有句子都分配
            splits.append(" ".join(sentences[start_idx:end_idx]).strip())  # 组合句子
            start_idx = end_idx  # 更新索引

        # 确保始终有num_splits段
        while len(splits) < self.num_splits:
            splits.append("")  # 如果不够num_splits组，填充空字符串
        return splits
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        problem = item["question"]
        answer = item["answer"]
        
        # Tokenize
        problem_tokens = torch.tensor(self.tokenizer.encode(problem), dtype=torch.long)

        answer_splits = self.split_answer(answer)
        # 生成num_splits个 `targets`
        targets = []
        for i in range(self.num_splits):
            target_text = answer_splits[i]
            target_tokens = torch.tensor(self.tokenizer.encode("\n" + target_text) + [self.eos_token_id], dtype=torch.long)
            targets.append(target_tokens)
        
        return {
            "input_ids": problem_tokens,
            "targets": targets
        }

class CollateFn:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch):
        """Collate function for dynamic padding."""
        max_input_len = max(len(item["input_ids"]) for item in batch)
        num_splits = len(batch[0]["targets"])

         # 计算 num_splits 个 `targets` 的最大长度
        max_target_lens = [max(len(item["targets"][i]) for item in batch) for i in range(num_splits)]

        # Padding
        input_ids = []
        targets = [[] for _ in range(num_splits)]
        loss_masks = [[] for _ in range(num_splits)]
        attention_masks = []

        for item in batch:
            pad_len = max_input_len - len(item["input_ids"])
            input_ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), self.pad_token_id, dtype=torch.long)]))
            attention_masks.append(torch.cat([torch.ones(len(item["input_ids"]), dtype=torch.float), torch.zeros(pad_len, dtype=torch.float)]))

            for i in range(num_splits):
                target_pad_len = max_target_lens[i] - len(item["targets"][i])
                targets[i].append(torch.cat([item["targets"][i], torch.full((target_pad_len,), self.pad_token_id, dtype=torch.long)]))
                loss_masks[i].append(torch.cat([torch.ones(len(item["targets"][i]), dtype=torch.float), torch.zeros(target_pad_len, dtype=torch.float)]))
    
        return {
            "input_ids": torch.stack(input_ids),  # (batch, max_input_len)
            "attention_mask": torch.stack(attention_masks),  # (batch, max_input_len)
            "targets": [torch.stack(t) for t in targets],  # 5 个 tensor，每个 (batch, max_target_len)
            "loss_masks": [torch.stack(m) for m in loss_masks]  # 5 个 tensor，每个 (batch, max_target_len)
        }