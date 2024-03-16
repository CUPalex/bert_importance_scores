from transformers import BertModel
import torch

class BertWithNewHead(torch.nn.Module):
    def __init__(self, output_dim, seed):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for p in self.bert.parameters():
            p.requires_grad = False
        torch.manual_seed(seed)
        self.linear = torch.nn.Linear(768, output_dim)
    def forward(self, x, **kwargs):
        x = self.bert(x, **kwargs)
        return self.linear(x.last_hidden_state)[:, 0, :]