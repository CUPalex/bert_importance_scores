from transformers import AutoTokenizer, BertModel
import torch
import wandb
import logging
import tqdm

from task import Task
from model import BertWithNewHead
    
class IdentityAttn(torch.nn.Module):
    def forward(self, x, *args, **kwargs):
        return (x, None)
    
class IdentityMlp(torch.nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class Pipeline:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.task = Task()

    def initialize_task_seed(self, task, seed, batch_size):
        self.seed = seed
        self.task.create_datasets_and_dataloaders(task, self.tokenizer, batch_size)
        self.device = torch.device("mps") if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = BertWithNewHead(output_dim=self.task.num_classes, seed=seed)
        self.model.to(self.device)
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters())
        logging.info(f"DEVICE: {self.device}")

    def train_epoch(self):
        self.model.train()
        sum_losses = 0.
        for batch in tqdm.tqdm(self.task.dataloader_train):
            inputs = batch["input_ids"].to(self.device)
            attn_masks = batch["attention_mask"].to(self.device)
            gold_outputs = batch["labels"].flatten().to(self.device).to(torch.int64)
            
            predicted_logits = self.model(inputs, attention_mask=attn_masks)

            batch_loss = self.loss(predicted_logits, gold_outputs)

            wandb.log({"step/loss": batch_loss.item(),
                       "step/mse": torch.nn.functional.mse_loss(torch.argmax(predicted_logits, dim=-1).to(torch.float32),
                                                                gold_outputs.to(torch.float32), reduction="mean").item(),
                        "inputs/labels": gold_outputs.detach().cpu().numpy(),
                        "inputs/predicted": predicted_logits.detach().cpu().numpy()})
            sum_losses += batch_loss.item()
            
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
        wandb.log({"epoch/loss" : sum_losses / len(self.task.dataloader_train)})
    
    def validate(self, dataloader, log_to_wandb=True):
        self.model.eval()
        sum_losses = 0.
        correct = 0.
        total = 0.
        sum_mse = 0.
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader):
                inputs = batch["input_ids"].to(self.device)
                attn_masks = batch["attention_mask"].to(self.device)
                gold_outputs = batch["labels"].flatten().to(self.device).to(torch.int64)
                assert (gold_outputs < 2).all() and (gold_outputs >= 0).all(), gold_outputs
                
                predicted_logits = self.model(inputs, attention_mask=attn_masks)
                batch_loss = self.loss(predicted_logits, gold_outputs)

                sum_mse += torch.nn.functional.mse_loss(torch.argmax(predicted_logits, dim=-1).to(torch.float32),
                                                        gold_outputs.to(torch.float32), reduction="sum").item()
                sum_losses += batch_loss.item()
                correct += (torch.argmax(predicted_logits, dim=-1) == gold_outputs).sum().item()
                total += gold_outputs.shape[0]

        if log_to_wandb:
            wandb.log({"epoch/acc" : correct / total, "epoch/val_loss": sum_losses / len(dataloader), "epoch/val_mse": sum_mse / len(dataloader)})
        return correct / total, sum_losses / len(dataloader), sum_mse / len(dataloader)
    
    def train(self, until_val_loss_goes_up=False, num_epochs=10):
        wandb.init(project="coli",
                   name=f"{self.task.name}-seed-{self.seed}",
                   config={
                       "task": self.task.name,
                       "num_classes": self.task.num_classes,
                       "chance_performance": self.task.chance_performance,
                       "dataset_train_size": len(self.task.dataset_train),
                       "dataset_val_size": len(self.task.dataset_val),
                       "dataset_test_size": len(self.task.dataset_test),
                   })

        prev_loss = None
        for epoch in range(num_epochs):
            self.train_epoch()
            _, loss, _ = self.validate(self.task.dataloader_val)
            wandb.log({"epochs": epoch})
            if until_val_loss_goes_up and prev_loss is not None and prev_loss < loss:
                break

        wandb.finish()

    def find_importance_scores(self):
        wandb.init(project="coli", name=f"{self.task.name}-seed-{self.seed}-imp-scores")
        num_layers = 12
        accs = [[0, 0, 0, 0, 0, 0] for l in range(num_layers)]
        acc, loss, mse = self.validate(self.task.dataloader_test, log_to_wandb=False)
        wandb.log({"test_acc": acc, "test_loss": loss, "test_mse": mse})
        for layer in range(num_layers):
            attn = self.model.bert.encoder.layer[layer].attention
            self.model.bert.encoder.layer[layer].attention = IdentityAttn()
            acc, loss, mse = self.validate(self.task.dataloader_test, log_to_wandb=False)
            accs[layer][0] = acc
            accs[layer][1] = loss
            accs[layer][2] = mse
            self.model.bert.encoder.layer[layer].attention = attn

            intermediate = self.model.bert.encoder.layer[layer].intermediate
            output = self.model.bert.encoder.layer[layer].output
            self.model.bert.encoder.layer[layer].intermediate = IdentityMlp()
            self.model.bert.encoder.layer[layer].output = IdentityMlp()
            acc, loss, mse = self.validate(self.task.dataloader_test, log_to_wandb=False)
            accs[layer][3] = acc
            accs[layer][4] = loss
            accs[layer][5] = mse
            self.model.bert.encoder.layer[layer].intermediate = intermediate
            self.model.bert.encoder.layer[layer].output = output
        wandb.log({"accs_with_removal": wandb.Table(data=accs, columns=["attn_acc", "attn_loss", "attn_mse", "mlp_acc", "mlp_loss", "mlp_mse"])})
        wandb.finish()
