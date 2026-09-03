import time

import torch

from src.config import GPT_CONFIG
from src.data.dataloader import create_dataloader
from src.data.tokenizer import GPTTokenizer
from src.inference.generate import generate, text_to_token_ids, token_ids_to_text
from src.model.gpt_model import GPTModel
from src.training.loss import GPTLoss


class Trainer:

    def __init__(self, model, train_loader, val_loader, optimizer, device, tokenizer):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.tokenizer = tokenizer

        self.loss_fn = GPTLoss()

        self.train_losses = []
        self.val_losses = []
        self.tokens_seen = []
        self.global_step = 0

    def calc_loss_batch(self, input_batch, target_batch):
        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)

        logits = self.model(input_batch)

        return self.loss_fn(logits, target_batch)

    def calc_loss_loader(self, data_loader, num_batches=None):
        if len(data_loader) == 0:
            return float("nan")

        if num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))

        total_loss = 0.0

        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break

            loss = self.calc_loss_batch(input_batch, target_batch)
            total_loss += loss.item()

        return total_loss / num_batches

    def evaluate(self, eval_iter):
        self.model.eval()

        with torch.no_grad():
            train_loss = self.calc_loss_loader(
                self.train_loader,
                eval_iter
            )

            val_loss = self.calc_loss_loader(
                self.val_loader,
                eval_iter
            )

        self.model.train()

        return train_loss, val_loss

    def generate_sample(self, start_context):
        self.model.eval()

        context_size = self.model.pos_emb.weight.shape[0]

        encoded = text_to_token_ids(
            start_context,
            self.tokenizer
        ).to(self.device)

        with torch.no_grad():
            token_ids = generate(
                self.model,
                encoded,
                max_new_tokens=25,
                context_size=context_size,
                temperature=1.4,
                top_k=25
            )

        decoded_text = token_ids_to_text(
            token_ids,
            self.tokenizer
        )

        print(decoded_text.replace("\n", " "))

        self.model.train()

    def train(self, num_epochs, eval_freq, eval_iter, start_context):
        tokens_seen = 0

        for epoch in range(num_epochs):
            self.model.train()

            for input_batch, target_batch in self.train_loader:
                self.optimizer.zero_grad()

                loss = self.calc_loss_batch(
                    input_batch,
                    target_batch
                )

                loss.backward()
                self.optimizer.step()

                tokens_seen += input_batch.numel()
                self.global_step += 1

                if self.global_step % eval_freq == 0:
                    train_loss, val_loss = self.evaluate(eval_iter)

                    self.train_losses.append(train_loss)
                    self.val_losses.append(val_loss)
                    self.tokens_seen.append(tokens_seen)

                    print(
                        f"Ep {epoch + 1} "
                        f"(Step {self.global_step:06d}): "
                        f"Train loss {train_loss:.3f}, "
                        f"Val loss {val_loss:.3f}"
                    )

            self.generate_sample(start_context)

        return self.train_losses, self.val_losses, self.tokens_seen


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    tokenizer = GPTTokenizer()

    with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Split text into train (90%) and val (10%)
    split_idx = int(len(text) * 0.9)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    train_loader = create_dataloader(
        train_text, tokenizer=tokenizer, batch_size=4, max_length=256, stride=128
    )
    val_loader = create_dataloader(
        val_text, tokenizer=tokenizer, batch_size=4, max_length=256, stride=128, shuffle=False
    )

    model = GPTModel(GPT_CONFIG).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004,
        weight_decay=0.1
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        tokenizer=tokenizer
    )

    start_time = time.time()

    trainer.train(
        num_epochs=1,
        eval_freq=5,
        eval_iter=2,
        start_context="Every effort moves you"
    )

    execution_time = (time.time() - start_time) / 60
    print(f"Training completed in {execution_time:.2f} minutes.")


if __name__ == "__main__":
    main()
