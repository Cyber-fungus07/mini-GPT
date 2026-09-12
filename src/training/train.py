import time
from pathlib import Path

import torch

from src.config import GPT_CONFIG
from src.data.dataloader import create_dataloader
from src.data.tokenizer import GPTTokenizer
from src.inference.generate import generate, text_to_token_ids, token_ids_to_text
from src.model.gpt_model import GPTModel
from src.training.loss import GPTLoss


class Trainer:

    def __init__(self, model, train_loader, val_loader, optimizer, device, tokenizer,
                 checkpoint_dir="checkpoints"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.tokenizer = tokenizer
        self.checkpoint_dir = Path(checkpoint_dir)

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

    def save_checkpoint(self, epoch, tokens_seen_count, filename="last.pt"):

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / filename

        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "tokens_seen_count": tokens_seen_count,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "tokens_seen": self.tokens_seen,
                "config": GPT_CONFIG,
            },
            path,
        )
        return path

    def load_checkpoint(self, path, load_optimizer=True):

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])

        if load_optimizer and "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        self.global_step = checkpoint.get("global_step", 0)
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.tokens_seen = checkpoint.get("tokens_seen", [])

        return checkpoint.get("epoch", 0), checkpoint.get("tokens_seen_count", 0)


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

    def train(self, num_epochs, eval_freq, eval_iter, start_context, start_epoch=0,
              tokens_seen=0, save_best=True):
        best_val_loss = min(self.val_losses) if self.val_losses else float("inf")

        for epoch in range(start_epoch, num_epochs):
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

            # End-of-epoch checkpoint: always update last.pt, plus best.pt on improvement
            current_val = self.val_losses[-1] if self.val_losses else float("nan")
            self.save_checkpoint(epoch + 1, tokens_seen, filename="last.pt")

            if save_best and current_val == current_val and current_val < best_val_loss:  # NaN-safe
                best_val_loss = current_val
                self.save_checkpoint(epoch + 1, tokens_seen, filename="best.pt")

            print(f"Checkpoint saved (epoch {epoch + 1}, val loss {current_val:.3f})")

        return self.train_losses, self.val_losses, self.tokens_seen


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    tokenizer = GPTTokenizer()

    with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Split on token boundary (not raw chars) so we never cut a word/token in half
    all_ids = tokenizer.encode(text)
    split_idx = int(len(all_ids) * 0.9)
    train_text = tokenizer.decode(all_ids[:split_idx])
    val_text = tokenizer.decode(all_ids[split_idx:])

    train_loader = create_dataloader(
        train_text, tokenizer=tokenizer, batch_size=4, max_length=256, stride=128, drop_last=True
    )
    val_loader = create_dataloader(
        val_text, tokenizer=tokenizer, batch_size=4, max_length=256, stride=128, shuffle=False, drop_last=False
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
        tokenizer=tokenizer,
        checkpoint_dir="checkpoints",
    )

    # Resume from last checkpoint if one exists
    start_epoch, tokens_seen = 0, 0
    resume_path = Path("checkpoints/last.pt")

    if resume_path.exists():
        start_epoch, tokens_seen = trainer.load_checkpoint(resume_path)
        print(f"Resumed from {resume_path} (epoch {start_epoch}, step {trainer.global_step})")

    start_time = time.time()

    trainer.train(
        num_epochs=5,
        eval_freq=5,
        eval_iter=2,
        start_context="Every effort moves you",
        start_epoch=start_epoch,
        tokens_seen=tokens_seen,
    )

    execution_time = (time.time() - start_time) / 60
    print(f"Training completed in {execution_time:.2f} minutes.")


if __name__ == "__main__":
    main()
