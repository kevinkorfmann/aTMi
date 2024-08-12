from tqdm import tqdm
import torch
import logging


def train_model(model, train_dataloader, criterion, optimizer, num_epochs=1, device=None, accelerator=None):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        with tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as batch_bar:
            for inputs, targets in batch_bar:
                inputs = inputs.to(device)
                targets = targets.to(device)
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                epoch_loss += loss.item()
                batch_bar.set_postfix(loss=f'{loss.item():.4f}')
                batch_bar.update(1)
        avg_loss = epoch_loss / len(train_dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Avg. Loss: {avg_loss:.4f}')
        logging.info(f'train,{epoch + 1},{avg_loss:.4f}')

def test_model(model, test_dataloader, criterion, num_epochs=1, device=None):
    model.eval()
    with torch.no_grad():
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            with tqdm(test_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as batch_bar:
                for inputs, targets in batch_bar:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    predictions = model(inputs)
                    loss = criterion(predictions, targets)
                    epoch_loss += loss.item()
                    batch_bar.set_postfix(loss=f'{loss.item():.4f}')
                    batch_bar.update(1)
            avg_loss = epoch_loss / len(test_dataloader)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Avg. Loss: {avg_loss:.4f}')
            logging.info(f'test,{epoch + 1},{avg_loss:.4f}')