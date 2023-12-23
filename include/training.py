from utility import*
from tqdm import tqdm

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    # Use tqdm for the progress bar
    with tqdm(train_loader, desc="Training", unit="batch") as tbar:
        for batch in tbar:
            images, labels, _ = batch
            # Move data to GPU if available
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Update the progress bar description with the current loss
            tbar.set_postfix(loss=loss.item())

    average_loss = total_loss / len(train_loader)
    return average_loss

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Use tqdm for the progress bar
    with tqdm(val_loader, desc="Validation", unit="batch") as tbar:
        for batch in tbar:
            images, labels,_ = batch
            # Move data to GPU if available
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            # Calculate accuracy
            predicted_labels = outputs
            correct_predictions += torch.sum(torch.abs(predicted_labels - labels) < 5.0)  # Adjust threshold as needed
            total_samples += labels.size(0)

            # Update the progress bar description with the current loss
            tbar.set_postfix(loss=loss.item())

    average_loss = total_loss / len(val_loader)
    accuracy = correct_predictions.item() / total_samples
    return average_loss, accuracy
