from utility import*
from tqdm import tqdm

def train_epoch(model, train_loader, criterion, scheduler, optimizer, device, epoch):
    model.train(True)
    total_loss = 0.0

    # Use tqdm for the progress bar
    with tqdm(train_loader, desc=f"Training Epoch {epoch}", unit="batch") as tbar:
        for batch in tbar:
            eye_left,face_features,labels,_ = batch
            # Move data to GPU if available
            eye_left, face_features, labels = eye_left.to(device),face_features.to(device),labels.to(device)

            # Forward pass
            outputs = model(eye_left,face_features)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.detach().item()

            # Update the progress bar description with the current loss
            tbar.set_postfix({'batch loss': f'{loss.detach().item():.2f}'})

    average_loss = total_loss / len(train_loader)
    return average_loss

def validate(model, val_loader, threshold, criterion, device, epoch):
    model.train(False)
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_correct_predictions = 0
        total_samples = 0

        # Use tqdm for the progress bar
        with tqdm(val_loader, desc=f"Validation Epoch {epoch}", unit="batch") as tbar:
            for batch in tbar:
                images, labels, _ = batch
                # Move data to GPU if available
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.detach().item()

                # Calculate accuracy
                predicted_labels = outputs
                current_correct_predictions = torch.sum(torch.norm(predicted_labels - labels) < threshold) # Adjust threshold as needed
                total_correct_predictions += current_correct_predictions
                
                total_samples += labels.size(0)

                batch_accuracy = current_correct_predictions / labels.size(0)*100

                # Update the progress bar description with the current loss and accuracy
                tbar.set_postfix({'batch_loss': f'{loss.detach().item():.2f}'})
                tbar.set_postfix({'batch accuracy': f'{batch_accuracy.detach().item():.2f}%'})

        average_loss = total_loss / len(val_loader)
        accuracy = total_correct_predictions.item() / total_samples * 100
        return average_loss, accuracy
