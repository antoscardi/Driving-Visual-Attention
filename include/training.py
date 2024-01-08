from utility import*
 

def train_epoch(model, train_loader, criterion, scheduler, optimizer, device, epoch):
    model.train(True)
    total_loss = 0.0

    # Use tqdm for the progress bar
    with tqdm(train_loader, desc=f"Training Epoch {epoch}", unit="batch") as tbar:
        for batch in tbar:
            eye_left, face_features, labels,_,_ = batch
            # Move data to GPU if available
            eye_left, face_features, labels = eye_left.to(device), face_features.to(device), labels.to(device)

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

def validate(model, bbox_accuracy, val_loader, threshold, criterion, device, epoch, BATCH_SIZE):
    model.train(False)
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_threshold_accuracy = 0.0
        total_bbox_accuracy = 0.0
        total_paper_accuracy = 0.0


        # Use tqdm for the progress bar
        with tqdm(val_loader, desc=f"Validation Epoch {epoch}", unit="batch") as tbar:
            for batch in tbar:
                eye_left,face_features,labels, bbox,_ = batch
                eye_left, face_features, labels = eye_left.to(device),face_features.to(device),labels.to(device)

                # Forward pass
                outputs = model(eye_left,face_features)
                loss = criterion(outputs, labels)
                total_loss += loss.detach().item()

                # Calculate predictions as sum in the batch
                l2_norm = torch.norm(outputs - labels, dim=1, p=2)
                batch_threshold_predictions = torch.sum(l2_norm < threshold) # Adjust Threshold
                batch_bbox_predictions = torch.sum(bbox_accuracy(outputs, bbox))
                batch_paper_predictions = torch.sum(l2_norm)

                # Calculate accuracies by dividing for the batch size 
                batch_threshold_accuracy = batch_threshold_predictions / BATCH_SIZE
                batch_bbox_accuracy = batch_bbox_predictions / BATCH_SIZE
                batch_paper_accuracy = batch_paper_predictions / BATCH_SIZE

                # Sum accuracies in each batch to get the total of one epoch
                total_threshold_accuracy += batch_threshold_accuracy.detach().item()
                total_bbox_accuracy += batch_bbox_accuracy.item()
                total_paper_accuracy += batch_paper_accuracy.detach().item()

                # Update the progress bar description with the current loss and accuracy
                tbar.set_postfix({'batch_loss': f'{loss.detach().item():.2f}'})
                tbar.set_postfix({'batch accuracy': f'{batch_threshold_accuracy.detach().item()*100:.2f}%'})

        average_loss = total_loss / len(val_loader)
        # Calculate the final acc by dividing for the number of batches
        threshold_accuracy = total_threshold_accuracy / len(val_loader)
        bbox_accuracy = total_bbox_accuracy / len(val_loader)
        paper_accuracy = total_paper_accuracy / len(val_loader)

        return average_loss, threshold_accuracy, bbox_accuracy, paper_accuracy
