import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from distillation.loss_functions.distillation_loss import DistillationLoss
from distillation.loss_functions.undeviating_distillation_loss import UndeviatingDistillationLoss
from distillation.loss_functions.hybrid_distillation_loss import HybridDistillationLoss
from distillation.loss_functions.enhanced_hybrid_distillation_loss import EnhancedHybridDistillationLoss
import copy

def distillation_training(teacher_model, student_model, trainloader, config):
    # Set models
    teacher_model.eval()  # Teacher in eval mode
    student_model.train()  # Student in training mode

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model.to(device)
    student_model.to(device)

    # Define loss and optimizer
    criterion = DistillationLoss(alpha=config["alpha"], temperature=config["temperature"])
    optimizer = optim.Adam(student_model.parameters(), lr=config['learning_rate'])

    # Create output directory for predictions
    output_dir = './outputs/KD_epoch_out'
    os.makedirs(output_dir, exist_ok=True)

    # Set up transformation to convert tensor to PIL image
    to_pil = transforms.ToPILImage()

    # Training loop
    for epoch in range(config['epochs']):
        running_loss = 0.0
        for batch in tqdm(trainloader):
            inputs, labels = batch[:2]  # Extract only inputs and labels
            inputs, labels = inputs.to(device), labels.to(device)

            # Get teacher and student outputs
            with torch.no_grad():
                teacher_output = teacher_model(inputs)
            student_output = student_model(inputs)

            # Resize student output to match teacher output dimensions if they don't match
            if student_output.shape != teacher_output.shape:
                student_output = F.interpolate(student_output, size=teacher_output.shape[2:], mode='bilinear',
                                               align_corners=False)

            # Ensure labels have one channel and resize to match student_output dimensions
            labels = labels[:, :1, :, :]  # Select only one channel if labels have multiple channels
            labels = F.interpolate(labels, size=student_output.shape[2:], mode='bilinear', align_corners=False)

            # Compute loss and backpropagate
            loss = criterion(student_output, teacher_output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{config['epochs']}], Loss: {running_loss / len(trainloader):.4f}")

        # Save images for visualization
        save_visualization(inputs, student_output, teacher_output, output_dir, epoch)

    print("Training completed.")

def undeviating_distillation_training(teacher_model, student_model, trainloader, config):
    """
    Train the student model using distillation from the teacher model.

    Args:
        teacher_model (nn.Module): The teacher model.
        student_model (nn.Module): The student model to be trained.
        trainloader (DataLoader): DataLoader containing the training data.
        config (dict): A dictionary containing hyperparameters and configuration for training.

    Returns:
        nn.Module: The trained student model.
    """
    # Set models to appropriate modes
    teacher_model.eval()  # Teacher in eval mode (no gradients)
    student_model.train()  # Student in train mode (gradients enabled)

    # Move models to device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model.to(device)
    student_model.to(device)

    # Define loss function and optimizer
    criterion = UndeviatingDistillationLoss(alpha=config["alpha"], temperature=config["temperature"])
    optimizer = optim.Adam(student_model.parameters(), lr=config['learning_rate'])

    # Directories for saving outputs and checkpoints
    output_dir = './outputs/KD_epoch_out'
    checkpoint_dir = './outputs/checkpoints'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Set up transformation for visualization (optional)
    to_pil = transforms.ToPILImage()

    # Early stopping and checkpoint variables
    best_loss = float('inf')
    best_model_weights = copy.deepcopy(student_model.state_dict())
    patience_counter = 0

    # Training loop
    for epoch in range(config['epochs']):
        running_loss = 0.0
        for batch in tqdm(trainloader):
            inputs = batch[0]  # Assuming batch[0] contains the input data (no labels)
            inputs = inputs.to(device)

            # Get outputs from teacher (no gradients required) and student
            with torch.no_grad():
                teacher_output = teacher_model(inputs)
            student_output = student_model(inputs)

            # Resize student output to match teacher output dimensions (if needed)
            if student_output.shape != teacher_output.shape:
                student_output = F.interpolate(student_output, size=teacher_output.shape[2:], mode='bilinear', align_corners=False)

            # Compute loss and backpropagate
            loss = criterion(student_output, teacher_output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(trainloader)
        print(f"Epoch [{epoch + 1}/{config['epochs']}], Loss: {epoch_loss:.4f}")

        # Save visualization for this epoch (optional)
        save_visualization(inputs, student_output, teacher_output, output_dir, epoch)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_weights = copy.deepcopy(student_model.state_dict())
            torch.save(best_model_weights, os.path.join(checkpoint_dir, 'best_student_model.pth'))
            print("Saved best model with loss:", best_loss)
            patience_counter = 0  # Reset patience counter if loss improves
        else:
            patience_counter += 1  # Increment if no improvement

        # Early stopping if patience counter exceeds limit
        if patience_counter >= config['patience']:
            print("Early stopping due to no improvement in loss.")
            break

    # Load best model weights after training
    student_model.load_state_dict(best_model_weights)
    print("Training completed.")

    return student_model


def hybrid_distillation_training(teacher_model, student_model, trainloader, config):
    # Set models
    teacher_model.eval()  # Teacher in eval mode
    student_model.train()  # Student in training mode

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model.to(device)
    student_model.to(device)

    # Initialize loss and optimizer
    criterion = HybridDistillationLoss(alpha=config["alpha"], temperature=config["temperature"])
    optimizer = optim.Adam(student_model.parameters(), lr=config['learning_rate'])

    # Output directories for predictions and checkpoints
    output_dir = './outputs/KD_epoch_out'
    checkpoint_dir = './outputs/checkpoints'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Early stopping and checkpoint variables
    best_loss = float('inf')
    best_model_weights = copy.deepcopy(student_model.state_dict())
    patience_counter = 0

    # Training loop
    for epoch in range(config['epochs']):
        running_loss = 0.0
        for batch in tqdm(trainloader):
            # Extract inputs and ground truth labels
            inputs, _, _, _, _, gt_labels, _ = batch  # gt_labels should match the layout from your dataloader
            inputs, gt_labels = inputs.to(device), gt_labels.to(device).float()

            # Teacher outputs
            with torch.no_grad():
                teacher_output = teacher_model(inputs)
                # teacher_output = torch.sigmoid(teacher_output) > 0.5  # Apply threshold for binary predictions

            # Student outputs
            student_output = student_model(inputs)
            if student_output.shape != teacher_output.shape:
                student_output = F.interpolate(student_output, size=teacher_output.shape[2:], mode='bilinear', align_corners=False)
            student_output_binary = torch.sigmoid(student_output) > 0.5  # Apply threshold for binary predictions

            # Apply bilateral filtering if configured
            # if config.get("apply_bilateral_filter", False):
            #     student_output_binary = apply_bilateral_filter(student_output_binary)

            # Check if gt_labels has only three dimensions, and add a channel dimension if needed
            if gt_labels.dim() == 3:
                gt_labels = gt_labels.unsqueeze(1)  # Adds a channel dimension at index 1

            # Now you can safely apply the original slicing
            gt_labels = gt_labels[:, :1, :, :]

            # Compute hybrid loss using processed student and teacher predictions, and ground truth
            loss = criterion(student_output, student_output_binary.float(), teacher_output.float(), ground_truth=gt_labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(trainloader)
        print(f"Epoch [{epoch + 1}/{config['epochs']}], Loss: {epoch_loss:.4f}")

        # Save visualizations of inputs, student predictions, and teacher predictions
        save_visualization(inputs, student_output, teacher_output, output_dir, epoch, ground_truth=gt_labels)

        # Check if this epoch has the best loss and save the model if it does
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_weights = copy.deepcopy(student_model.state_dict())
            torch.save(best_model_weights, os.path.join(checkpoint_dir, 'best_student_model.pth'))
            print("Saved best model with loss:", best_loss)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping if patience counter exceeds limit
        if patience_counter >= config['patience']:
            print("Early stopping due to no improvement in loss.")
            break

    # Load best model weights
    student_model.load_state_dict(best_model_weights)
    print("Training completed.")

def save_visualization(inputs, student_output, teacher_output, output_dir, epoch, ground_truth=None):
    # Create subdirectories for each epoch and for each type of output
    epoch_dir = os.path.join(output_dir, f"epoch_{epoch + 1}")
    inputs_dir = os.path.join(epoch_dir, "inputs")
    student_dir = os.path.join(epoch_dir, "student")
    teacher_dir = os.path.join(epoch_dir, "teacher")
    binarized_dir = os.path.join(epoch_dir, "student_binarized")

    os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(student_dir, exist_ok=True)
    os.makedirs(teacher_dir, exist_ok=True)
    os.makedirs(binarized_dir, exist_ok=True)

    # Create ground truth directory if ground truth is provided
    if ground_truth is not None:
        gt_dir = os.path.join(epoch_dir, "ground_truth")
        os.makedirs(gt_dir, exist_ok=True)

    # Move tensors to CPU and convert to images
    inputs = inputs.squeeze().cpu()
    student_output = student_output.squeeze().cpu()
    teacher_output = teacher_output.squeeze().cpu()
    student_binarized = (student_output > 0.5).float()  # Apply threshold for binarized output
    if ground_truth is not None:
        ground_truth = ground_truth.squeeze().cpu()

    # Resize outputs to match the dimensions of inputs
    student_output = F.interpolate(student_output.unsqueeze(0), size=inputs.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)
    teacher_output = F.interpolate(teacher_output.unsqueeze(0), size=inputs.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)
    student_binarized = F.interpolate(student_binarized.unsqueeze(0), size=inputs.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)
    if ground_truth is not None:
        ground_truth = F.interpolate(ground_truth.unsqueeze(0), size=inputs.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)

    # Normalize and convert to PIL images
    to_pil = transforms.ToPILImage()
    inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min() + 1e-5)
    student_output = (student_output - student_output.min()) / (student_output.max() - student_output.min() + 1e-5)
    teacher_output = (teacher_output - teacher_output.min()) / (teacher_output.max() - teacher_output.min() + 1e-5)
    student_binarized = (student_binarized - student_binarized.min()) / (student_binarized.max() - student_binarized.min() + 1e-5)
    if ground_truth is not None:
        ground_truth = (ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min() + 1e-5)

    # Save images in the appropriate directories
    for i in range(inputs.shape[0]):  # Loop through each image in the batch
        input_image = to_pil(inputs[i])
        student_image = to_pil(student_output[i])
        teacher_image = to_pil(teacher_output[i])
        binarized_image = to_pil(student_binarized[i])

        input_image.save(os.path.join(inputs_dir, f"input_{i + 1}.png"))
        student_image.save(os.path.join(student_dir, f"student_{i + 1}.png"))
        teacher_image.save(os.path.join(teacher_dir, f"teacher_{i + 1}.png"))
        binarized_image.save(os.path.join(binarized_dir, f"student_binarized_{i + 1}.png"))

        # Save ground truth if provided
        if ground_truth is not None:
            gt_image = to_pil(ground_truth[i])
            gt_image.save(os.path.join(gt_dir, f"ground_truth_{i + 1}.png"))

    print(f"Saved images for epoch {epoch + 1} in {epoch_dir}.")