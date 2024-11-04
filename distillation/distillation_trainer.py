import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from .distillation_loss import DistillationLoss
from .undeviating_distillation_loss import UndeviatingDistillationLoss
from .hybrid_distillation_loss import HybridDistillationLoss
from torchvision import transforms


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
    # Set models
    teacher_model.eval()  # Teacher in eval mode
    student_model.train()  # Student in training mode

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model.to(device)
    student_model.to(device)

    # Define loss and optimizer
    criterion = UndeviatingDistillationLoss(alpha=config["alpha"], temperature=config["temperature"])
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
            inputs = batch[0]  # Only using inputs since there are no labels
            inputs = inputs.to(device)

            # Get teacher and student outputs
            with torch.no_grad():
                teacher_output = teacher_model(inputs)
            student_output = student_model(inputs)

            # Resize student output to match teacher output dimensions if they don't match
            if student_output.shape != teacher_output.shape:
                student_output = F.interpolate(student_output, size=teacher_output.shape[2:], mode='bilinear', align_corners=False)

            # Compute loss and backpropagate
            loss = criterion(student_output, teacher_output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{config['epochs']}], Loss: {running_loss / len(trainloader):.4f}")

        # Save images for visualization
        save_visualization(inputs, student_output, teacher_output, output_dir, epoch)

    print("Training completed.")

def hybrid_distillation_training(teacher_model, student_model, trainloader, config):
    # Set models
    teacher_model.eval()  # Teacher in eval mode
    student_model.train()  # Student in training mode

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model.to(device)
    student_model.to(device)

    # Define hybrid loss and optimizer
    criterion = HybridDistillationLoss(
        alpha=config["alpha"],
        temperature=config["temperature"],
        beta=config.get("beta", 0.5)  # Set beta for supervised loss weight
    )
    optimizer = optim.Adam(student_model.parameters(), lr=config['learning_rate'])

    # Create output directory for predictions
    output_dir = './outputs/KD_epoch_out'
    os.makedirs(output_dir, exist_ok=True)

    # Training loop
    for epoch in range(config['epochs']):
        running_loss = 0.0
        for batch in tqdm(trainloader):
            # Extract inputs and ground truth labels as done in saliency.py
            inputs, _, _, _, _, gt_labels, _ = batch  # gt_labels is positioned as in saliency.py
            inputs = inputs.to(device)
            gt_labels = gt_labels.to(device).float()  # Ensure ground truth is in the correct format

            # teacher and student outputs
            with torch.no_grad():
                teacher_output = teacher_model(inputs)
            student_output = student_model(inputs)

            # Resize student output to match teacher output dimensions if needed
            if student_output.shape != teacher_output.shape:
                student_output = F.interpolate(student_output, size=teacher_output.shape[2:], mode='bilinear', align_corners=False)

            # Ensuring gt_labels have one channel and resize to match output dimensions
            gt_labels = gt_labels[:, :1, :, :]  # only one channel if gt_labels have multiple channels
            gt_labels = F.interpolate(gt_labels, size=student_output.shape[2:], mode='bilinear', align_corners=False)

            # Compute hybrid loss with distillation and supervised components
            loss = criterion(student_output, teacher_output, ground_truth=gt_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{config['epochs']}], Loss: {running_loss / len(trainloader):.4f}")

        # Save images for visualization
        save_visualization(inputs, student_output, teacher_output, output_dir, epoch)

    print("Training completed.")

def save_visualization(inputs, student_output, teacher_output, output_dir, epoch):
    # Create subdirectories for each epoch and for each type of output
    epoch_dir = os.path.join(output_dir, f"epoch_{epoch + 1}")
    inputs_dir = os.path.join(epoch_dir, "inputs")
    student_dir = os.path.join(epoch_dir, "student")
    teacher_dir = os.path.join(epoch_dir, "teacher")

    os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(student_dir, exist_ok=True)
    os.makedirs(teacher_dir, exist_ok=True)

    # Move tensors to CPU and convert to images
    inputs = inputs.squeeze().cpu()
    student_output = student_output.squeeze().cpu()
    teacher_output = teacher_output.squeeze().cpu()

    # Normalize and convert to PIL images
    to_pil = transforms.ToPILImage()
    inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min() + 1e-5)
    student_output = (student_output - student_output.min()) / (student_output.max() - student_output.min() + 1e-5)
    teacher_output = (teacher_output - teacher_output.min()) / (teacher_output.max() - teacher_output.min() + 1e-5)

    # Save images in the appropriate directories
    for i in range(inputs.shape[0]):  # Loop through each image in the batch
        input_image = to_pil(inputs[i])
        student_image = to_pil(student_output[i])
        teacher_image = to_pil(teacher_output[i])

        input_image.save(os.path.join(inputs_dir, f"input_{i + 1}.png"))
        student_image.save(os.path.join(student_dir, f"student_{i + 1}.png"))
        teacher_image.save(os.path.join(teacher_dir, f"teacher_{i + 1}.png"))

    print(f"Saved images for epoch {epoch + 1} in {epoch_dir}.")