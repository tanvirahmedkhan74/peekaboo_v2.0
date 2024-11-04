import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from .distillation_loss import DistillationLoss
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


def save_visualization(inputs, student_output, teacher_output, output_dir, epoch):
    # Move tensors to CPU and convert to images
    inputs = inputs.squeeze().cpu()
    student_output = student_output.squeeze().cpu()
    teacher_output = teacher_output.squeeze().cpu()

    # Normalize and convert to PIL images
    to_pil = transforms.ToPILImage()

    # Ensure input images are normalized
    inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min() + 1e-5)
    student_output = (student_output - student_output.min()) / (student_output.max() - student_output.min() + 1e-5)
    teacher_output = (teacher_output - teacher_output.min()) / (teacher_output.max() - teacher_output.min() + 1e-5)

    # Save images
    for i in range(inputs.shape[0]):  # Loop through each image in the batch
        input_image = to_pil(inputs[i])
        student_image = to_pil(student_output[i])
        teacher_image = to_pil(teacher_output[i])

        input_image.save(os.path.join(output_dir, f"epoch_{epoch + 1}_input_{i + 1}.png"))
        student_image.save(os.path.join(output_dir, f"epoch_{epoch + 1}_student_{i + 1}.png"))
        teacher_image.save(os.path.join(output_dir, f"epoch_{epoch + 1}_teacher_{i + 1}.png"))

    print(f"Saved images for epoch {epoch + 1} in {output_dir}.")