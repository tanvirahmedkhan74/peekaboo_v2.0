import torch
import torch.optim as optim
from tqdm import tqdm
from .distillation_loss import DistillationLoss

def distillation_training(teacher_model, student_model, trainloader, config):
    # Set models
    teacher_model.eval()  # Teacher in eval mode
    student_model.train()  # Student in training mode

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model.to(device)
    student_model.to(device)

    # Define loss and optimizer
    criterion = DistillationLoss(alpha=0.5, temperature=3.0)
    optimizer = optim.Adam(student_model.parameters(), lr=config['learning_rate'])

    # Training loop
    for epoch in range(config['epochs']):
        running_loss = 0.0
        for inputs, labels in tqdm(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Get teacher and student outputs
            with torch.no_grad():
                teacher_output = teacher_model(inputs)
            student_output = student_model(inputs)

            # Compute loss and backpropagate
            loss = criterion(student_output, teacher_output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {running_loss/len(trainloader):.4f}")
    print("Training completed.")
