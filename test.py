import os
import torch
import torchvision.transforms as transforms
from model import PeekabooModel
from models.student_base_model import StudentModel
from datasets.datasets import build_dataset
from misc import load_config


def save_predictions(input_image, student_preds, teacher_preds, output_dir, idx):
    """Save the input image, student and teacher predictions."""

    # Ensure prediction values are in the 0-1 range
    student_preds = (student_preds - student_preds.min()) / (student_preds.max() - student_preds.min() + 1e-5)
    teacher_preds = (teacher_preds - teacher_preds.min()) / (teacher_preds.max() - teacher_preds.min() + 1e-5)

    # Convert tensors to PIL images
    input_pil = transforms.ToPILImage()(input_image)
    student_pred_pil = transforms.ToPILImage()(student_preds)
    teacher_pred_pil = transforms.ToPILImage()(teacher_preds)

    # Save images
    input_pil.save(os.path.join(output_dir, f"input_image_{idx + 1}.png"))
    student_pred_pil.save(os.path.join(output_dir, f"student_prediction_{idx + 1}.png"))
    teacher_pred_pil.save(os.path.join(output_dir, f"teacher_prediction_{idx + 1}.png"))

    print(f"Saved input, student, and teacher predictions for image {idx + 1}")


def test_and_save_predictions(teacher_model, student_model, dataset, output_dir, num_images=5):
    # Set up directories for saving predictions and inputs
    teacher_output_dir = os.path.join(output_dir, "teacher_predictions")
    student_output_dir = os.path.join(output_dir, "student_predictions")
    inputs_dir = os.path.join(output_dir, "inputs")
    os.makedirs(teacher_output_dir, exist_ok=True)
    os.makedirs(student_output_dir, exist_ok=True)
    os.makedirs(inputs_dir, exist_ok=True)

    # Load a small portion of the dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    sigmoid = nn.Sigmoid()

    # Run predictions on a subset of the dataset
    teacher_model.eval()
    student_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model.to(device)
    student_model.to(device)

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if idx >= num_images:
                break

            inputs, _, _, _, _, _, _ = data
            inputs = inputs.to(device)

            # Save the input image
            input_image = inputs.squeeze().cpu()
            input_image = transforms.ToPILImage()(input_image)
            input_image.save(os.path.join(inputs_dir, f"input_image_{idx + 1}.png"))

            # Teacher's prediction
            teacher_preds = teacher_model(inputs)
            teacher_preds = sigmoid(teacher_preds).squeeze().cpu()
            teacher_pred_image = transforms.ToPILImage()(teacher_preds)
            teacher_pred_image.save(os.path.join(teacher_output_dir, f"teacher_prediction_{idx + 1}.png"))

            # Student's prediction
            student_preds = student_model(inputs)
            student_preds = sigmoid(student_preds).squeeze().cpu()
            student_pred_image = transforms.ToPILImage()(student_preds)
            student_pred_image.save(os.path.join(student_output_dir, f"student_prediction_{idx + 1}.png"))

    print(f"Predictions saved in {teacher_output_dir}, {student_output_dir}, and {inputs_dir}")


def main():
    # Load configuration
    config_path = "configs/peekaboo_DUTS-TR.yaml"
    config, _ = load_config(config_path)

    # Specify output directory for predictions
    output_dir = "./outputs/test_predictions"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the student and teacher models
    teacher_model = PeekabooModel(
        vit_model=config.model["pre_training"],
        vit_arch=config.model["arch"],
        vit_patch_size=config.model["patch_size"],
        enc_type_feats=config.peekaboo["feats"],
    )
    teacher_model.decoder_load_weights("./data/weights/peekaboo_decoder_weights_niter500.pt")

    student_model = StudentModel()
    student_model.decoder_load_weights('./outputs/peekaboo-DUTS-TR-vit_small8/student_model_final.pth')

    # Build the test dataset
    dataset = build_dataset(
        root_dir='./datasets_local/',
        dataset_name=config.training["dataset"],
        dataset_set=config.training["dataset_set"],
        config=config,
        for_eval=True,
    )

    # Run testing and save predictions
    test_and_save_predictions(student_model, teacher_model, dataset, output_dir, num_images=10)


if __name__ == "__main__":
    main()