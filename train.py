# Code for Peekaboo
# Author: Hasib Zunair
# Modified from https://github.com/valeoai/FOUND

"""Training code for Peekaboo"""

import os
import sys
import json
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from model import PeekabooModel
from evaluation.saliency import evaluate_saliency
from misc import batch_apply_bilateral_solver, set_seed, load_config, Logger

from datasets.datasets import build_dataset

from distillation.distillation_trainer import distillation_training, undeviating_distillation_training, hybrid_distillation_training
from models.student_base_model import StudentModel


def get_argparser():
    parser = argparse.ArgumentParser(
        description="Training of Peekaboo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--exp-name", type=str, default=None, help="Exp name.")
    parser.add_argument(
        "--log-dir", type=str, default="outputs", help="Logging and output directory."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Root directories of training and evaluation datasets.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/peekaboo_DUTS-TR.yaml",
        help="Path of config file.",
    )
    parser.add_argument(
        "--save-model-freq", type=int, default=250, help="Frequency of model saving."
    )
    parser.add_argument(
        "--visualization-freq",
        type=int,
        default=10,
        help="Frequency of prediction visualization in tensorboard.",
    )

    parser.add_argument(
        "--distillation", action="store_true", help="Use knowledge distillation for training."
    )

    args = parser.parse_args()
    return args


def train_model(
    model,
    config,
    dataset,
    dataset_dir,
    visualize_freq=10,
    save_model_freq=500,
    tensorboard_log_dir=None,
):

    # Diverse
    print(f"Data will be saved in {tensorboard_log_dir}")
    save_dir = tensorboard_log_dir
    if tensorboard_log_dir is not None:
        # Logging
        if not os.path.exists(tensorboard_log_dir):
            os.makedirs(tensorboard_log_dir)
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(tensorboard_log_dir)

    # Deconvolution, train only the decoder
    sigmoid = nn.Sigmoid()
    model.decoder.train()
    model.decoder.to("cuda")

    ################################################################################
    #                                                                              #
    #                      Setup loss, optimizer and scheduler                     #
    #                                                                              #
    ################################################################################

    criterion = nn.BCEWithLogitsLoss()
    criterion_mse = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=config.training["lr0"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.training["step_lr_size"],
        gamma=config.training["step_lr_gamma"],
    )

    ################################################################################
    #                                                                              #
    #                                Dataset                                       #
    #                                                                              #
    ################################################################################

    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.training["batch_size"], shuffle=True, num_workers=2
    )

    ################################################################################
    #                                                                              #
    #                                Training loop                                 #
    #                                                                              #
    ################################################################################

    n_iter = 0
    for epoch in range(config.training["nb_epochs"]):
        running_loss = 0.0
        tbar = tqdm(enumerate(trainloader, 0), leave=None)
        for i, data in tbar:

            # Get the inputs
            inputs, masked_inputs, _, input_nonorm, masked_input_nonorm, _, _ = data

            ######## For debug #######
            # def to_img(ten):
            #     #ten =(input_nonorm[0].permute(1,2,0).detach().cpu().numpy()+1)/2
            #     ten =(ten.permute(1,2,0).detach().cpu().numpy())
            #     ten=(ten*255).astype(np.uint8)
            #     #ten=cv2.cvtColor(ten,cv2.COLOR_RGB2BGR)
            #     return ten
            # import pdb; pdb.set_trace()
            # im = to_img(input_nonorm[0])
            # plt.imshow(im); plt.show()

            # Inputs and masked inputs
            inputs = inputs.to("cuda")
            masked_inputs = masked_inputs.to("cuda")

            # zero the parameter gradients
            optimizer.zero_grad()

            ################################################################################
            #                                                                              #
            #                                Unsupervised Segmenter                        #
            #                                                                              #
            ################################################################################

            # Get predictions
            preds = model(inputs)
            # Binarization
            preds_mask = (sigmoid(preds.detach()) > 0.5).float()
            # Apply bilateral solver
            preds_mask_bs, _ = batch_apply_bilateral_solver(data, preds_mask.detach())
            # Flatten
            flat_preds = preds.permute(0, 2, 3, 1).reshape(-1, 1)

            #### Compute unsupervised segmenter loss ####
            alpha = 1.5
            preds_bs_loss = alpha * criterion(
                flat_preds, preds_mask_bs.reshape(-1).float()[:, None]
            )
            print(preds_bs_loss)
            writer.add_scalar("Loss/L_seg", preds_bs_loss, n_iter)
            loss = preds_bs_loss

            ################################################################################
            #                                                                              #
            #                            Masked Feature Predictor (MFP)                    #
            #                                                                              #
            ################################################################################

            # Get predictions
            preds_mfp = model(masked_inputs)
            # Binarization
            preds_mask_mfp = (sigmoid(preds_mfp.detach()) > 0.5).float()
            # Apply bilateral solver
            preds_mask_mfp_bs, _ = batch_apply_bilateral_solver(
                data, preds_mask_mfp.detach()
            )
            # Flatten
            flat_preds_mfp = preds_mfp.permute(0, 2, 3, 1).reshape(-1, 1)

            #### Compute masked feature predictor loss ####
            beta = 1.0
            preds_bs_cb_loss = beta * criterion(
                flat_preds_mfp, preds_mask_mfp_bs.reshape(-1).float()[:, None]
            )
            writer.add_scalar("Loss/L_mfp", preds_bs_cb_loss, n_iter)
            loss += preds_bs_cb_loss

            ################################################################################
            #                                                                              #
            #                       Predictor Consistency Loss (PCL)                       #
            #                                                                              #
            ################################################################################

            gamma = 1.0
            task_sim_loss = gamma * criterion_mse(
                preds_mask_bs.reshape(-1).float()[:, None],
                preds_mask_mfp_bs.reshape(-1).float()[:, None],
            )
            writer.add_scalar("Loss/L_pcl", task_sim_loss, n_iter)
            loss += task_sim_loss

            ### Compute loss between soft masks and their binarized versions ####
            self_loss = criterion(flat_preds, preds_mask.reshape(-1).float()[:, None])

            self_loss = self_loss * 4.0
            loss += self_loss
            writer.add_scalar("Loss/L_regularization", self_loss, n_iter)

            ################################################################################
            #                                                                              #
            #                       Update weights and scheduler step                      #
            #                                                                              #
            ################################################################################

            loss.backward()
            optimizer.step()
            writer.add_scalar("Loss/total_loss", loss, n_iter)
            writer.add_scalar("params/lr", optimizer.param_groups[0]["lr"], n_iter)
            scheduler.step()

            ################################################################################
            #                                                                              #
            #                       Visualize predictions and show stats                   #
            #                                                                              #
            ################################################################################

            # Visualize predictions in tensorboard
            if n_iter % visualize_freq == 0:
                # images and predictions
                grid = torchvision.utils.make_grid(input_nonorm[:5])
                writer.add_image("training/images", grid, n_iter)
                p_grid = torchvision.utils.make_grid(preds_mask[:5])
                writer.add_image("training/preds", p_grid, n_iter)

                # masked images and predictions
                m_grid = torchvision.utils.make_grid(masked_input_nonorm[:5])
                writer.add_image("training/masked_images", m_grid, n_iter)
                mp_grid = torchvision.utils.make_grid(preds_mask_mfp[:5])
                writer.add_image("training/masked_preds", mp_grid, n_iter)
            # Statistics
            running_loss += loss.item()
            tbar.set_description(
                f"{dataset.name}| train | iter {n_iter} | loss: ({running_loss / (i + 1):.3f}) "
            )

            ################################################################################
            #                                                                              #
            #                           Save model and evaluate                            #
            #                                                                              #
            ################################################################################

            # Save model
            if n_iter % save_model_freq == 0 and n_iter > 0:
                model.decoder_save_weights(save_dir, n_iter)

            # Evaluation
            if n_iter % config.evaluation["freq"] == 0 and n_iter > 0:
                for dataset_eval_name in config.evaluation["datasets"]:
                    val_dataset = build_dataset(
                        root_dir=dataset_dir,
                        dataset_name=dataset_eval_name,
                        for_eval=True,
                        dataset_set=None,
                    )
                    evaluate_saliency(
                        val_dataset, model=model, n_iter=n_iter, writer=writer
                    )

            if n_iter == config.training["max_iter"]:
                model.decoder_save_weights(save_dir, n_iter)
                print("\n----" "\nTraining done.")
                writer.close()
                return model

            n_iter += 1

        print(f"##### Number of epoch is {epoch} and n_iter is {n_iter} #####")

    # Save model
    model.decoder_save_weights(save_dir, n_iter)
    print("\n----" "\nTraining done.")
    writer.close()
    return model


def test_and_save_predictions(model, dataset, output_dir, model_type="student", num_images=5):
    # Set up directories for saving predictions and inputs
    output_dir = os.path.join(output_dir, f"{model_type}_predictions")
    inputs_dir = os.path.join(output_dir, "inputs")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(inputs_dir, exist_ok=True)

    # Load a small portion of the dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    sigmoid = nn.Sigmoid()
    to_pil = transforms.ToPILImage()

    # Prepare the model for evaluation
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if idx >= num_images:
                break

            # Unpack data (assuming only inputs are used here)
            inputs, _, _, _, _, _, _ = data
            inputs = inputs.to(device)

            # Normalize and save the input image
            input_image = inputs.squeeze().cpu()
            input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min() + 1e-5)
            input_image = to_pil(input_image)
            input_image.save(os.path.join(inputs_dir, f"input_image_{idx + 1}.png"))

            # Generate and normalize prediction
            preds = model(inputs)
            preds = sigmoid(preds).squeeze().cpu()
            preds = (preds - preds.min()) / (preds.max() - preds.min() + 1e-5)

            # Convert prediction to a PIL image and save as PNG
            pred_image = to_pil(preds)
            pred_image.save(os.path.join(output_dir, f"{model_type}_prediction_{idx + 1}.png"))

    print(f"{model_type.capitalize()} predictions and input images saved to {output_dir} and {inputs_dir}")


def main():
    ########## Get arguments ##########
    args = get_argparser()

    ########## Setup ##########

    # Load config yaml file
    config, config_ = load_config(args.config)

    # Experiment name
    exp_name = "{}-{}{}".format(
        config.training["dataset"], config.model["arch"], config.model["patch_size"]
    )
    if args.exp_name is not None:
        exp_name = f"{args.exp_name}-{exp_name}"

    # Log dir
    output_dir = os.path.join(args.log_dir, exp_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save config
    with open(f"{output_dir}/config.json", "w") as f:
        print(f"Config saved in {output_dir}/config.json.")
        json.dump(args.__dict__, f)

    # Log output to file
    sys.stdout = Logger(os.path.join(output_dir, "log_train.txt"))
    print("=========================\nConfigs:{}\n=========================")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print(
        "Hyperparameters from config file: "
        + ", ".join(f"{k}={v}" for k, v in config_.items())
    )
    print("=========================")

    ########## Reproducibility ##########
    set_seed(config.training["seed"])

    ########## Build training set ##########
    dataset = build_dataset(
        root_dir=args.dataset_dir,
        dataset_name=config.training["dataset"],
        dataset_set=config.training["dataset_set"],
        config=config,
        for_eval=False,
    )

    ########## Define Models ##########
    if args.distillation:
        # Knowledge distillation setup with teacher and student models
        teacher_model = PeekabooModel(
            vit_model=config.model["pre_training"],
            vit_arch=config.model["arch"],
            vit_patch_size=config.model["patch_size"],
            enc_type_feats=config.peekaboo["feats"],
        )
        teacher_model.decoder_load_weights(config.training["teacher_weights_path"])
        teacher_model.eval()  # Set teacher model to evaluation mode

        student_model = StudentModel()  # Initialize student model

        ########## Training with Knowledge Distillation ##########
        print("Starting distillation training...")
        undeviating_distillation_training(
            teacher_model=teacher_model,
            student_model=student_model,
            trainloader=torch.utils.data.DataLoader(
                dataset, batch_size=config.distillation["batch_size"], shuffle=True, num_workers=2
            ),
            config={
                "learning_rate": config.distillation["lr1"],
                "epochs": config.distillation["nb_epochs"],
                "alpha": config.distillation["alpha"],
                "temperature": config.distillation["temperature"],
                "patience": config.distillation["patience"],
            }
        )
        student_model_path = os.path.join(output_dir, "student_model_final.pth")
        torch.save(student_model.state_dict(), student_model_path)
        print("Distillation training completed.")

        # Run testing and save predictions for both teacher and student models
        print("Saving sample predictions for teacher model...")
        test_and_save_predictions(teacher_model, dataset, output_dir, model_type="teacher")

        print("Saving sample predictions for student model...")
        test_and_save_predictions(student_model, dataset, output_dir, model_type="student")

    else:
        # Standard training for Peekaboo model
        model = PeekabooModel(
            vit_model=config.model["pre_training"],
            vit_arch=config.model["arch"],
            vit_patch_size=config.model["patch_size"],
            enc_type_feats=config.peekaboo["feats"],
        )

        ########## Standard Training and Evaluation ##########
        print(f"\nStarted training on {dataset.name} [tensorboard dir: {output_dir}]")
        model = train_model(
            model=model,
            config=config,
            dataset=dataset,
            dataset_dir=args.dataset_dir,
            tensorboard_log_dir=output_dir,
            visualize_freq=args.visualization_freq,
            save_model_freq=args.save_model_freq,
        )
        print(f"\nTraining done, Peekaboo model saved in {output_dir}.")

if __name__ == "__main__":
    main()