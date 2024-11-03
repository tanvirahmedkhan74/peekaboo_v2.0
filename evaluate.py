import argparse
import torch
from model import PeekabooModel
from models.student_base_model import StudentModel
from misc import load_config
from datasets.datasets import build_dataset
from evaluation.saliency import evaluate_saliency, student_evaluation_saliency
from evaluation.uod import evaluation_unsupervised_object_discovery

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation of Peekaboo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--eval-type", type=str, choices=["saliency", "uod"], help="Evaluation type."
    )
    parser.add_argument(
        "--dataset-eval",
        type=str,
        choices=["ECSSD", "DUT-OMRON", "DUTS-TEST", "VOC07", "VOC12", "COCO20k"],
        help="Name of evaluation dataset.",
    )
    parser.add_argument(
        "--dataset-set-eval", type=str, default=None, help="Set of the dataset."
    )
    parser.add_argument(
        "--apply-bilateral", action="store_true", help="Use bilateral solver."
    )
    parser.add_argument(
        "--evaluation-mode",
        type=str,
        default="multi",
        choices=["single", "multi"],
        help="Type of evaluation.",
    )
    parser.add_argument(
        "--model-weights",
        type=str,
        default="data/weights/decoder_weights.pt",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/peekaboo_DUTS-TR.yaml",
    )
    parser.add_argument(
        "--student-model", action="store_true", help="Evaluate the student model instead of the teacher model."
    )
    args = parser.parse_args()
    print(args.__dict__)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configuration
    config, _ = load_config(args.config)

    # Load the appropriate model based on the --student-model flag
    if args.student_model:
        model = StudentModel()  # Load student model
    else:
        model = PeekabooModel(
            vit_model=config.model["pre_training"],
            vit_arch=config.model["arch"],
            vit_patch_size=config.model["patch_size"],
            enc_type_feats=config.peekaboo["feats"],
        )

    # Move the model to the device
    model = model.to(device)

    # Load weights
    model.decoder_load_weights(args.model_weights)
    model.eval()
    print(f"Model {args.model_weights} loaded correctly.")

    # Build the validation dataset
    val_dataset = build_dataset(
        root_dir=args.dataset_dir,
        dataset_name=args.dataset_eval,
        dataset_set=args.dataset_set_eval,
        for_eval=True,
        evaluation_type=args.eval_type,
    )
    print(f"\nBuilding dataset {val_dataset.name} (#{len(val_dataset)} images)")

    # Start evaluation
    print(f"\nStarted evaluation on {val_dataset.name}")
    if args.eval_type == "saliency":
        if args.student_model:
            # Use student evaluation function for saliency
            student_evaluation_saliency(
                dataset=val_dataset,
                student_model=model,
                evaluation_mode=args.evaluation_mode,
                apply_bilateral=args.apply_bilateral,
            )
        else:
            # Use teacher evaluation function for saliency
            evaluate_saliency(
                dataset=val_dataset,
                model=model,
                evaluation_mode=args.evaluation_mode,
                apply_bilateral=args.apply_bilateral,
            )
    elif args.eval_type == "uod":
        if args.apply_bilateral:
            raise ValueError("Bilateral solver is not implemented for unsupervised object discovery.")
        # Use UOD evaluation for either model (assuming same function applies)
        evaluation_unsupervised_object_discovery(
            dataset=val_dataset,
            model=model,
            evaluation_mode=args.evaluation_mode,
        )
    else:
        raise ValueError("Other evaluation method not implemented.")