import os
import pandas as pd
from utils import evaluate_single_image

def evaluate_directory(output_dir, dataset_dir):
    """Evaluate all images in directory and return metrics as dataframe."""
    results = []

    # Get list of image_ids based on subdirectory names in outputs
    image_ids = [name for name in os.listdir(os.path.join(output_dir, "Segmentation", "valid"))
                 if os.path.isdir(os.path.join(output_dir, "Segmentation", "valid", name))]

    for image_id in image_ids:
        pred_path = os.path.join(output_dir, "Segmentation", "valid", image_id, "mask.jpg")
        true_path = os.path.join(dataset_dir, "Segmentation", "masks", "valid", f"{image_id}.png")
        
        if os.path.exists(pred_path) and os.path.exists(true_path):
            accuracy, precision, recall, f1 = evaluate_single_image(pred_path, true_path)
            results.append({"image_id": image_id, "accuracy": accuracy, "precision": precision,
                            "recall": recall, "f1": f1})
            print(f"Evaluated image {image_id}: accuracy={accuracy:.4f}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")
    
    return pd.DataFrame(results)


if __name__ == '__main__':
    output_dir = "../outputs"
    dataset_dir = "../datasets"

    df = evaluate_directory(output_dir, dataset_dir)
    df.to_csv('../results/GroundedSAM.csv', index=False)

    print("Evaluation results saved to '../results/GroundedSAM.csv'")
