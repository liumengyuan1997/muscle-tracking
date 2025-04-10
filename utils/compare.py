import os
import numpy as np
import cv2

def dice_score(mask1, mask2):
    """
    Calculate the Dice Similarity Coefficient (DSC) between two binary masks.
    """
    intersection = np.sum(mask1 * mask2)
    total = np.sum(mask1) + np.sum(mask2)
    if total == 0:  # Handle the edge case where both masks are empty
        return 1.0 if intersection == 0 else 0.0
    return 2.0 * intersection / total

def calculate_dice_scores(folder_gt, folder_pred):
    """
    Calculate the Dice score for all masks in two folders.
    
    Parameters:
    - folder_gt: Path to the folder containing ground truth masks.
    - folder_pred: Path to the folder containing predicted masks.

    Returns:
    - A dictionary with filenames and their corresponding Dice scores.
    """
    # Get sorted lists of files
    gt_files = sorted([f for f in os.listdir(folder_gt) if f.endswith('.png')])
    pred_files = sorted([f for f in os.listdir(folder_pred) if f.endswith('.png')])
    
    dice_scores = {}

    for gt_file, pred_file in zip(gt_files, pred_files):
        # Load the binary masks
        gt_path = os.path.join(folder_gt, gt_file)
        pred_path = os.path.join(folder_pred, pred_file)

        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        # Ensure binary masks (values 0 and 255), normalize to 0 and 1
        gt_mask = (gt_mask > 127).astype(np.uint8)
        pred_mask = (pred_mask > 127).astype(np.uint8)

        # Calculate Dice score
        score = dice_score(gt_mask, pred_mask)
        dice_scores[gt_file] = score

        print(f"Dice score for {gt_file}: {score:.4f}")

    return dice_scores

# Example usage
folder_ground_truth = "/Users/liumengyuan/Downloads/muscle_data/06S1_axial_Masks"  # Replace with your ground truth folder path
folder_predictions = "/Users/liumengyuan/Desktop/muscle_seg/muscle_tracking/mask_wavelet"    # Replace with your prediction folder path

dice_scores = calculate_dice_scores(folder_ground_truth, folder_predictions)
# print("dice scores:", dice_scores)
# Specify the file path
output_file = "dice_scores_original_wavelet.txt"

# Write scores to the file
with open(output_file, "w") as file:
    for filename, score in dice_scores.items():
        file.write(f"{filename}\t{score:.4f}\n")

print(f"Dice scores saved to {output_file}")

# Average Dice Score
average_dice = sum(dice_scores.values()) / len(dice_scores)
print(f"Average Dice Score: {average_dice:.4f}")
