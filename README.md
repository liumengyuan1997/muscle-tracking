# An Efficient Approach for Muscle Segmentation and 3D Reconstruction Using Keypoint Tracking in MRI Scans

[**arXiv Paper**](https://arxiv.org/pdf/2507.08690) | [**Muscle Labeling Procedure**](https://docs.google.com/document/d/1CHVpKfoM_PgN9bEd-YNyCINFiWcAiY5YIaH83aISVK4/edit?usp=sharing)

Magnetic resonance imaging (MRI) enables non-invasive, high-resolution analysis of muscle structures. However, automated segmentation remains limited by high computational costs, reliance on large training datasets, and reduced accuracy when segmenting smaller muscles.  

Convolutional neural network (CNN)-based methods, while powerful, often suffer from:  
- Substantial computational overhead  
- Limited generalizability across diverse populations  
- Poor interpretability  

This study proposes a **training-free segmentation approach** using keypoint tracking with **Lucas–Kanade optical flow**, incorporating two keypoint selection methodologies: (1) manual selection based on visual observation and (2) semi-automatic selection using **two-dimensional wavelet transform (2D DWT)**.

**Performance**  
- Achieves a mean Dice similarity coefficient (DSC) ranging from **0.6 to 0.7**, depending on the keypoint selection strategy  
- Performs comparably to state-of-the-art CNN-based models  
- Substantially reduces computational demands  
- Enhances interpretability  

This scalable framework presents a robust and explainable alternative for muscle segmentation in both clinical and research applications.  

---

## Usage

**Dataset**  
- Data samples are available in the `data` folder.  

**Keypoints Selection Method**  
| Method                     | Tracking Employing             | Script                              |
|----------------------------|------------------|--------------------------------------|
| **Manual Keypoint Selection** | Initial selection | `LKopflow_manual_select.py`         |
|                            | Reselection      | `LKopflow_manual_reselect.py`       |
| **Semi-Automatic Keypoint Selection using 2D DWT** | Initial selection | `wavelet_tracking.py`              |
|                            | Reselection      | `wavelet_tracking_reselect.py`      |

