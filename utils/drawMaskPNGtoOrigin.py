import cv2
import os

# 文件夹路径
image_folder = "/Users/liumengyuan/Downloads/muscle_data/06S1_axial_Original"
mask_folder = "/Users/liumengyuan/Downloads/muscle_data/06S1_axial_Masks"
output_folder = "06S1_axial_image_mask_overlay"

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 遍历图像文件
for filename in os.listdir(image_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        name, ext = os.path.splitext(filename)
        mask_filename = f"{name}_mask{ext}"
        image_path = os.path.join(image_folder, filename)
        mask_path = os.path.join(mask_folder, mask_filename)
        output_path = os.path.join(output_folder, filename)

        # 读取图片和 mask
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"跳过：{filename}")
            continue

        # 二值化 mask
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # 查找轮廓
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_layer = image.copy()
        contour_layer[:] = (0, 0, 0)

        # 在图像上画轮廓
        # overlay = image.copy()
        cv2.drawContours(contour_layer, contours, -1, (0, 0, 255), 1)  # 红色边缘

        overlay = cv2.addWeighted(image, 1.0, contour_layer, 0.4, 0)
        # 保存结果
        cv2.imwrite(output_path, overlay)

print("✅ 完成 mask 边缘 overlay！")
