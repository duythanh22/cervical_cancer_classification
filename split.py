import os
import shutil
import random

# Đường dẫn gốc chứa dữ liệu
source_path = r"F:\THÀNH\IAST\Original_Data\Split_data\Data"
# Đường dẫn lưu dữ liệu mẫu
destination_path = r"F:\THÀNH\IAST\Original_Data\Split_data\Sample_data"

# Tạo các thư mục đích nếu chưa tồn tại
if not os.path.exists(destination_path):
    os.makedirs(destination_path)

# Các thư mục con: Train, Valid, Test
subfolders = ['Train', 'Valid', 'Test']

# Lặp qua từng thư mục con
for folder in subfolders:
    source_folder = os.path.join(source_path, folder)
    dest_folder = os.path.join(destination_path, folder)

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Duyệt qua từng nhãn bên trong
    for label in os.listdir(source_folder):
        source_label_folder = os.path.join(source_folder, label)
        dest_label_folder = os.path.join(dest_folder, label)

        # Tạo thư mục nhãn đích
        if not os.path.exists(dest_label_folder):
            os.makedirs(dest_label_folder)

        # Danh sách tất cả các tệp trong thư mục nhãn
        files = os.listdir(source_label_folder)

        # Chọn ngẫu nhiên 10% tệp
        sample_files = random.sample(files, int(0.03 * len(files)))

        # Sao chép các tệp vào thư mục mẫu
        for file in sample_files:
            source_file = os.path.join(source_label_folder, file)
            dest_file = os.path.join(dest_label_folder, file)
            shutil.copy(source_file, dest_file)

print("Hoàn thành sao chép dữ liệu mẫu.")
