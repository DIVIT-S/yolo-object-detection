import glob
import os
import shutil

# Target directory: ~/YOLO/dataset/
target_base = os.path.expanduser('~/YOLO/dataset')
downloads_dir = os.path.expanduser('~/Downloads')

# Exact folder names from your Downloads screenshot
sources = [
    {
        'folder': 'YOLO Object Detection v1i',
        'new_id': 0,
        'prefix': 'carton',
    },
    {'folder': 'Cement Bags YOLO v8', 'new_id': 1, 'prefix': 'cement'},
    {'folder': 'Wood Log Dataset v2', 'new_id': 2, 'prefix': 'wood'},
]

for src in sources:
  src_path = os.path.join(downloads_dir, src['folder'])

  # Map 'valid' from Downloads to 'val' in your project
  for src_split in ['train', 'valid', 'test']:
    dst_split = 'val' if src_split == 'valid' else src_split

    src_img_dir = os.path.join(src_path, src_split, 'images')
    src_lbl_dir = os.path.join(src_path, src_split, 'labels')

    if not os.path.exists(src_img_dir):
      print(f"Skipping missing directory: {src_img_dir}")
      continue

    # Ensure target output directories exist
    dst_img_dir = os.path.join(target_base, 'images', dst_split)
    dst_lbl_dir = os.path.join(target_base, 'labels', dst_split)
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    # Copy images and remap label IDs
    for img_path in glob.glob(os.path.join(src_img_dir, '*.*')):
      fname = os.path.basename(img_path)
      stem, _ = os.path.splitext(fname)

      new_img_name = f"{src['prefix']}_{fname}"
      new_lbl_name = f"{src['prefix']}_{stem}.txt"

      dst_img_path = os.path.join(dst_img_dir, new_img_name)
      dst_lbl_path = os.path.join(dst_lbl_dir, new_lbl_name)

      # Copy Image
      shutil.copy(img_path, dst_img_path)

      # Process and remap label file
      src_txt_path = os.path.join(src_lbl_dir, f'{stem}.txt')
      if os.path.exists(src_txt_path):
        with open(src_txt_path, 'r') as f:
          lines = f.readlines()

        remapped_lines = []
        for line in lines:
          parts = line.strip().split()
          if parts:
            parts[0] = str(src['new_id'])  # Force Class ID (0, 1, or 2)
            remapped_lines.append(' '.join(parts) + '\n')

        with open(dst_lbl_path, 'w') as f:
          f.writelines(remapped_lines)

print('SUCCESS: All 3 datasets merged into ~/YOLO/dataset!')