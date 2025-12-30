import os
from pdf2image import convert_from_path

# Path to the folder containing  3fold, 4fold, 6fold PDF folders
data_dir = "tilings_dataset"

# Loop over each symmetry folder
for fold in ["3-fold", "4-fold", "6-fold", "other"]:
    fold_path = os.path.join(data_dir, fold)
    pdf_files = [f for f in os.listdir(fold_path) if f.lower().endswith(".pdf")]
    
    # Make an 'images' subfolder to store converted PNGs
    images_dir = os.path.join(fold_path, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(fold_path, pdf_file)
        # Convert PDF pages to images 
        pages = convert_from_path(pdf_path, dpi=300)  # dpi=300 for good quality
        for i, page in enumerate(pages):
            # Save each page as a PNG
            image_filename = os.path.splitext(pdf_file)[0] + f"_{i+1}.png"
            image_path = os.path.join(images_dir, image_filename)
            page.save(image_path, "PNG")
        print(f"Converted {pdf_file} -> {len(pages)} image(s)")
