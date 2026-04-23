import os, shutil

def backup_src_files(base_folder):
    target_folder = base_folder + "/src/"
    os.mkdir(target_folder)
    for file_name in ["main.py", "models.py", "resnet18.py", "vgg.py", "small_vit.py", "cnn1d.py"]:
        shutil.copy(file_name, target_folder + file_name)
    shutil.copytree("utils", target_folder + "utils/")