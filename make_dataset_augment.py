import shutil
import sys
import os
from PIL import Image

degrees = [0, 90, 180, 270]
data_path_read = sys.argv[1]
data_path_write = sys.argv[2]

for alphabeta in os.listdir(data_path_read):
    alphabeta_path = os.path.join(data_path_read, alphabeta)
    path_write1 = data_path_write[:-2] + '-' + alphabeta
    for charactor in os.listdir(alphabeta_path):
        charactor_path = os.path.join(alphabeta_path, charactor)
        path_write2 = path_write1 + '-' + charactor
        for deg in degrees:
            to_write = os.path.join(data_path_write, path_write2 + '-' + str(deg))
            os.makedirs(to_write)
            for drawer in os.listdir(charactor_path):
                drawer_path = os.path.join(charactor_path, drawer)
                img = Image.open(drawer_path)
                # shutil.copyfile(drawer_path, os.path.join(data_path_write, path_write2, drawer))
                img = img.rotate(deg)
                img.save(os.path.join(to_write, drawer))
