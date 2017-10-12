#!/usr/bin/python3
import os
import shutil

def _arrange_flower_folders(dataset_dir,work_dir,label_file):
    flower_root = os.path.join(dataset_dir, work_dir)
    if not os.path.exists(flower_root):  
        os.mkdir(flower_root)
    pic_id=1    
    for pic_label in open(os.path.join(dataset_dir, label_file)):
        pic_label_int=int(pic_label)
        pic_label_int-=1
        pic_label_str=str(pic_label_int)
        print("pic_id=",pic_id, "pic_label_str=",pic_label_str)

        filename="image_"+str(pic_id).zfill(5)+".jpg"
        flower_from = os.path.join(os.path.join(dataset_dir,'jpg'),filename)
    
        flower_dir = os.path.join(flower_root, 'flower_'+pic_label_str.strip().zfill(5))
        if not os.path.exists(flower_dir):
            os.mkdir(flower_dir)
        print("flower_from=",flower_from)
        shutil.copy(flower_from,flower_dir)
        print(filename)
        pic_id+=1

if __name__=="__main__":
    dataset_dir="/home/lzlu/flower102" 
    _arrange_flower_folders(dataset_dir,'flower_photos_xxx','flower102_labels.txt')
    
