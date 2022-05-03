# COCO: trainval2017, val2017
data_dir='data'
dataset_dir='coco'

mkdir -p $data_dir  # if not exsit, create a folder

# annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -o annotations_trainval2017.zip -d $data_dir/$dataset_dir/
mv $data_dir/$dataset_dir/annotations/ $data_dir/$dataset_dir/ann_trainval2017/

# images
wget http://images.cocodataset.org/zips/val2017.zip
unzip -q -o ./val2017.zip -d $data_dir/$dataset_dir/ # no verbose

wget http://images.cocodataset.org/zips/train2017.zip
unzip -q -o ./train2017.zip -d $data_dir/$dataset_dir/ # no verbose

rm *.zip*