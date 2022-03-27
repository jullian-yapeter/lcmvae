mkdir -p data  # if not exsit, create a folder

# COCO: trainval2017, val2017
data_dir='data'
dataset_dir='coco'

# caption
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -o annotations_trainval2017.zip -d $data_dir/$dataset_dir/
mv $data_dir/$dataset_dir/annotations/ $data_dir/$dataset_dir/ann_trainval2017/

# images
wget http://images.cocodataset.org/zips/val2017.zip
unzip -q -o ./val2017.zip -d $data_dir/$dataset_dir/ # no verbose

wget http://images.cocodataset.org/zips/train2017.zip
unzip -q -o ./train2017.zip -d $data_dir/$dataset_dir/ # no verbose

rm *.zip*


# TODO: Creating a Sufficiently Large Dataset
# wget http://images.cocodataset.org/zips/train2014.zip
# wget http://images.cocodataset.org/zips/val2014.zip
# wget http://images.cocodataset.org/zips/test2014.zip
# laion-400: dataset for CLIP https://laion.ai/laion-400-open-dataset/