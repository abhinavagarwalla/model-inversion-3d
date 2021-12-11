python invert_classification.py --log_dir=pointnet2_ssg_wo_normals --class_to_invert=$1 --output_dir=./output_no_reg/
python invert_classification.py --log_dir=pointnet2_ssg_wo_normals --class_to_invert=$1 --output_dir=./output_reg/ --regularize
# python invert_classification.py --log_dir=pointnet2_ssg_wo_normals --class_to_invert=6 --output_dir=./output_no_reg/
# python invert_classification.py --log_dir=pointnet2_ssg_wo_normals --class_to_invert=7 --output_dir=./output_no_reg/
# python invert_classification.py --log_dir=pointnet2_ssg_wo_normals --class_to_invert=8 --output_dir=./output_no_reg/