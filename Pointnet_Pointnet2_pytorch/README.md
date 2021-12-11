# 3D Point Cloud Inversion

## Requirements

The dataset setup and models is taken from the repository: [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch).

Additionally, please install the requirements specific to inversion and our environment using:  `pip install -r requirements.txt`.

## Usage

Use `sh launch_no_reg.sh <class_id_to_invert>`, where '<class_id_to_invert>' is the index from the class list below.

For example, for inverting person use: `sh launch_no_reg.sh 24`
The generated point clouds would be saved in the `output_dir` specified in the arguments.

### Class List
airplane, bathtub, bed, bench, bookshelf, bottle, bowl, car, chair, cone, cup, curtain, desk, door, dresser, flower_pot, glass_box, guitar, keyboard, lamp, laptop, mantel, monitor, night_stand, person, piano, plant, radio, range_hood, sink, sofa, stairs, stool, table, tent, toilet, tv_stand, vase, wardrobe, xbox
