# 16824: Visual Learning and Recognition Course Project

Several applications of computer vision like autonomous driving, warehouse management, etc. are nearing deployment in real-world. However these approaches critically depend on black-box 3D neural networks. Interpreting and understanding these networks is an important and challenging problem for vision community. This project attempts to interpret the learning of 3D neural networks through the lens of model inversion. Specifically, we investigate what a 3D model learns by trying to re-create an optimal input based on a perceived output. Recent methods present solutions for inverting classification and detection networks, but only for 2D inputs. This project extend these approaches to 3D which is significantly more complex and ill-posed. We showcase results on inversion of 3D deep learning architectures for classification and detection and further analyse our findings.

## Inverting image-based 3D Object Detection Model
![Detection](/resource/detection_inversion.png)

Please refer to the folder `vis_det` for inverting 3D detection model.

## Inverting point-cloud-based 3D Classification Model
![Classfication](/resource/classification_inversion.png)

Please refer to the folder `Pointnet_Pointnet2_pytorch` for inverting 3D classification model.
