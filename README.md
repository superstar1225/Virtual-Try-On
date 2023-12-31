## Virtual Try-On via Misaligment-Aware Normalization &mdash; Official PyTorch Implementation

**\*\*\*\*\* New: Our dataset is released \*\*\*\*\***<br>


![Teaser image](./assets/teaser.png)

> **VITON-HD: High-Resolution Virtual Try-On via Misalignment-Aware Normalization**<br>

> **Abstract:** *The task of image-based virtual try-on aims to transfer an in-shop clothing item onto the corresponding region of a person, which is commonly tackled by fitting the item to the desired body part and fusing the warped item with the person. While an increasing number of studies have been conducted, the resolution of synthesized images is still limited to low (e.g., 256x192), which acts as the critical limitation against satisfying online consumers. As the resolution increases, the artifacts in the misaligned areas between the warped clothes and the desired clothing regions become noticeable in the final results; the architectures used in existing methods have low performance in generating high-quality body parts and maintaining the texture sharpness of the clothes. To address the challenges, we propose a novel virtual try-on method that successfully synthesizes 1024x768 virtual try-on images. Specifically, we first prepare the segmentation map to guide our virtual try-on synthesis, and then roughly fit the target clothing item to a given person's body. Next, we propose ALIgnment-Aware Segment (ALIAS) normalization and ALIAS generator to handle the misaligned areas and preserve the details of 1024x768 inputs. Through rigorous comparison with existing methods, we demonstrate that VITON-HD highly surpasses the baselines in terms of synthesized image quality both qualitatively and quantitatively.*

## Installation

Clone this repository:

```
git clone https://github.com/paulst1225/Virtual-Try-On.git
cd ./VITON-HD/
```

Install PyTorch and other dependencies:

```
conda create -y -n [ENV] python=3.8
conda activate [ENV]
conda install -y pytorch=[>=1.6.0] torchvision cudatoolkit=[>=9.2] -c pytorch
pip install opencv-python torchgeometry
```

## Dataset

We collected 1024 x 768 virtual try-on dataset for **our research purpose only**.
You can download a preprocessed dataset from [VITON-HD DropBox](https://www.dropbox.com/s/10bfat0kg4si1bu/zalando-hd-resized.zip?dl=0).
The frontal view woman and top clothing image pairs are split into a training and a test set with 11,647 and 2,032 pairs, respectively. 


## Pre-trained networks

We provide pre-trained networks and sample images from the test dataset. Please download `*.pkl` and test images from the [VITON-HD Google Drive folder](https://drive.google.com/drive/folders/1UMxv2yvHI-eTUcEBD9aL857GRQahU5Wf) and unzip `*.zip` files. `test.py` assumes that the downloaded files are placed in `./checkpoints/` and `./datasets/` directories.

## Testing

To generate virtual try-on images, run:

```
CUDA_VISIBLE_DEVICES=[GPU_ID] python test.py --name [NAME]
```

The results are saved in the `./results/` directory. You can change the location by specifying the `--save_dir` argument. To synthesize virtual try-on images with different pairs of a person and a clothing item, edit `./datasets/test_pairs.txt` and run the same command.

## License

All material is made available under [Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicate any changes** that you've made.

## Clothing Mask
In a virtual try-on system, a clothing mask is used to accurately represent and overlay virtual garments onto a person's body in real-time or in a rendered image. The clothing mask serves as a silhouette or template that helps the system understand the shape and contours of the person's body, allowing it to accurately place the virtual clothing on the right areas.

The results are saved in the `./datasets/test/clothing-mask/` directory. To make clothing-mask images with different clothing items, edit `./datasets/clothing.txt`. This clothing images are included in the `./datasets/test/cloth/` directory.

Run the work.py file to save the clothing-mask images.

## Testing clothing-mask

To generate clothing mask images, run:

```
python work.py
```
