# Large Field-of-View Integral Imaging Pickup System
Python implementation of large field-of-view integral imaging pickup system.   
   
_“Computational large field-of-view RGB-D integral imaging system”, Sensors, November 2021_   
Download paper : [Computational large field-of-view RGB-D integral imaging system](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwig1vL-s5X1AhVhwosBHcr0C-oQFnoECA4QAQ&url=https%3A%2F%2Fwww.mdpi.com%2F1424-8220%2F21%2F21%2F7407%2Fpdf&usg=AOvVaw1zVSxamSj4r9g1Cf-YVjCO)

_In this system, use the code_ [Computational integral imaging pickup system](https://github.com/jgnooo/integral-imaging-pickup)

## Process
<p align="center"><img src="https://user-images.githubusercontent.com/55485826/148034898-16077426-5e3e-4d6f-999d-e56910632841.png"></p>

## Pre-trained monocular depth estimation model
* [Trained by NYU RGB-D V2](https://drive.google.com/uc?export=download&id=1k8McRE2vOtrkHmG9ZU6Cd-IUDtr2Fbbv) (650 MB)

## Usage
- Download depth estimation model file.
    - Go to the link above, and download model.
    - Locate file at `monodepth` or `/your/own/path/`.
- Prepare the input image.
    - Locate the input color image to `inputs` directory or `/your/own/path/`.
- Start large FOV integral imaging pickup system.
    ```Bash
    python main.py \
        --color_path ./inputs/image_file_name or /your/own/path/ \
        --output_path ./results or /your/own/path/ \
        --model_path ./monodepth/model.h5 or /your/own/path/ \
        --is_gpu
    ```

## Results of our system
- Sub-aperture Image Array
<p align="center"><img src="https://user-images.githubusercontent.com/55485826/147924806-4f6fcbb2-9525-4171-8642-322c7dc442d9.png"></p>

<p align="center"><img src="https://user-images.githubusercontent.com/55485826/147925539-6d2c945d-1670-4880-b663-3349b54c9596.gif"></p>

## To-Do List
- Update codes (Depth sub-aperture images).