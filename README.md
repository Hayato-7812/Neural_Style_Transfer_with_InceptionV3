# Neural_Style_Transfer_with_InceptionV3

This repository contains the implementation of Neural Style Transfer using TensorFlow. The project includes code for loading images, converting them to grayscale, performing style transfer, and saving the results.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Introduction
Neural Style Transfer is a technique that combines the content of one image with the style of another. This implementation uses TensorFlow and the InceptionV3 model to achieve this. The project demonstrates how to load images, preprocess them, perform style transfer, and visualize the results.

## Installation
To run this project, you need to have Python and TensorFlow installed. You can install the necessary dependencies using the following command:

```bash
pip install tensorflow matplotlib tqdm
```

## Usage
1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/neural-style-transfer.git
    cd neural-style-transfer
    ```

2. **Prepare the images**:
    - Place your content and style images in the `content` directory.

3. **Run the notebook**:
    - Open the Jupyter notebook `Neural_Style_Transfer_with_InceptionV3.ipynb` and run the cells to perform style transfer.

## Results
The results of the style transfer will be saved in the `results` directory. You can view the original content image, the grayscale image, and the final stylized image.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## Tree
```
.
├── LICENSE
├── Neural_Style_Transfer_with_InceptionV3.ipynb
├── README.md
└── content
    ├── Edy.jpg
    ├── Steve.jpeg
    ├── The_starry_night.jpg
    ├── Tübingen_Neckarfront_3.JPG
    ├── style1.jpeg
    └── style2.jpeg
```
