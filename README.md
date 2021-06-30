
# Common Ground Control

> Robotic excavation with UR10 and Kinect

## Requirements

Install the following tools:

- [Anaconda](https://www.anaconda.com/products/individual)
- [Visual Studio Code](https://code.visualstudio.com/) and extensions: [python](https://marketplace.visualstudio.com/items?itemName=ms-python.python), [pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance) and [editorconfig](https://marketplace.visualstudio.com/items?itemName=EditorConfig.EditorConfig)
- [libfreenect2](https://github.com/OpenKinect/libfreenect2) (see steps below)

## Getting started

Create a environment using `Anaconda prompt`:

    cd FOLDER_OF_REPO
    conda env create -n NAME_OR_TITLE -f environment.yml

Activate the environment:

    conda activate NAME_OR_TITLE
    python -m compas_rhino.install

### Python interface for Kinect driver
  
1. Get [libfreenect](https://github.com/OpenKinect/libfreenect2) last release (Release 0.2.0, libfreenect2-0.2.0-usbdk-vs2015-x64.zip)

1. Install [Zadig](https://github.com/OpenKinect/libfreenect2/blob/master/README.md#windows--visual-studio) usb driver.

1. Get repo [pylibfreenect2 (v0.1.4 release)](https://github.com/r9y9/pylibfreenect2).

1. open ``setup.py`` from ``pylibfreenect2`` change ``'/usr/local/'`` to ``"C:/.../libfreenect2-0.2.0-usbdk-vs2015-x64"`` run setup.py: 

       python setup.py install

1. Download [libusb-1.0.dll](https://github.com/libusb/libusb/releases/tag/v1.0.22) and replace it in ``libfreenect2\bin`` folder.

1. Copy files from ``libfreenect2\bin`` to ``C:\Users\user\anaconda3\envs'env name'\Lib\site-packages\pylibfreenect2``
    
1. Install [kinect-toolbox](https://github.com/nikwl/kinect-toolbox) by following their steps. 


🚀 You're ready! 
