Video presentation link: https://youtu.be/0aHPQ0t-_pE
Video output link: https://youtu.be/r2tf0E-BPXs

How to run number identification on images:

 Libs requirements:
  - tensorflow
  - numpy
  - Pillow

 Input: images 1.jpg, 2.jpg, 3.jpg, 4.jpg, 5.jpg in the input folder

 Run: execute `python run.py` in the terminal

 Output: 1.png, 2.png, 3.png, 4.png, 5.png images in the output folder with green bounding boxes and labels next to them added


Other files:
 - README.txt: this file
 - detect.py: includes function for number detections
 - run.py: executes detection process on images and video frames
 - model1.h5: keras model file with model structure and weights
 - FreeSerif.ttf: font used by Pillow to add labels on output images
 - mat2json.py: converts Matlab format dataset metadata into json format
 - Notebooks (requires additional jupyter library installed for viewing and matplotlib for execution):
    + prepare_data.ipynb: processes SVHN data into dataset ready for CNN training. Data has to be predownloaded into data folder and processed using mat2json.py script
    + modeling.ipynb: creates custom CNN model and trains it
    + prediction.ipynb: prototype notebook for number identification
    + modeling_vgg.ipynb: modeling experiments with VGG16 network
 - Report.pdf: report