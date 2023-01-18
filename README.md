# Stream 1 – Spacecraft Detection: Detecting space objects: localize and classify the object by estimating its bounding box and label.

apatation from the original work of: https://github.com/eleow/tfKerasFRCNN
    
### Training

    frcnn_train.py
    
    
### Testing

    frcnn_test.py

### Environment and libraries

    python 3.6.9 or python 3.7.6
    ubuntu 18.04 or Windows 10
    
### libraries
    
    pip install --upgrade pip
    
    pip install pandas==1.1.5
    pip install tensorflow-gpu==1.13.1 protobuf==3.19.6
    pip install scikit-build==0.13
    pip install opencv-python==3.4.2.17
    pip install Pillow==8.4.0
    pip install imgaug==0.4.0
    pip install numba==0.53.1
    
    ---------------------
    # Windows Install 
    pip install pydot==1.4.2
    pip install pydotplus==2.0.2
    pip install graphviz==0.20.1
    
### Folder structure

    frcnn_stream1
    	|__ data
    	    |__ test
    		|__ train
    		|__ val
    		|__ labels
    	|__ scripts
    		|__ train
    			|__ imgxxxxxx.png
    		|__ FRCNN.py
    		|__ fcnn_test.py
    		|__ frcnn_train.py
    		|__ pickle2txt.py
    		|__ stream1_cvia.pickle
    		|__ stream1_cvia.txt
    		|__ annotation_train_cvia_demo_alldata.txt
    		|__ FRCNN_vgg.csv
    		|__ train_cvia.csv


