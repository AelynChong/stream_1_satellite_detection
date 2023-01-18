# stream_1_satellite_detection

apatation from the original work of: https://github.com/eleow/tfKerasFRCNN


### Environment and libraries

    ```bash
    python 3.6.9 or python 3.7.6
    ubuntu 18.04 or Windows 10
    ```
    
### libraries
    
    ```bash
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
    ```
    
### Folder structure
    
    ```bash
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
    	|__ jupyter_notebook_test.txt
    ```
    
### Virtual Environment
    
### Install virtual environment
    
    ```bash
    pip install virtualenv
    ```
    
### Create virtual environment
    
    ```bash
    # python -m venv <name_venv>
    python -m venv stream_1
    ```
    
### Activate virtual environment
    
    ```bash
    # windows
    Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
    .\stream_1\Scripts\activate
    
    # linux
    source stream_1/bin/activate
    ```
    
### Deactivate virtual environment
    
    ```bash
    # windows
    deactivate
    
    # linux
    deactivate
    ```
