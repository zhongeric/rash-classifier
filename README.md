# rash-classifier
Building an application to classify rashes, from the ground up.

NOTE:

server_main 
 
    The main program that must be run to create the model, edit the bottom 'path_model' to determine
    the name of the saved .keras model.
  
server_image_download

    The program used to pull data from google-images, limited to the first 100 currently. Multiple keywords can be
    specified for more images.
    
      Images should be in the format
      
      train/
          class1
          class2
          ...
      test/
          class1
          class2
          ...
    
pleasework

    The program used to convert the .keras model to a .mlmodel for integration into swift
    
realrashy.py
  
    The program used to formulate a dataset cache for server_main.py.

dataset.py
cache.py
download.py

    Helper programs used by realrashy.py and more.
