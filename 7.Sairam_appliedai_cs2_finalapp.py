#!/usr/bin/env python
# coding: utf-8
import dash
from dash import html, dcc
import plotly.express as px
import pandas as pd
import cv2 
import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

#####
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

####

from dash.dependencies import (
    Input, Output
)

import os
import random

'''
* make a text file
* Example input.txt
* load the text file using pandas.
* load your model and predict
'''

########################################
external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css'
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config['suppress_callback_exceptions'] = True
app.title = 'Agriculture_CROPS_Classification' # title of the application
server = app.server
########################################

# /home/ramchowdary/Desktop/APPLIED AI COURSE/casestudy-1/re-query/customer-dissatisfaction/deployment/required_columns.pkl

# defining the path of the base model 

model_path = "/home/ramchowdary/Desktop/APPLIED AI COURSE/2.Self_case_study-2/best_model/model_final.h5"


# Frontend (basic html)

app.layout = html.Div([
    html.H3('Agriculture-CROP-Prediction'),
 
    # input section
    html.Div([
        # debounce=True allows to press enter and then the input is read
        # press enter when you provide the filepath in the input filed
        # type of the file is .txt
        dcc.Input(
            id='input-filepath', type='url', value='', 
            placeholder='Path of the CROP image', debounce=True
        )
    ]),

    # output section
    html.Div(id='output-prediction') # return the output

], className='container') # container reduces the width and displayed in the middle  of the page.


# Backend - callback
# use the dash callback 
@app.callback(
    Output('output-prediction', 'children'),
    [Input('input-filepath', 'value')]
)

def show_prediction(file_path): # take the local file path as the input
    # checkk the file exists ar not
    if not os.path.isfile(path=file_path):
        return html.Div([html.P("File path either invalid or empty")])
    else:
        print("LOADING......")
        imgc = cv2.imread(file_path)
        imgc = cv2.cvtColor(imgc, cv2.COLOR_BGR2RGB) # convert to rgb image
        crop_image_fig = px.imshow(imgc)
        crop_image_fig.update_layout(coloraxis_showscale=False,
                                    autosize=True, height=400,
                                    margin=dict(l=0, r=0, b=0, t=0)
                                    )
        crop_image_fig.update_xaxes(showticklabels=False)
        crop_image_fig.update_yaxes(showticklabels=False)

        output_result = html.Div([
                                  dcc.Graph(id='crop-image', figure=crop_image_fig)
                                 ])
        ####
        print("PREDICTING THE IMAGE......")
        img_path = file_path
        class_labels = ["JUTE","MAIZE","RICE","SUGARCANE","WHEAT"]

        # read thwe image and applying the resize and reshaping that image into channels last 
        img = cv2.imread(img_path) # reead the image 
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224)) # resizing the given image into 224,224
        img = img/255. # scaling the image
        img = np.array(img).reshape((1, 224, 224, 3))
        # model prediction  
        # load the model
        clf = load_model(model_path)
        
        Y_prediction = clf.predict(img) # predicting the labels in the array format 1-row , 5-columns.
        y_pred = np.argmax(Y_prediction[0]) 
        # maximum probability value among all the predicted probability of each class. in the first row 

        
        #output_val = class_labels[y_pred] # predict the query point.
        output_val  = "prediction : {0} with {1:.2f} % ".format(class_labels[y_pred],Y_prediction[0,y_pred]*100)
        print("DONE....")
        print("="*50)
        ###
        return html.Div([ output_result,
                          html.P(str(output_val))
                        ])

if __name__ == '__main__':
    app.run_server(debug=True)

