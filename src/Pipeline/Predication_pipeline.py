import os,sys
import numpy as np
from src.Logger import logging
from src.Exception_Handler import Custom_Exception
from src.Utils import load_obj
#using the streamlit app as web API
import streamlit as st
from PIL import Image
from abc import ABC,abstractmethod



class Predication_pipeline(ABC):
    def __init__(self):

        pass

    @abstractmethod
    def streamlit_api(self):
        pass

    def load_preprocess_image(self,image,target_size=(100,100)):
        #alreay open (image object)
        # img=Image.open(image_path)
        img=image.resize(target_size)
        img_array=np.array(img)
        #add batch dimension
        img_array=np.expand_dims(img_array,axis=0)

        #scale the value(pixel)
        img_array=img_array.astype('float32') /255
        return img_array

    def initiate_prediction(self,image):
        model=load_obj(os.path.join("artifacts","model.pickle"))
        preprocessed_img=self.load_preprocess_image(image)
        prediction=model.predict(preprocessed_img)

        prediction_class=1 if prediction >=0.5 else 0  # set a threshold at 0.5
        label_mapping={
            0:"Cat",
            1:"Dog"
        }
        prediction_label=label_mapping[prediction_class]

        return prediction_label
    
class Streamlit_api(Predication_pipeline):
    def streamlit_api(self):
        st.markdown("""
           # Dogs :dog: vs Cats :cat: classification        
        """)
        uploaded_img=st.file_uploader("upload an img....",type=["jpg","jpeg","png"])
        if uploaded_img is not None:
            image=Image.open(uploaded_img)
            col1,col2=st.columns(2)
            with col1:
                resized_img=image.resize((150,150),Image.Resampling.BILINEAR)
                st.image(resized_img)

            with col2:
                if st.button("classify"):

                    #preprocess the image and predict the output
                    prediction=self.initiate_prediction(image)
                    
                    st.success(f"Prediction :{str(prediction)}")
                


if __name__=="__main__":
    #making the object of the of the abstract method in the child class
    obj=Streamlit_api()
    obj.streamlit_api()
        
        
    