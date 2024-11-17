import os,sys
from dataclasses import dataclass

from src.Utils import save_obj
from src.Logger import logging
from src.Exception_Handler import Custom_Exception
import tensorflow
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model ,Sequential
from tensorflow.keras.layers import Input,Lambda ,Dense,Flatten,Conv2D


@dataclass
class Model_config:
    model_path :str=os.path.join("artifacts","model.pickle")
class Model_Trainer:
    def __init__(self,training_dataaug,testing_dataaug,image_size):
        self.model_path_obj=Model_config()
        self.training_data=training_dataaug
        self.testing_data=testing_dataaug,
        self.image_size=image_size
        self.batch_size=32

    def initiate_model(self):
        #using the transfer learning technique
        #load the VGG19 model without top layer
        vgg19=VGG19(weights='imagenet',
                    include_top=False,
                    input_shape=(self.image_size,self.image_size,3)
                    )
        
        # Freeze the base model layers 
        for layer in vgg19.layers:
            layer.trainable=False

        x=Flatten()(vgg19.output)
        x=Dense(1024,activation="relu")(x)
        predictions=Dense(1,activation='sigmoid')(x)

        #create model object
        model=Model(inputs=vgg19.input,outputs=predictions)

        #compile the model
        model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

        #fit the model
        history=model.fit(
            self.training_data,
            batch_size=32,
            #steps_per_epoch=len(self.training_data), #Number of steps per epoch
            epochs=50,
            validation_data=self.testing_data,
            #validation_steps=len(self.testing_data)
        )

        save_obj(self.model_path_obj.model_path,
                 obj=model)

        return(
            model,history
        )