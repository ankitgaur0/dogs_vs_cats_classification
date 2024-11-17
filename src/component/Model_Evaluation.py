import os,sys
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
from urllib.parse import urlparse
import mlflow.tensorflow.autologging
from src.Exception_Handler import Custom_Exception
from src.Logger import logging


class Model_Evaluation:
    def __init__(self,history,model,testing_augdata):
        self.testing_augdata=testing_augdata
        self.model=model
        self.history=history
        
    def initiate_evaluation(self):
        try:
            '''logging.info(f"the accuracy of the model is :{self.history.history["accuracy"]}, \n and loss on training data :{self.history.history["loss"]}")
            logging.info(f"the val_accuracy of the model is :{self.history.history["val_accuracy"]} ,\n and val_loss on training data :{self.history.history["val_loss"]}")

            logging.info("ploting the accuracy and val accuaracy graph")
            logging.info(f"{plt.subplot(1,2,2)},{plt.plot(self.history.history["accuracy"],label="Training Accuracy")},{plt.plot(self.history.history["val_accuracy"],label="validation accuarcy"),{plt.title("Model Accuracy")}},{plt.xlabel('Epochs')},{plt.ylabel("Accuracy")},{plt.tight_layout()},{plt.legend()}, {plt.show()}")
            logging.info(f"{plt.subplot(1,2,2)},{plt.plot(self.history.history["loss"],label="Training loss")},{plt.plot(self.history.history["val_loss"],label="validation loss"),{plt.title("Model loss")}},{plt.xlabel('Epochs')},{plt.ylabel("loss")},{plt.tight_layout()},{plt.legend()}, {plt.show()}")'''
            mlflow.set_experiment("Dogs vs Cats Classification experiment")
            mlflow.set_registry_uri("https://dagshub.com/ankitgaur0/dogs_vs_cats_classification.mlflow")
            tracking_url_store=urlparse(mlflow.get_tracking_uri()).scheme
            with mlflow.start_run() as run:
                val_loss, val_accuracy=self.model.evaluate(self.testing_augdata,steps=len(self.testing_augdata))
                # now log the parameter and metrics
                mlflow.log_param("epochs",50)
                mlflow.log_param("Batch size",32)
                mlflow.log_param("optimizer","adam")
                mlflow.log_metric("val_loss",val_loss)
                mlflow.log_metric("val_accuracy",val_accuracy)
                
                #log validation metrics from  history
                for epoch in range(50):
                    mlflow.log_metric("training_accuracy",self.history.history["accuracy"][epoch],step=epoch)
                    mlflow.log_metric("training_loss",self.history.history["loss"][epoch],step=epoch)
                    mlflow.log_metric("validation_accuracy",self.history.history["val_accuracy"][epoch],step=epoch)
                    mlflow.log_metric("validation_loss",self.history.history["val_loss"][epoch],step=epoch)

                #log the keras model
                mlflow.keras.log_model(self.model,"dog_vs_cat_classification_model")

                #log the model summary
                model_summary=[]
                self.model.summary(print_fn=lambda x:model_summary.append(x))
                model_summary_str="\n".join(model_summary)
                with open("model_summary.txt","w") as f:
                    f.write(model_summary_str)
                
                mlflow.log_artifact("model_summary.txt")

                #plot the accuracy and loss graph and log them as artifacts
                plt.figure(figsize=(20,4))
                #accuracy plot
                plt.subplot(1,2,1)
                plt.plot(self.history.history["accuracy"],label="Training_accuracy")
                plt.plot(self.history.history["val_accuracy"],label="Validation_accuracy")
                plt.title("Model Accuracy")
                plt.xlabel("Epochs")
                plt.ylabel("accuracy")
                plt.legend()

                #loss plot
                plt.subplot(1,2,1)
                plt.plot(self.history.history["loss"],label="Training_loss")
                plt.plot(self.history.history["val_loss"],label="Validation_loss")
                plt.title("Model Loss")
                plt.xlabel("Epochs")
                plt.ylabel("loss")
                plt.legend()

                # save the plot
                plot_path="accuracy_loss_plot.png"
                plt.savefig(plot_path)

                # save the plot as an artifact in mlflow
                mlflow.log_artifact(plot_path)


        except Exception as e:
            raise Custom_Exception(e,sys)

