# BargainingBotTensorflowModel
This repo is part of the [Bargaining Bot](https://github.com/shounakmulay/BargainingBot).
This is the machine learning model that predicts the price that the bot uses to bargain with users.

This is a **Linear Regression Model** made using the **Keras** API
(Note that this model was **not** made using the latest version of tensorflow i.e Tensorflow 2.0)
##
[This is the dataset used for training](https://github.com/shounakmulay/BargainingBotTensorflowModel/blob/master/PricePredictionDataset_50k.csv). 
##
### Try it for yourself:
#### [Try in Google Colab](https://colab.research.google.com/drive/1WQtmZqrzKNKHTnOtspu5VMl_QeZy6OEu)

### OR

```
git clone https://github.com/shounakmulay/BargainingBotTensorflowModel.git 
```
##
Once the model is trained the last steps export the model in the format that can be used for serving by the [Cloud ML Engine](https://cloud.google.com/ml-engine/).
* Upload this model to a bucket in Firestore Storage. 
* Then create a new model in ML Engine. 
* Next create a version for this model.
  * Select the python version you used.
  * Select the framework as Tensorflow.
  * Select the framework version as the version of Tensorflow you used.
  * Select the Cloud Storage directory to which you previously saved the exported model.
  * Save the version.
* Wait for the version to be deployed and ready for prediction. (A green tick will appear left to the name of the version)
* If your are following along the entire project your code in the [Webhook repo](https://github.com/shounakmulay/BargainingBotDialogflowWebhook) will now work once you enter the proper path to the model in your code.
 
