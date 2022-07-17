
import tensorflow as tf
import os

for saved_model in os.listdir("saved_models"):
    if not saved_model.endswith(".tflite"):
       temp=os.listdir("saved_models/"+saved_model)[0]
       model_path="saved_models/{}/{}".format(saved_model, temp)
       print("\n#################Converting {}#################\n".format(saved_model))
       converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
       converter._experimental_new_quantizer = True 
       converter.optimizations = [tf.lite.Optimize.DEFAULT]
       weights_int8_model = converter.convert()
       tflite_name=saved_model+".tflite"
       tflite_file = os.path.join("converted", tflite_name)

       with open(tflite_file, "wb") as f:
        f.write(weights_int8_model)

  
  
