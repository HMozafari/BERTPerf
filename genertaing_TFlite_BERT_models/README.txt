###########
A- Generating TFlite models
###########
1- $ cd genertaing_TFlite_BERT_models

2- Set up the environment on your machine by following the instruction in the setup_x86.

3- After setting up the environment, login: $sudo source python37_env/bin/activate
 
4- $ cd bert-training

5- You will then need to run the following python scripts in the following order:

 	
 -"generate_checkpoints.py": Instead of training each model from scratch we generate checkpoints with maximum dimensions and then alter it to generate smaller models. The generated checkpoints will be in "checkpoints" directory.

 	
 -"export_saved_models_mo.py": This generates models in design space in saved_model format. It uses the checkpoints we have generated. The models will be generated in "saved_models" directory.

 	
 -"convert_nightly.py": To run this file, you need to get out  of the python environment that you are in now. Then  run "$pip3 install tf-nightly". What I use is  tf nightly 2.6.0-dev20210516 but it should work with latest tf nightly (I think) . This file converts the models from saved_model format to tflite. The generated tflite will be in "converted" directory. The generated tflite models support dynamic input shapes. That is, you can run it with tflite benchmark with different batch size/sequence length.

####
Note, if you want to change my design space, you can through changing the design space dictionary in /bert-training/cofigFiles/generate_config.py file. Then delete the existing config files and run "generate_config.py" to generate your new config files. You will then need to rerun the above python files again. Please note that you need to go by the design rules of BERT if you want to change the design space, for instance the embedding size (H) has to be divisible by attention heads (A). If you select parameters that do not go along with each other, the python files will complain. 
####

6- copy the tflite bert models in "converted directory" to MEIL_BERTPerf/souce_code_BERTPerf/TFlite_BERT_models


#####################
B- Train BERTPerf
#####################
1- $ cd souce_code_BERTPerf
2- $ chmod +x ./apply_cpu_shielding && sudo ./apply_cpu_shielding
3- $ sudo python3 train_BERTPerf
4- The output predictor is souce_code_BERTPerf/Latency/big/4_thread/constrained/layer_level/2_percent/predictor.pkl


######################
C- Query the predictor
######################
You can copy the predictor you built in the previous step to its corresponding place in /pre-trained_BERTPerf/latency_predictors OR you can just use -
the pre-trained predictors by running the query python file in /pre-trained_BERTPerf: $cd /pre-trained_BERTPerf && python3 query.py --help, to see the input options.

