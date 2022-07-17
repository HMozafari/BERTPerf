
import os



design_space={
	 128:[[64, 128, 192, 256, 384, 512], [1,2,3,4,5,6,7,8,9,10,11,12], [2,4], 	   [512,1024]],
	 256:[[64, 128, 192, 256, 384, 512], [1,2,3,4,5,6,7,8,9,10,11,12], [2,4,8], 	   [512, 1024,1536]],
	 384:[[64, 128, 192, 256, 384, 512], [1,2,3,4,5,6,7,8,9,10,11,12], [4,6,8,12],      [1024, 1536, 2048]],
	 512:[[64, 128, 192, 256, 384, 512], [1,2,3,4,5,6,7,8,9,10,11,12], [4,8],     [1536, 2048, 2560]],
	 640:[[64, 128, 192, 256, 384, 512], [1,2,3,4,5,6,7,8,9,10,11,12], [8,10],   [2048, 2560, 3072]],
	 768:[[64, 128, 192, 256, 384, 512], [1,2,3,4,5,6,7,8,9,10,11,12], [8, 12],   [2560, 3072]]
	 }


#Get the already generated models:
generated_models_list=list()

for generated_model in os.listdir("saved_models"):
    generated_models_list.append(generated_model)


 
for config_file in os.listdir("./ConfigFiles"):
  if config_file.endswith(".json"):
    target_model_name=config_file.split('.')[0]
    if target_model_name not in generated_models_list: 
       
       
       #checkpoint
       embedding_size=int(config_file.split('_')[0].split('-')[1])
       intermediate_size=int(config_file.split('_')[3].split('-')[1].split('.')[0])
       max_layers_design_space=design_space[embedding_size][1][-1]
       max_heads_design_space=design_space[embedding_size][2][-1]
       
       checkpoint_dir="H-{}_L-{}_A-{}_I-{}".format(embedding_size, max_layers_design_space, max_heads_design_space, intermediate_size)
       max_position_embeddings=design_space[embedding_size][0][-1]
       cmd=f"python3 run_squad_from_mobile_mo.py \
  --activation_quantization=false \
  --use_post_quantization=false \
  --data_dir=checkpoints/{checkpoint_dir}/Data   \
  --output_dir=checkpoints/{checkpoint_dir}/Output \
  --vocab_file=./vocab.txt \
  --max_seq_length={max_position_embeddings} \
  --max_query_length=8 \
  --bert_config_file=./ConfigFiles/{config_file} \
  --train_file=./train_mo.json \
  --tflite_model_name={target_model_name} \
  --export_dir=saved_models"

       os.system(cmd)
    else:
       print("Already generated")
