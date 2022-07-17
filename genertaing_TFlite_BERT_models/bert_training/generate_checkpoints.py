import os



design_space={
	 128:[[64, 128,  256, 384, 512], [1,2,3,4,5,6,7,8,9,11,12], [2,4], 	   [512,1024]],
	 256:[[64, 128,  256, 384, 512], [1,2,3,4,5,6,7,8,9,11,12], [2,4,8], 	   [512, 1024,1536]],
	 384:[[64, 128,  256, 384, 512], [1,2,3,4,5,6,7,8,9,11,12], [4,6,8,12],      [1024, 1536, 2048]],
	 512:[[64, 128,  256, 384, 512], [1,2,3,4,5,6,7,8,9,11,12], [4,8],     [1536, 2048, 2560]],
	 640:[[64, 128,  256, 384, 512], [1,2,3,4,5,6,7,8,9,11,12], [8,10],   [2048, 2560, 3072]],
	 768:[[64, 128,  256, 384, 512], [1,2,3,4,5,6,7,8,9,11,12], [8, 12],   [2560, 3072]]
	 }


for embedding_size in list(design_space.keys()): 
  max_layer=design_space[embedding_size][1][-1]
  max_heads=design_space[embedding_size][2][-1]
  max_position_embeddings=design_space[embedding_size][0][-1]
  for intermediate_size in design_space[embedding_size][3]:  
      checkpoint_dir_name="checkpoints/H-{}_L-{}_A-{}_I-{}".format(embedding_size,max_layer, max_heads, intermediate_size)
      cmd=f"mkdir {checkpoint_dir_name}"
      os.system(cmd)
      cmd=f"mkdir {checkpoint_dir_name}/Output"
      os.system(cmd)
      cmd=f"mkdir {checkpoint_dir_name}/Data"
      os.system(cmd)
      
      
      checkpoint_config_file_name="H-{}_L-{}_A-{}_I-{}.json".format(embedding_size, max_layer, max_heads, intermediate_size)
      cmd=f"python3 run_squad_from_mobile_mo.py \
  --bert_config_file=./ConfigFiles/{checkpoint_config_file_name} \
  --data_dir={checkpoint_dir_name}/Data \
  --do_lower_case \
  --do_train \
  --doc_stride=16 \
  --learning_rate=4e-05 \
  --max_answer_length=8 \
  --max_query_length=8 \
  --max_seq_length={max_position_embeddings} \
  --n_best_size=1 \
  --is_training=true \
  --num_train_epochs=1 \
  --output_dir={checkpoint_dir_name}/Output \
  --train_file=./train_mo.json \
  --train_batch_size=1 \
  --vocab_file=./vocab.txt \
  --warmup_proportion=0 "
      os.system(cmd)
        


