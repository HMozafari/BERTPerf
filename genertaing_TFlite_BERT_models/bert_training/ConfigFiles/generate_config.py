 
design_space={
	 128:[[64, 128,  256, 384, 512], [1,2,3,4,5,6,7,8,9,10,11,12], [2,4], 	   [512,1024]],
	 256:[[64, 128,  256, 384, 512], [1,2,3,4,5,6,7,8,9,10,11,12], [2,4,8], 	   [512, 1024,1536]],
	 384:[[64, 128,  256, 384, 512], [1,2,3,4,5,6,7,8,9,10,11,12], [4,6,8,12],      [1024, 1536, 2048]],
	 512:[[64, 128,  256, 384, 512], [1,2,3,4,5,6,7,8,9,10,11,12], [4,8],     [1536, 2048, 2560]],
	 640:[[64, 128,  256, 384, 512], [1,2,3,4,5,6,7,8,9,10,11,12], [8,10],   [2048, 2560, 3072]],
	 768:[[64, 128,  256, 384, 512], [1,2,3,4,5,6,7,8,9,10,11,12], [8, 12],   [2560, 3072]]
	 } 
	 

for embedding_size in list(design_space.keys()): 
  for nlayer in design_space[embedding_size][1]:
   for nheads in design_space[embedding_size][2]:
     for intermediate_size in design_space[embedding_size][3]:
        max_position_embeddings = design_space[embedding_size][0][-1]
        config_file_name="H-{}_L-{}_A-{}_I-{}.json".format(embedding_size, nlayer, nheads, intermediate_size)
        f= open(config_file_name,"w+")
        f.write("{\n")
        f.write("\"hidden_size\": {},\n".format(embedding_size))
        f.write("\"hidden_act\": \"gelu\",\n")
        f.write("\"initializer_range\": 0.02,\n")
        f.write("\"vocab_size\": 30522,\n")
        f.write("\"hidden_dropout_prob\": 0.1,\n")
        f.write("\"num_attention_heads\": {},\n".format(nheads))
        f.write("\"type_vocab_size\": 2,\n")
        f.write("\"max_position_embeddings\": {},\n".format(max_position_embeddings)) #for modelling.py 
        f.write("\"num_hidden_layers\": {},\n".format(nlayer))
        f.write("\"intermediate_size\": {},\n".format(intermediate_size))
        f.write("\"attention_probs_dropout_prob\": 0.1\n")
#f.write("\"trigram_input\": true,\n")
#f.write("\"use_bottleneck\": false,\n")
#f.write("\"num_feedforward_networks\": 1,\n")
#f.write("\"key_query_shared_bottleneck\": false,\n")
#f.write("\"classifier_activation\": false,\n")
#f.write("\"normalization_type\": layer_norm,\n")

        f.write("}\n")

        f.close()

