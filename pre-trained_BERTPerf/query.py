import configargparse
import pickle
import sys

def load_obj(cluster, threads ):
    try:
       with open(f"latency_predictors/{cluster}/{threads}_thread/predictor.pkl", 'rb') as f:
           return pickle.load(f)
           
    except FileNotFoundError as e:
       print("ValueError: The number of threads is either 1/2/4 and the cluster is either big/little.")
       sys.exit() 
 
 
        
def get_sat_region_prediction():
    embedding_predicted= predictor[config_key]['embedding_saturation_region_prediction']
    first_layer_predicted= predictor[config_key]['first_layer_saturation_region_prediction']
    remaining_layers= predictor[config_key]['second_layer_saturation_region_prediction']
    layers_latencies=[embedding_predicted]+[first_layer_predicted]+[remaining_layers for layer in range(L-1)]
    return layers_latencies
  
  
    
def get_linear_region_prediction():
    m1,c1=predictor[config_key]['first_layer_linear_region_prediction']
    m2,c2=predictor[config_key]['second_layer_linear_region_prediction']
    me,ce=predictor[config_key]['embedding_linear_region_prediction']
    first_layer_predicted= L*m1+c1
    remaining_layers=L*m2+c2
    embedding_predicted=L*me+ce
    layers_latencies=[embedding_predicted]+[first_layer_predicted] + [remaining_layers for layer in range(L-1)]
    return layers_latencies



def print_design_space():
   design_space={
	 128:[[1,2,4],[64, 128, 256, 384, 512], [1,2,3,4,5,7,8,9,10,11,12], [2,4], 	   [512,1024]],
	 256:[[1,2],[64, 128, 256, 384, 512], [1,2,3,4,5,7,8,9,10,11,12], [2,4,8], 	   [512, 1024, 1536]],
	 384:[[1,2],[64, 128, 256, 384, 512], [1,2,3,4,5,7,8,9,10,11,12], [4,6,8,12],      [1024, 1536, 2048]],
	 512:[[1],[64, 128, 256, 384, 512], [1,2,3,4,5,7,8,9,10,11,12], [4,8],     [1536, 2048, 2560]],
	 640:[[1],[64, 128, 256, 384, 512], [1,2,3,4,5,7,8,9,10,11,12], [8,10],   [2048, 2560, 3072]],
	 768:[[1],[64, 128, 256, 384, 512], [1,2,3,4,5,7,8,9,10,11,12], [8, 12],   [2560, 3072]]}
   for key, value in design_space.items():
    print(key, ' : ', value)
    
    
    
def check_full_sat():
   try:
     return (predictor[config_key]['linear_region']=='No')
   except KeyError as e:
      print("keyError: The configuration you are querying is not in the design space. \nPlease find the design space below in the format e: {b, s, l, a, i}")
      print_design_space()
      sys.exit()
    
    
    
    
    
    

if __name__=='__main__':
    
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c','--cluster',type=str,default='big',  help='Type of cluster to run inference on: big/LITTLE')
    parser.add_argument('-t','--threads', type=int, default=4,help='Number of threads to run inference with: 1/2/4')
    parser.add_argument('-b','--batch-size', type=int,default=1, help='Inference batch size: 1/2/4.')
    parser.add_argument('-s','--seq-length',type=int, default=64,  help='Sequence length: 64/128/256/384/512')
    parser.add_argument('-e','--hidden-size', type=int,default=128,  help='Width of a layer input/output, also refered to as embedding size: 128/256/384/512/640/768')
    parser.add_argument('-i','--intermediate-size', type=int, default=512,  help='Width of the FFN network: 512/1024/1536/2048/3072')
    parser.add_argument('-a','--att-heads-number', type=int, default=2,help='Number of attention heads: 2/4/6/8/10/12 ')
    parser.add_argument('-l','--layers-number', type=int, default=1, help='Number of layers: 1/2/3/4/5/6/7/8/9/10/11/12')
    parser.add_argument('--show-layers',type=str,  default='no', help='Whether to show individual layers latencies: true/false')


    args = parser.parse_args()    
    C=args.cluster
    T=args.threads
    B=args.batch_size
    S=args.seq_length
    H=args.hidden_size
    I=args.intermediate_size
    A=args.att_heads_number
    L=args.layers_number
    show_layers=args.show_layers
    config_key="H-{}_B-{}_S-{}_I-{}_A-{}".format(H,B,S,I,A)
    layers_latencies=list()
    
    
    ###Loading the predictor###
    predictor=load_obj(C,T)
        
    #### Querying the predictor ####
    is_full_sat=check_full_sat()
    if(is_full_sat):
       if L in predictor[config_key]['profiled'].keys():
           layers_latencies=predictor[config_key]['profiled'][L][1]
       else:
           layers_latencies=get_sat_region_prediction()
         
    else: 
       if L in predictor[config_key]['profiled'].keys():
           layers_latencies=predictor[config_key]['profiled'][L][1]
       else:
           sat_point=predictor[config_key]['saturation_point']
           if (L<sat_point):
               layers_latencies=get_linear_region_prediction()
           else:
               layers_latencies=get_sat_region_prediction()
    
    ### Printing out predictions ####
    print(f"Predicted model latency (ms): {sum(layers_latencies)}")
    if(show_layers=='true' or show_layers=='True'):
       print(f"Predicted layers latencies (ms): {layers_latencies}")
       
      
   
