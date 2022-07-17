import math 
from math import ceil
import os
import copy
import random
import time
import pickle
import numpy as np
from sys import exit

####################
def range_include_end(start, end):
    return range(start, end+1)
####################



####################    
def get_op_index_and_latency(line):
    """
    A function to return the latency of a given operation(line) and its index(order).
   
    Input:
    line: the line/operation in the csv latency file as a string.
    
    output: 
    op_index: the index/order of the operations. 
    avg_ms: the latency of the operation.
    
    """
    linecoloumns = line.split(",")
    op_index=int(linecoloumns[8].split(":")[1])
    avg_ms=float(linecoloumns[3])
    return op_index, avg_ms 
#################### 
 


####################
def get_inference_info(csv_output_path, L): 
   """
    A function to break down the end-to-end latency to layers'/embedding latencies. The function also provides maximum variation of measurements.
    
    Input:
    csv_output_path: the path to latency file of the measured model. The latency file is the output of the benchmarking toot. 
    L: number of layers
    
    Output:
    layers_latencies[0]: latency of first layer.
    layers_latencies[1]: latency of second layer.
    worst_model_variation_percentage: variation between max and min latencies.
    layers_latencies: a list of latencies of all layers.
    embedding_latency: the latency of embedding block.
    
   """

   layers=L

   #Dictionary to hold operations indices for blocks in all layers 
   blocks_ops_index_dict_init={}
   
   #Defining the operations indices in layer 0
   embedding=[ i for i in range_include_end(0,47)] + [j for j in range_include_end(51,55)] +[68,69]
   blocks_ops_index_dict_init['embedding']=embedding  
   layer0=[ i for i in range_include_end(48,50)] + [j for j in range_include_end(56,67)] + [k for k in range_include_end(70,103)]
   blocks_ops_index_dict_init['layer0']=layer0  

   
   
   #Filling blocks_ops_index_dict
   blocks_ops_index_dict={}
   blocks_ops_index_dict=copy.deepcopy(blocks_ops_index_dict_init)
   start_index_for_next_layer=104
   for layer_number in range(1,int(layers)):
      key="layer"+str(layer_number)
      blocks_ops_index_dict[key]=[i for i in  range_include_end(start_index_for_next_layer, start_index_for_next_layer +48)]
      start_index_for_next_layer=start_index_for_next_layer+49
   end_index=103+(int(layers)-1)*49 

   
   
   #Filling op_index_and_latency_dict
   op_index_and_latency_dict={}
   with open(csv_output_path, 'r') as file:
      for row in  file.readlines()[22: (22+end_index+1)]:
         [op_index, latency]=get_op_index_and_latency(row)
         op_index_and_latency_dict[str(op_index)]=latency
      file.seek(0)
      avg_ms_line=file.readlines()[22+end_index+55] 
  
   #Filling  blocks_latencies_dict
   blocks_latencies_dict={}
   for block_name in (blocks_ops_index_dict.keys()): 
     corresponding_latencies= [op_index_and_latency_dict[str(op_index)] for op_index in blocks_ops_index_dict[block_name]]
     blocks_latencies_dict[block_name]=sum(corresponding_latencies)
   
   
   
   #Getting inference info
   model_latency_avg=float(avg_ms_line.split('avg=')[1].split('std=')[0])/1000 
   model_latency_min=float(avg_ms_line.split('min=')[1].split('max=')[0])/1000 
   model_latency_max=float(avg_ms_line.split('max=')[1].split('avg=')[0])/1000 
   worst_model_variation_percentage=max(abs(model_latency_min-model_latency_avg), abs(model_latency_max-model_latency_avg))*100/model_latency_avg
   
   print("####\nGetting inference info for: {} \nWorst_model_variation_percentage: {}\n####".format(csv_output_path, worst_model_variation_percentage))
   
   layers_latencies=[blocks_latencies_dict['layer'+str(layer_number)] for layer_number in range(0, int(layers))]
   embedding_latency=blocks_latencies_dict['embedding']
   print("Layers Latency: {}".format(layers_latencies))


   if L!=1:
    avg_layer_latency=sum(layers_latencies[1:])/len(layers_latencies[1:])
    return layers_latencies[0],avg_layer_latency, worst_model_variation_percentage, layers_latencies, embedding_latency
   else:
    return layers_latencies[0],layers_latencies[0], worst_model_variation_percentage, layers_latencies, embedding_latency
####################   


####################
def save_obj(obj, name, path ):
    with open("Latency/"+path+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)
####################


####################
def load_obj(path, name ):
    with open("Latency/"+path + name + '.pkl', 'rb') as f:
        return pickle.load(f)
#################### 
  
  
####################
def time_sleep():
    """
    A function to make the cpu idle between measurements for cooling the cpu down.
    """
    if H==128:
        time.sleep(8) 
    elif H==256:
        time.sleep(10) 
    elif H==384:
        time.sleep(12) 
    elif H==512: 
        time.sleep(14) 
    elif H==640 or H==768:
        time.sleep(16) 
####################

     

####################
def profile(L):
     """
     A function to profile a given model. H, A, I, S are defined in the main function. L is selected according to the binary search function (findFirstOccurrence).
     
     Input:
     L: Number of layers
     
     Output: 
     first_layer_latency
     second_layer_latency
     csv_output_path: the path of the csv file containing the inference latency information
     layers_latencies: a list of latencies of all layers
     worst_model_variation_percentage
     embedding_latency 
     
     """

     model_name_no_extension="H-{}_L-{}_A-{}_I-{}".format(H,L,A,I)
     model_name="H-{}_L-{}_A-{}_I-{}.tflite".format(H,L,A,I)
     csv_output_path=f"./Latency/{latency_dir}/{model_name_no_extension}_B-{B}_S-{S}_T-{T}.csv"
     csv_output_file=f"{model_name_no_extension}_B-{B}_S-{S}_T-{T}.csv"
     already_profiled=os.listdir(os.path.join("./Latency",latency_dir))
     
     model_name_no_extension="H-{}_L-{}_A-{}_I-{}".format(H,L,A,I)
     model_name="H-{}_L-{}_A-{}_I-{}.tflite".format(H,L,A,I)
     csv_output_path=f"./Latency/{latency_dir}/{model_name_no_extension}_B-{B}_S-{S}_T-{T}.csv"
     csv_output_file=f"{model_name_no_extension}_B-{B}_S-{S}_T-{T}.csv"
     already_profiled=os.listdir(os.path.join("./Latency",latency_dir))
     num_runs=2
     warmup_runs_12=get_warmup_runs_12()
     balance=0.75 
     warmup_runs=(max(ceil( ( ((num_runs+warmup_runs_12)*12)*balance-num_runs*L )/L),warmup_runs_12))
     if (csv_output_file not in already_profiled or (repeat_binary_search)):
        
        print(f"#######\n Profiling {model_name_no_extension}_B-{B}_S-{S}_T-{T} \n#######")
        os.system(f"cset proc -s user --exec ./benchmark_model -- --graph={source_dir}/{model_name}  --max_profiling_buffer_entries=8092 --num_threads={T} --num_runs={num_runs}  --profiling_output_csv_file={csv_output_path} --enable_op_profiling=true --input_layer=input_mask:0,input_ids:0,segment_ids:0 --input_layer_shape={B},{S}:{B},{S}:{B},{S} --warmup_runs={warmup_runs} --min_secs=0.001 --warmup_min_secs=0.0000000001")
        os.system("for i in $(pgrep benchmark_model); do kill -9 $i; done")
        first_layer_latency,second_layer_latency,worst_model_variation_percentage, layers_latencies, embedding_latency=get_inference_info(csv_output_path, L)
        while(worst_model_variation_percentage >= model_variation_constraint):
            os.system(f"rm -f {csv_output_path}")
            print("####\n Reprofiling \n####")
            time_sleep()

            os.system(f"cset proc -s user --exec ./benchmark_model -- --graph={source_dir}/{model_name}  --max_profiling_buffer_entries=8092 --num_threads={T} --num_runs={num_runs}  --profiling_output_csv_file={csv_output_path} --enable_op_profiling=true --input_layer=input_mask:0,input_ids:0,segment_ids:0 --input_layer_shape={B},{S}:{B},{S}:{B},{S} --warmup_runs={warmup_runs} --min_secs=0.001 --warmup_min_secs=0.0000000001")
            os.system("for i in $(pgrep benchmark_model); do kill -9 $i; done")
            first_layer_latency, second_layer_latency, worst_model_variation_percentage, layers_latencies, embedding_latency=get_inference_info(csv_output_path, L)
        time_sleep()
     else:
        first_layer_latency, second_layer_latency,worst_model_variation_percentage, layers_latencies, embedding_latency =get_inference_info(csv_output_path, L)
          
     return first_layer_latency, second_layer_latency, csv_output_path , layers_latencies, worst_model_variation_percentage,  embedding_latency       
####################   
  
  
  
####################     
def get_warmup_runs_12():
 if ((H==384 and B==2 and S==512)or ((H==512 or H==768) and S==512)): 
     return 0
 else:
  return 1
####################


####################
def iterator(input_list):
    while 1:
        for item in input_list:
            yield item
####################


####################
def get_diff(model_to_profile, ref_first_layer_latency, profiled_list, first_layer_latency_list, second_layer_latency_list, embedding_latency_list):
 new_model=False
 if (model_to_profile+1) not in profiled_list:
  profiled_list.append(model_to_profile+1)
  new_model=True
 
 L=model_to_profile+1 
 first_layer_latency, second_layer_latency, _,_,_, embedding_latency =profile(L)
 

 if (new_model):
  first_layer_latency_list.append(first_layer_latency)
  embedding_latency_list.append(embedding_latency)
  if (L!=1):
   second_layer_latency_list.append(second_layer_latency)
   

 return (abs(first_layer_latency-ref_first_layer_latency)/ref_first_layer_latency)*100
####################



####################
def apply_metadata_fn():
  """ 
  A function to transfer metadata (j_switch and bundles classifications) from a pretrained predictor to a new predictor that uses different number of threads.
  """
  saturation_point=metadata_dict[config_key]['saturation_point']
  first_layer_latency_list=[]
  second_layer_latency_list=[]
  embedding_latency_list=[]
   
  profiled_list=[]
  if saturation_point==-1:
     linear_region_status=0
     first_layer_latency,second_layer_latency,_,_,_, embedding_latency= profile(12) 
     first_layer_latency_list.append(first_layer_latency)
     second_layer_latency_list.append(second_layer_latency)
     embedding_latency_list.append(embedding_latency)
     profiled_list.append(12)
  else:

     linear_region_status=1

     first_layer_latency,second_layer_latency,_,_,_, embedding_latency= profile(12) 
     first_layer_latency_list.append(first_layer_latency)
     second_layer_latency_list.append(second_layer_latency)
     embedding_latency_list.append(embedding_latency)
     profiled_list.append(12)

     if saturation_point!=2:
      first_layer_latency,second_layer_latency,_,_,_, embedding_latency= profile(2) 
      first_layer_latency_list.append(first_layer_latency)
      second_layer_latency_list.append(second_layer_latency)
      embedding_latency_list.append(embedding_latency)
      profiled_list.append(2)

     first_layer_latency,second_layer_latency,_,_,_, embedding_latency= profile(saturation_point) 
     first_layer_latency_list.append(first_layer_latency)
     second_layer_latency_list.append(second_layer_latency)
     embedding_latency_list.append(embedding_latency)
     profiled_list.append(saturation_point)



  return linear_region_status, saturation_point, profiled_list, first_layer_latency_list, second_layer_latency_list,embedding_latency_list 
####################




####################
def findFirstOccurrence(right):
 """
 A function that implement binary search to find j_switch
 """
 ref_first_layer_latency, ref_second_layer_latency, _,_,_,ref_embedding_latency=profile(12)
 first_layer_latency_list=[]
 second_layer_latency_list=[]
 embedding_latency_list=[]
 
 first_layer_latency_list.append(ref_first_layer_latency)
 second_layer_latency_list.append(ref_second_layer_latency)
 embedding_latency_list.append(ref_embedding_latency)

 profiled_list=[]
 profiled_list.append(12)
 left=0 
 print("#####\nLeft is {}, and right is {}\n#####".format(left,right))
 if (right!=-1):
  if (get_diff(left, ref_first_layer_latency, profiled_list, first_layer_latency_list, second_layer_latency_list, embedding_latency_list) >= model_variation_constraint+increment):
  #then there must be a linear region

    result=-1
    # loop till the search space is exhausted
    while left <= right:
 
        # find the mid-value in the search space and compares it with the target
        mid = (left + right) // 2
 
        # if the key is located, update the result and
        # search towards the left (lower indices)
        if get_diff(mid,ref_first_layer_latency, profiled_list, first_layer_latency_list, second_layer_latency_list, embedding_latency_list)< model_variation_constraint+increment:
            result = mid
            right = mid - 1
 
        # if the key is more than the middle element, discard the left half
        else:
            left = mid + 1
    
    global repeat_binary_search 
    if result==-1:
        repeat_binary_search=True
    elif  ((((profile(result+1)[0]-ref_first_layer_latency)/ref_first_layer_latency)*100) > model_variation_constraint+increment):
        repeat_binary_search=True
    else:
        repeat_binary_search=False
    # return the leftmost index
    linear_region_status=1
    saturation_point=result +1
    return linear_region_status,saturation_point , profiled_list, first_layer_latency_list, second_layer_latency_list,  embedding_latency_list
  else:
   repeat_binary_search=False
   linear_region_status=0
   saturation_point=-1
 else: 
  repeat_binary_search=False
  linear_region_status=0
  saturation_point=-1
 
 return linear_region_status, saturation_point, profiled_list, first_layer_latency_list, second_layer_latency_list,  embedding_latency_list
####################

####################
def determine_right():
 previous_key=list(training_dict.keys())[-1]
 previous_H=int(previous_key.split('_')[0].split('-')[1])
 previous_B=int(previous_key.split('_')[1].split('-')[1])
 previous_S=int(previous_key.split('_')[2].split('-')[1])
 previous_I=int(previous_key.split('_')[3].split('-')[1])
 
 if H!=previous_H or B!=previous_B or S!=previous_S :
     right=11

 elif I!=previous_I:
     target_sat_point_key="H-{}_B-{}_S-{}_I-{}_A-{}".format(H,B,S,previous_I, A)
     right=training_dict[target_sat_point_key]['saturation_point']
 else: 
     target_sat_point_key=list(training_dict.keys())[-1]
     right=training_dict[target_sat_point_key]['saturation_point']
 return right
#####################


####################
def build_training_dict():
   print("#######\nBuilding Training Dictionary for {}\n######".format(config_key))
   

   if(apply_metadata):
     linear_region_status, saturation_point, profiled_list, first_layer_latency_list, second_layer_latency_list,  embedding_latency_list=apply_metadata_fn()
   else:
     if len(list(training_dict.keys()))!=0:
      right=determine_right()
     else: #this means it's the first key in the dictionary
      right=11 
 
     linear_region_status, saturation_point, profiled_list, first_layer_latency_list, second_layer_latency_list,  embedding_latency_list=findFirstOccurrence(right)
  

     while(repeat_binary_search ):
      print("#######\n Repeating the binary search; No saturation point found: {}  \n#######".format(saturation_point)) 
      linear_region_status, saturation_point, profiled_list, first_layer_latency_list, second_layer_latency_list,  embedding_latency_list=findFirstOccurrence(right)
   
 
   print("#######\nProfiled_list:{}\n#######".format(profiled_list))
   print("#######\nSaturation point:{}\n#######".format(saturation_point))
  
   key_list = ["profiled", "saturation_point", "linear_region", "first_layer_linear_region_prediction", "first_layer_saturation_region_prediction"]
   training_dict[config_key]= dict.fromkeys(key_list)
 
   training_dict[config_key]['profiled']= dict.fromkeys([model for model in profiled_list ])
 

   #Filling in the training dictionary for the config
   if linear_region_status==0: #full saturation
    state=1
    training_dict[config_key]['linear_region']='No'
    training_dict[config_key]['first_layer_linear_region_prediction']='NA'
    training_dict[config_key]['first_layer_saturation_region_prediction']=first_layer_latency_list[0]
    training_dict[config_key]['second_layer_saturation_region_prediction']=second_layer_latency_list[0]
    training_dict[config_key]['embedding_saturation_region_prediction']=embedding_latency_list[0]
    training_dict[config_key]['saturation_point']=saturation_point #-1
   else:
     state=2 #saturation and linear
     training_dict[config_key]['linear_region']='Yes'
     training_dict[config_key]['first_layer_linear_region_prediction']=get_linear_model(profiled_list, first_layer_latency_list, saturation_point, False)
     training_dict[config_key]['second_layer_linear_region_prediction']=get_linear_model(profiled_list, second_layer_latency_list, saturation_point,True)
     training_dict[config_key]['embedding_linear_region_prediction']=get_linear_model(profiled_list, embedding_latency_list, saturation_point,False)
     
     training_dict[config_key]['first_layer_saturation_region_prediction']=first_layer_latency_list[0]
     training_dict[config_key]['second_layer_saturation_region_prediction']=second_layer_latency_list[0]
     training_dict[config_key]['embedding_saturation_region_prediction']=embedding_latency_list[0]
     training_dict[config_key]['saturation_point']=saturation_point #saturation point

  
 
   for index, model in enumerate(profiled_list):
     _,_,_,layers_latencies,worst_model_variation, embedding_latency=profile(model)
     training_dict[config_key]['profiled'][model]=[[worst_model_variation],layers_latencies]
 
 
   print(training_dict)
   test(state, profiled_list, saturation_point)
   save_obj(training_dict,"predictor", latency_dir)
####################

####################
def test(state, profiled_list, saturation_point):
  
 models_to_test_sat=[]
 models_to_test_linear=[] 
 
 if I in design_space_test[H][4] and  A in design_space_test[H][3] and B in design_space[H][0] and S in design_space[H][1]:
   if state==1: #full saturation 
        if 1 not in profiled_list:
         models_to_test_sat.append(1) 
        models_to_test_sat.append(4)
        models_to_test_sat.append(6)
        models_to_test_sat.append(9)
   else: #state=2 linear and saturation
    if saturation_point==2 or saturation_point==3: #then we do not have models to test in the linear region. We can test 2 points in the saturation region other than those already profiled
     # 12, 1,6, 3 and 2 is the profiled list.
     points_profiled_saturaion_region=[model for  model in profiled_list if model >= 2]
     candidates=list(set([4,5,7,8,9,10,11]) - set(points_profiled_saturaion_region))
     models_to_test_sat.append(candidates[0]) #L=4
     middle=int(((len(candidates)-1)/2))
     models_to_test_sat.append(candidates[middle]) 
     models_to_test_sat.append(candidates[-1]) #L=11

    elif saturation_point==11 or saturation_point==12: #then we do not have models to test in staturation region since 12 is always tested. We can test 2 poitns in the linear region other than those profiled
     #we also know that 12, 1, 6, and 11 have been surely profiled, In fact, 12, 1,6, 9 and 11  is the profiled list.
     points_profiled_linear_region=[model for  model in profiled_list if model < 11]
     candidates= list(set([2,3,4,5,7,8,9,10]) - set(points_profiled_linear_region))
     models_to_test_linear.append(candidates[0]) #L=2
     middle=int(((len(candidates)-1)/2))
     models_to_test_linear.append(candidates[middle]) 
     models_to_test_linear.append(candidates[-1]) #l=10

    else: #saturation point else where
     points_profiled_linear_region=[model for  model in profiled_list if model < saturation_point]
     candidates_linear=list(set([i for i in range(2, saturation_point)])-set(points_profiled_linear_region))
     if len(candidates_linear)!=0:
      models_to_test_linear.append(candidates_linear[0])  
      middle=int(((len(candidates_linear)-1)/2))
      models_to_test_linear.append(candidates_linear[middle]) 
      models_to_test_linear.append(candidates_linear[-1])

     points_profiled_saturation_region=list(set(profiled_list)-set(points_profiled_linear_region))
     candidates_saturation=list(set([i for i in range(saturation_point, 12)])-set(points_profiled_saturation_region))
     if len(candidates_saturation)!=0:
      models_to_test_sat.append(candidates_saturation[0])
      middle=int(((len(candidates_saturation)-1)/2))
      models_to_test_sat.append(candidates_saturation[middle])
      models_to_test_sat.append(candidates_saturation[-1])  

 if len(models_to_test_linear)!=0:
  print("#######\n Runing Tests for points in linear region for config {} \n#######".format(config_key))
  print("Already profiled in training: {}".format(profiled_list))
  print("Saturation point: {}".format(saturation_point))
  print("Chosen for test: {}\n".format(set(models_to_test_linear)))
  
  training_dict[config_key]['linear_test']=dict.fromkeys([model for model in models_to_test_linear ])   

  m1,b1=training_dict[config_key]['first_layer_linear_region_prediction']
  m2,b2=training_dict[config_key]['second_layer_linear_region_prediction']
  me,be=training_dict[config_key]['embedding_linear_region_prediction']
  
  for model in set(models_to_test_linear):
   training_dict[config_key]['linear_test'][model]=dict.fromkeys(['predicted', 'ground_truth', 'residual' ]) 
   first_layer_predicted= model*m1+b1
   second_layer_predicted=model*m2+b2
   embedding_predicted=model*me+be
   predicted= embedding_predicted+first_layer_predicted +(model-1)*second_layer_predicted
   predicted_list=[embedding_predicted]+[first_layer_predicted] + [second_layer_predicted for l in range(model-1)]

   training_dict[config_key]['linear_test'][model]['predicted']=predicted
   _,_,csv_output_path,layers_latencies,_,embedding_latency=profile(model)
   ground_truth=sum(layers_latencies)+embedding_latency 
   print("Predicted: {}".format(predicted_list))
   print("Truth: {}".format([embedding_latency]+layers_latencies))
   residual=abs(((ground_truth-predicted)/ground_truth)*100)
   print("#######\nResidual: {}\n#######".format(residual))
   while(residual>=model_variation_constraint+increment):
    os.system(f"rm -f {csv_output_path}")
    _,_,csv_output_path,layers_latencies,_,embedding_latency=profile(model)
    ground_truth=sum(layers_latencies)+embedding_latency 
    residual=abs(((ground_truth-predicted)/ground_truth)*100)
    print("Predicted: {}".format(predicted_list))
    print("Truth: {}".format([embedding_latency]+layers_latencies))
    print("#######\nResidual: {}\n#######".format(residual))
   training_dict[config_key]['linear_test'][model]['ground_truth']=ground_truth
   training_dict[config_key]['linear_test'][model]['residual']=residual

   
   
  
 if len(models_to_test_sat)!=0:
  print("#######\n Runing Tests for points in saturation region for config {} \n#######".format(config_key))
  print("Already profiled in training: {}".format(profiled_list))
  print("Saturation point: {}".format(saturation_point))
  print("Chosen for test: {}\n".format(set(models_to_test_sat)))
  
  training_dict[config_key]['sat_test']=dict.fromkeys([model for model in models_to_test_sat ]) 
   
  
  for model in set(models_to_test_sat):
   training_dict[config_key]['sat_test'][model]=dict.fromkeys(['predicted', 'ground_truth', 'residual' ]) 
   first_layer_predicted=training_dict[config_key]['first_layer_saturation_region_prediction']
   second_layer_predicted=training_dict[config_key]['second_layer_saturation_region_prediction']
   embedding_predicted=training_dict[config_key]['embedding_saturation_region_prediction']
   predicted= embedding_predicted+first_layer_predicted +(model-1)*second_layer_predicted
   predicted_list=[embedding_predicted]+[first_layer_predicted] + [second_layer_predicted for l in range(model-1)]


   
   training_dict[config_key]['sat_test'][model]['predicted']=predicted
   _,_,csv_output_path,layers_latencies,_,embedding_latency=profile(model)
   ground_truth=sum(layers_latencies)+embedding_latency 
   print("Predicted: {}".format(predicted_list))
   print("Truth: {}".format([embedding_latency]+layers_latencies))
   residual=abs(((ground_truth-predicted)/ground_truth)*100)
   print("#######\nResidual: {}\n#######".format(residual))
   while(residual>=model_variation_constraint+increment):
    os.system(f"rm -f {csv_output_path}")
    _,_,csv_output_path,layers_latencies,_,embedding_latency=profile(model)
    ground_truth=sum(layers_latencies)+embedding_latency 
    residual=abs(((ground_truth-predicted)/ground_truth)*100)
    print("Predicted: {}".format(predicted_list))
    print("Truth: {}".format([embedding_latency]+layers_latencies))
    print("#######\nResidual: {}\n#######".format(residual))
   training_dict[config_key]['sat_test'][model]['ground_truth']=ground_truth
   training_dict[config_key]['sat_test'][model]['residual']=residual
####################
     
     
#################### 
def get_linear_model(profiled_list, layer_latency_list, saturation_point, second_layer):

 local_profiled_list=profiled_list.copy()
 if second_layer:
    if 1 in local_profiled_list: local_profiled_list.remove(1)

 points_linear_region=[index for index, model in enumerate(local_profiled_list) if model <= saturation_point]

 x =[local_profiled_list[index] for index in points_linear_region ]
 x_np=np.array(x)
 
 y= [ layer_latency_list[index] for index in points_linear_region ]
 y_np = np.array(y)
 m, b = np.polyfit(x_np, y_np, 1)
 return m, b 
 #################### 



#################
# Main Function
#################



if  __name__ == "__main__":
 source_dir="./TFlite_BERT_models"
 latency_dir="big/4_thread/constrained/layer_level/2_percent/"


 design_space={
	 128:[[1,2,4],[64, 128, 256, 384, 512], [1,2,3,4,5,7,8,9,10,11,12], [2,4], 	   [512,1024]],
	 256:[[1,2],[64, 128, 256, 384, 512], [1,2,3,4,5,7,8,9,10,11,12], [2,4,8], 	   [512, 1024, 1536]],
	 384:[[1,2],[64, 128, 256, 384, 512], [1,2,3,4,5,7,8,9,10,11,12], [4,6,8,12],      [1024, 1536, 2048]],
	 512:[[1],[64, 128, 256, 384, 512], [1,2,3,4,5,7,8,9,10,11,12], [4,8],     [1536, 2048, 2560]],
	 640:[[1],[64, 128, 256, 384, 512], [1,2,3,4,5,7,8,9,10,11,12], [8,10],   [2048, 2560, 3072]],
	 768:[[1],[64, 128, 256, 384, 512], [1,2,3,4,5,7,8,9,10,11,12], [8, 12],   [2560, 3072]]
	 }

 design_space_test={
	 128:[[1,2,4],[64, 128, 256, 384, 512], [1,2,3,4,5,7,8,9,10,11,12], [2], 	   [512,1024]],
	 256:[[1,2],[64, 128, 256, 384, 512], [1,2,3,4,5,7,8,9,10,11,12], [4], 	   [512, 1024, 1536]],
	 384:[[1,2],[64, 128, 256, 384, 512], [1,2,3,4,5,7,8,9,10,11,12], [6],      [1024, 1536, 2048]],
	 512:[[1],[64, 128, 256, 384, 512], [1,2,3,4,5,7,8,9,10,11,12], [8],     [1536, 2048, 2560]],
	 640:[[1],[64, 128, 256, 384, 512], [1,2,3,4,5,7,8,9,10,11,12], [10],   [2048, 2560, 3072]],
	 768:[[1],[64, 128, 256, 384, 512], [1,2,3,4,5,7,8,9,10,11,12], [12],   [2560, 3072]]
	 }

 

 model_variation_constraint=2
 increment=0
 apply_metadata=False
 if (apply_metadata):
  metadata_latency_dir="big/4_thread/constrained/layer_level/2_percent/"
  metadata_dict=load_obj(metadata_latency_dir,"predictor")


 training_dict=load_obj(latency_dir,"predictor")
 training_dict={}
 save_obj(training_dict,"predictor", latency_dir)

 print("###########################Warming Up################################")
 os.system(f"cset proc -s user --exec ./benchmark_model -- --graph=./TFlite_BERT_models/H-128_L-1_A-2_I-512.tflite  --max_profiling_buffer_entries=8092 --num_threads=4 --num_runs=50  --enable_op_profiling=true --input_layer=input_mask:0,input_ids:0,segment_ids:0 --input_layer_shape=1,64:1,64:1,64 --warmup_runs=1 --min_secs=0.001 --warmup_min_secs=0.001")
 print("###########################Starting################################")
 for H in design_space.keys():
  for B in design_space[H][0]:
   for S in design_space[H][1]:
    for I in design_space[H][4]:
     for A in design_space[H][3]:
 
      T=4
      repeat_binary_search=False
      config_key="H-{}_B-{}_S-{}_I-{}_A-{}".format(H,B,S,I,A)
      training_dict=load_obj(latency_dir,"predictor")
      build_training_dict()
      
