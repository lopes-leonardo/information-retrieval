import pyUDLF
from pyUDLF import run_calls as udlf
from pyUDLF.utils import inputType
import numpy as np

# for method in ["CPRR", "LHRR", "BFSTREE"]:
method = "LHRR"
print(method)
input_data = inputType.InputType()                                              
input_data.set_method_name(method)                                             
input_data.set_input_files("datasets/20-news-sbert-ranked-lists.txt")                   
input_data.set_lists_file("datasets/20-news-lists.txt")                          
input_data.set_classes_file("datasets/20-news-classes.txt")                      
input_data.set_param("UDL_TASK", "UDL")                                         
input_data.set_ranked_lists_size(400)                                          
input_data.set_param("SIZE_DATASET", 19997)                                      
input_data.set_param("PARAM_CPRR_K", 300)                                        
input_data.set_param("PARAM_LHRR_K", 300)
input_data.set_param("PARAM_LHRR_T", 1)  
input_data.set_param("INPUT_FILE_FORMAT", "RK")
input_data.set_param("OUTPUT_FILE_FORMAT", "RK")    
input_data.set_param("OUTPUT_RK_FORMAT", "NUM")
output = udlf.run(input_data, get_output=True, compute_individual_gain=True)   
rk = output.get_rks()
np.save("output/20-news-sbert-lhrr-300", rk)                                                                    
print("P@10", output.get_log()["P@10"])
print("P@50", output.get_log()["P@50"])
print("P@100", output.get_log()["P@100"])
print("MAP", output.get_log()["MAP"])