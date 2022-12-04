## 一些说明

1. nvprof工具已被弃用，使用NVIDIA Nsight Systems  
[cuda profile](https://compchemjorge.wordpress.com/2021/01/18/profiling-gpu-code-with-nsight-systems/)  
```nsys profile --stats=true ./run```   
```ncu -o profile_test -f -k "warmup" --target-processes all --section "MemoryWorkloadAnalysis" --launch-count 1  ./divergence```  
```nsys-ui```    
```ncu-ui```   



2. 参考  
[cuda编程](https://face2ai.com/program-blog/#GPU%E7%BC%96%E7%A8%8B%EF%BC%88CUDA%EF%BC%89)  
《professional CUDA C Programing》