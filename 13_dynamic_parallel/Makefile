nested_Hello_World:
	nvcc -arch=sm_75  nested_hello_world.cu -o nested_hello_world -lcudadevrt --relocatable-device-code true
nested_Reduce:
	nvcc -arch=sm_75  nestedReduce.cu -o nestedReduce -lcudadevrt --relocatable-device-code true
nested_ReduceNosync:
	nvcc -arch=sm_75  nestedReduceNosync.cu -o nested_ReduceNosync -lcudadevrt --relocatable-device-code true
clean:
	rm nested_Hello_World nestedReduce
