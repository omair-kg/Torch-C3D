# Torch-C3D
Torch implementation of C3D net
This is a torch implementation of the C3D net by [Tran et al.] (http://vlg.cs.dartmouth.edu/c3d/) trained on UCF101. 

- Data
The dataloader is based on the datasources class by [Michael Mathieu](https://github.com/MichaelMathieu/datasources).
Instead of loading the clip from video files, i first extracted images from the sequences at 25 fps. 

The class for loading the clips is ucf101_listloader. It reads in the clips to load from a text file (train_01 and test_01).

In addition there is also a class ucf_frame. This iterates over the complete set of sequences (each sequence visited once) and selects one clip at random. For my results i am using the listloader.

- Multi threading
To speed up processing I've implemented multithreading inspired by the awesome codebase from [Soumith Chintala](https://github.com/soumith/imagenet-multiGPU.torch)

- Model
The trained from scratch model uses smaller version of the model as defined in the paper and in the caffe code. Adapted from DeepMark

The model can also be trained using the sports1m initialization. The code for this is commented out.

Sports1m initializations can be downloaded from here

From the torch discussion [forum](https://groups.google.com/forum/#!topic/torch7/TAyQtAQ-Ijc) [model](https://www.google.com/url?q=https%3A%2F%2Fyadi.sk%2Fd%2Fjsvv0nBIw8xyQ&sa=D&sntz=1&usg=AFQjCNFvxz29RsoS-d0NshpFdN-DISqkLQ)

Provided by [gulvarol](https://github.com/gulvarol/ltc/issues/1) [model](https://github.com/gulvarol/ltc/releases/download/c3d/c3d.t7)
