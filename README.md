Consider the digits data contained in the files data0,data1,....data9 (downloaded  from http://cis.jhu.edu/~sachin/digit/digit.html)

Each file has 1000 training examples. Each training example is of size 28x28 pixels. 
The pixels are stored as unsigned chars (1 byte) and take values from 0 to 255. 
The first 28x28 bytes of the file correspond to the first training example, the next 28x28 bytes correspond to the next example and so on.

We have digits data and associated with every instance of a digit we have an image which is actually translated into a set of 784 variables
which are greyscale values of the intensities at different pixel in the image.We have 10000 cases ,on each case we have row which are 784 
pixel intensities for the hand drawn digit.

We reduce the dimension usig PCA and do digit classification using LDA. Two  methods are used here to estimate the error 
(misclassification rate) of a classifier i.e Resubstitution and Cross-validation. 
