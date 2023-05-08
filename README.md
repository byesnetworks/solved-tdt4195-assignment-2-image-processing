Download Link: https://assignmentchef.com/product/solved-tdt4195-assignment-2-image-processing
<br>
In this assignment, we will introduce you to classifying images with Convolutional Neural Networks (CNNs). Then, we will look into how we can do image filtering in the frequency domain.

<h1>Convolutional Neural Networks</h1>

Figure 1: A CNN containing all the basic elements of a LeNet architecture. The network contains two convolutional layers, two pooling layers, and a single fully-connected layer. The last pooled feature maps are vectorized and serve as the input to a fully-connected neural network. The class to which the input image belongs is determined by the output neuron with the highest value. Figure source: Chapter 12, Digital Image processing (Gonzalez)

In this assignment, we will implement a Convolutional Neural Network (CNN) to recognize digits from MNIST. The basic operations of CNNs are very similar to Fully Connected Neural Networks (FCNNs): (1) a sum of products is formed, (2) a bias value is added, (3) the resulting number is passed through an activation function, and (4) the activation value becomes a single input to a following layer. However, there are some crucial differences between these two networks.

A CNN learns 2-D features directly from raw image data, while a FCNN takes in a single vector. To illustrate this, take a close look at Figure 1. In a FCNN, we feed the output of every neuron in a layer directly into the input of every neuron in the next layer. By contrast, in a convolutional layer, a single value of the output is determined by a convolution over a spatial neighborhood of the input (hence the name convolutional neural net). Therefore, CNNs are not fully connected and they are able to re-use parameters all over the image.

<h2>Computing the output shape of convolutional layers</h2>

This section will give you a quick overview of how to compute the number of parameters and the output shapes of convolutional layers. For a more detailed explanation, look at the recommended resources.

A convolutional layer takes in an image of shape <strong>H<sub>1 </sub></strong>× <strong>W<sub>1 </sub></strong>× <strong>C<sub>1</sub></strong>, where each parameter corresponds to the height, width, and channel, respectively. The output of a convolutional layer will be <em>H</em><sub>2 </sub>× <em>W</em><sub>2 </sub>× <em>C</em><sub>2</sub>. <em>H</em><sub>2 </sub>and <em>W</em><sub>2 </sub>depend on the receptive field size (<strong>F</strong>) of the convolution filter, the stride at which they are applied (<strong>S</strong>), and the amount of zero padding applied to the input (<strong>P</strong>). The exact formula is:

<h3>                                                                     <em>W</em><sub>2 </sub>= (<em>W</em><sub>1 </sub>− <em>F<sub>W </sub></em>+ 2<em>P<sub>W</sub></em>)<em>/S<sub>W </sub></em>+ 1<em>,                                                 </em>(1)</h3>

where <em>F<sub>W </sub></em>is the receptive field of the of the convolutional filter for the width dimension, which is the same as the width of the filter. <em>P<sub>W </sub></em>is the padding of the input in the width dimension, and <em>S<sub>W </sub></em>is the stride of the convolution operation for the width dimension.

For the height dimension, we have a similar equation:

<h3>                                                                       <em>H</em><sub>2 </sub>= (<em>H</em><sub>1 </sub>− <em>F<sub>H </sub></em>+ 2<em>P<sub>H</sub></em>)<em>/S<sub>H </sub></em>+ 1                                                   (2)</h3>

where <em>F<sub>H </sub></em>is the receptive field of the of the convolutional filter for the height dimension, which is the same as the height of the filter. <em>P<sub>H </sub></em>is the padding of the input in the height dimension, and <em>S<sub>H </sub></em>is the stride the convolution operation for the height dimension. Finally, the output size of the channel dimension, <strong>C<sub>2</sub></strong>, is the same as the number of filters in our convolutional layer.

<strong>Simple example: </strong>Given a input image of 32<em>x</em>32<em>x</em>3, we want to forward this through a convolutional layer with 32 filters. Each filter has a filter size of 4 × 4, a padding of 2 in both the width and height dimension, and a stride of 2 for both the with and height dimension. This gives us <em>W</em><sub>1 </sub>= 32<em>,H</em><sub>1 </sub>= 32,

<em>C</em><sub>1 </sub>= 3, <em>F<sub>W </sub></em>= 4<em>,F<sub>H </sub></em>= 4, <em>P<sub>W </sub></em>= 2<em>,P<sub>H </sub></em>= 2 and <em>S<sub>W </sub></em>= 2<em>,S<sub>H </sub></em>= 2. By using Equation 1, we get <em>W</em><sub>2 </sub>= (32 − 4 + 2 · 2)<em>/</em>2 + 1 = 17. By applying Equation 2 for <em>H</em><sub>2 </sub>gives us the same number, and the final output shape will be 17 × 17 × 32, where <em>W</em><sub>2 </sub>= 17<em>,H</em><sub>2 </sub>= 17<em>,C</em><sub>2 </sub>= 32.

<strong>To compute the number of parameters</strong>, we look at each filter in our convolutional layer. Each filter will have <em>F<sub>H </sub></em>×<em>F<sub>W </sub></em>×<em>C</em><sub>1 </sub>= 48 number of weights in it. Including all filters in the convolutional layer, the layer will have a total of <em>F<sub>H </sub></em>× <em>F<sub>W </sub></em>× <em>C</em><sub>1 </sub>× <em>C</em><sub>2 </sub>= 1536 weights. The number of biases will be the same as the number of output filters, <em>C</em><sub>2</sub>. In total, we have 1536 + <em>C</em><sub>2 </sub>= 1568 parameters.

<h2>Task 1: Theory</h2>

Table 1: A simple CNN. Number of hidden units specifies the number of hidden units in a fully-connected layer. The number of filters specifies the number of filters/kernels in a convolutional layer. The activation function specifies the activation function that should be applied after the fully-connected/convolutional layer. The flatten layer takes an image with shape (Height) × (Width) × (Number of Feature Maps), and flattens it to a single vector with size (Height) · (Width) · (Number of Feature Maps).

<table width="594">

 <tbody>

  <tr>

   <td width="48">Layer</td>

   <td width="284">Layer Type</td>

   <td width="190">Number of Hidden Units/Filters</td>

   <td width="72">Activation</td>

  </tr>

  <tr>

   <td width="48">1</td>

   <td width="284">Conv2D (kernel size=5, stride=1, padding=2)</td>

   <td width="190">32</td>

   <td width="72">ReLU</td>

  </tr>

  <tr>

   <td width="48">1</td>

   <td width="284">MaxPool2D (kernel size=2, stride=2)</td>

   <td width="190">–</td>

   <td width="72">–</td>

  </tr>

  <tr>

   <td width="48">2</td>

   <td width="284">Conv2D (kernel size=3, stride=1, padding=1)</td>

   <td width="190">64</td>

   <td width="72">ReLU</td>

  </tr>

  <tr>

   <td width="48">2</td>

   <td width="284">MaxPool2D (kernel size=2, stride=2)</td>

   <td width="190">–</td>

   <td width="72">–</td>

  </tr>

  <tr>

   <td width="48">3</td>

   <td width="284">Conv2D (kernel size=3, stride=1, padding=1)</td>

   <td width="190">128</td>

   <td width="72">ReLU</td>

  </tr>

  <tr>

   <td width="48">3</td>

   <td width="284">MaxPool2D (kernel size=2, stride=2)</td>

   <td width="190">–</td>

   <td width="72">–</td>

  </tr>

  <tr>

   <td width="48"> </td>

   <td width="284">Flatten</td>

   <td width="190">–</td>

   <td width="72">–</td>

  </tr>

  <tr>

   <td width="48">4</td>

   <td width="284">Fully-Connected</td>

   <td width="190">64</td>

   <td width="72">ReLU</td>

  </tr>

  <tr>

   <td width="48">5</td>

   <td width="284">Fully-Connected</td>

   <td width="190">10</td>

   <td width="72">Softmax</td>

  </tr>

 </tbody>

</table>

<ul>

 <li>[0<em>.</em>1<em>pt</em>] Given a single convolutional layer with a stride of 1, kernel size of 5 × 5, and 6 filters. If I want the output shape (Height×Width) of the convolutional layer to be equal to the input image, how much padding should I use on each side?</li>

</ul>

Consider a CNN whose inputs are RGB color images of size 512×512. The network has two convolutional layers. Using this information, answer the following:

<ul>

 <li>[0<em>.</em>2<em>pt</em>] You are told that the spatial dimensions of the feature maps in the first layer are 504×504, and that there are 12 feature maps in the first layer. Assuming that no padding is used, the stride is 1, and the kernel used are square, and of an odd size, what are the spatial dimensions of these kernels? Give the answer as (Height) × (Width).</li>

 <li>[0<em>.</em>1<em>pt</em>] If subsampling is done using neighborhoods of size 2 × 2, with a stride of 2, what are the spatial dimensions of the pooled feature maps in the first layer? (assume the input has a shape of 504 × 504). Give the answer as (Height) × (Width).</li>

 <li>[0<em>.</em>2<em>pt</em>] The spatial dimensions of the convolution kernels in the second layer are 3 × 3. Assuming no padding and a stride of 1, what are the sizes of the feature maps in the second layer? (assume the input shape is the answer from the last task). Give the answer as (Height) × (Width).</li>

 <li>[0<em>.</em>3<em>pt</em>] Table 1 shows a simple CNN. How many parameters are there in the network? In this network, the number of parameters is the number of weights + the number of biases. Assume the network takes in an 32 × 32 image.</li>

</ul>

<h2>Task 2: Programming</h2>

In this task, you can choose to use either the provided python files (task2.py, task2c.py) or jupyter notebooks (task2.ipynb, task2c.ipynb). We recommend you to use the compute resources (either phyiscal computers in the lab our our <a href="https://github.com/hukkelas/TDT4195-StarterCode/blob/master/cluster_tutorial/cluster_info.md">remote servers</a> <a href="https://github.com/hukkelas/TDT4195-StarterCode/blob/master/cluster_tutorial/cluster_info.md">a</a>vailable for the course to make neural network training faster

In this task, we will implement the network described in Table 1 with Pytorch. This network is similar to one of the first successful CNN architectures trained on the MNIST database (LeNet). We will classify digits from the MNIST database. If we use the network Table 1 on images with shape 28 × 28, the convolutional layer will have an output shape of 3<em>.</em>5 × 3<em>.</em>5, which gives undefined behavior. Therefore, to simplify the design of the network we will resize the MNIST digits from 28 × 28 to 32 × 32. This is already defined in the given starter code.

With this task, we have given you starter code similar to the one given in assignment 1. We have set the hyperparameters for all tasks. <strong>Do not change these</strong>, unless stated otherwise in each subtask.

<ul>

 <li>Implement the network in Table 1. Implement this in the jupyter notebook (or python file) task2.py/ipynb. Report the final accuracy on the validation set for the trained network. Include a plot of the training and validation loss during training.</li>

</ul>

By looking at the final train/validation loss/accuracy, do you see any evidence of overfitting? Shortly summarize your reasoning.

<ul>

 <li>The optimizer in pytorch is the method we use to update our gradients. Till now, we have used standard stochastic gradient descent (SGD). Understanding what the different optimizers do is out of the scope of this course, but we want to make you aware that they exist. <a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a></li>

</ul>

Adam is one of the most popular optimizers currently. Change the SGD optimizer to Adam (use torch.optim.Adam instead of torch.optim.SGD), and train your model from scratch.

Use a learning rate of 0<em>.</em>001.

Plot the training/validation loss from both models (the model with Adam and the one with SGD) in the same graph and include this in your report. (Note, you should probably change the plt.ylim argument to [0, 0.1]).

<ul>

 <li>Interpreting CNNs is a challenging task. One way of doing this, is to visualize the learned weights in the first layer as a <em>K </em>× <em>K </em>× 3 image <a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a>, where <em>K </em>is the kernel size..</li>

</ul>

Understanding what the filter does can be difficult. Therefore, we can visualize the activation by passing an image through a given filter. The result of this will be a grayscale image.

Run the image zebra.jpg through the first layer of the ResNet50 network. Visualize the filter, and the grayscale activation of a the filter, by plotting them side by side. Use the pre-trained network ResNet50 and visualize the convolution filters with indices [5<em>,</em>8<em>,</em>19<em>,</em>22<em>,</em>34].

Include the visualized filters and activations in your report.

Implement this in the jupyter notebook (or python file) task2c.py/ipynb.

<em>Tip: </em>The visualization should look something like this if done right:

Figure 2: Visualization of filters and activations in ResNet50. Each column visualizes the (top row) 7 × 7 filter of the first layer, and (bottom row) the corresponding grayscale activation. This is done on the following indices: [0<em>,</em>1<em>,</em>2<em>,</em>3<em>,</em>4<em>,</em>5]

(d) Looking at the visualized filter, and its corresponding activation on the zebra image, describe what kind of feature each filter extracts. Explain your reasoning.

<h1>Filtering in the Frequency Domain</h1>

The Fourier transform is an important signal processing tool that allows us to decompose a signal into its sine and cosine components <a href="#_ftn3" name="_ftnref3"><sup>[3]</sup></a> For a digital image, we use a discrete Fourier transform (DFT) to approximate the fourier transform, which samples from the continuous fourier transform. It does not contain all frequencies, but the number of frequencies sampled are enough to represent the complete image. A 2D version of the DFT can be seen in Equation 3. It transforms an <em>N </em>×<em>M </em>image in the spatial domain to the frequency domain. The number of frequencies in the frequency domain is equal to the number of pixels in the spatial domain.

<h2>                                                                                                   (3)</h2>

where <em>f</em>(<em>x,y</em>) ∈ R<sup>N</sup><sup>×</sup><sup>M </sup>is the image in the spatial domain, and <em>F</em>(<em>u,v</em>) ∈ C<sup>N</sup><sup>×</sup><sup>M </sup>is the image in the frequency domain.

We can perform a convolution in the spatial domain by doing a pointwise multiplication multiplication in the frequency domain. This is known as the <em>convolutional theorem </em>(which can be seen in Equation 4), where F is the Fourier transform, ∗ is the convolution operator, and · is pointwise multiplication.

F{<em>f </em>∗ <em>g</em>} = F{<em>f</em>} · F{<em>g</em>}                                                        (4)

Performing a convolution with the convolutional theorem can be faster than a standard convolution in the spatial domain, as the fast fourier transform has runtime O(<em>N</em><sup>3</sup>) assuming <em>N </em>= <em>M</em>.

<h2>Task 3: Theory</h2>

Before starting on this task, we recommend you to look at the recommended resources about the frequency domain.

<ul>

 <li>Given the images in the spatial and frequency domain in Figure 3, pair each image in the spatial domain (first row) with a single image in the frequency domain (second row). Explain your reasoning.</li>

</ul>

1a                                   1b                                   1c                                   1d                                   1e                                   1f

2a                                   2b                                   2c                                   2d                                   2e                                   2f

Figure 3: A set of images visualized in the spatial domain (first row) and the frequency domain (second row). The frequency images visualizes the amplitude |F{<em>g</em>}|.

<ul>

 <li>What are high-pass and low-pass filters?</li>

 <li>The amplitude |F{<em>g</em>}| of two commonly used convolution kernels can be seen in Figure 4. For each kernel (a, and b), figure out what kind of kernel it is (high- or low-pass). Shortly explain your reasoning.</li>

</ul>

(a)                                              (b)

Figure 4: The amplitude |F{<em>g</em>}| of two convolution kernels that have been transformed by the Fourier transform. The DC component have been shifted to the center for all images. This means that low frequencies can be found around the center of each image, while high frequencies can be found far from the center of each image.

<h2>Task 4: Programming</h2>

Numpy has several useful functions to perform filtering in the frequency domain:

<ul>

 <li><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fft2.html">fft.fft2</a><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fft2.html">:</a> Compute the 2-dimensional discrete Fourier Transform</li>

 <li><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.ifft2.html">fft.ifft2</a><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.ifft2.html">:</a> Compute the 2-dimensional inverse discrete Fourier Transform.</li>

 <li><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fftshift.html">fft.fftshift</a><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fftshift.html">:</a> Shift the zero-frequency component to the center of the spectrum.</li>

 <li>Implement a function that takes an grayscale image, and a kernel in the frequency domain, and applies the convolution theorem (seen in Equation 4). Try it out on a low-pass filter and a high-pass filter on the grayscale image ”camera man”(im = skimage.data.camera()).</li>

</ul>

Include in your report the filtered images and the before/after amplitude |F{<em>f</em>}| of the transform. Make sure to shift the zero-frequency component to the center before displaying the amplitude.

Implement this in the function convolve_im in task4a.py/task4a.ipynb. The high-pass and low-pass filter is already defined in the starter code.

You will observe a ”ringing” effect in the filtered image. What is the cause of this?

<ul>

 <li>Implement a function that takes an grayscale image, and a kernel in the spatial domain, and applies the convolution theorem. Try it out on the gaussian kernel given in assignment 1, and a horizontal sobel filter (<em>G<sub>x</sub></em>).</li>

</ul>

Include in your report the filtered images and the before/after amplitude |F{<em>f</em>}| of the transform. Make sure to shift the zero-frequency component to the center before displaying the amplitude.

Implement this in the function convolve_im in task4b.py/task4b.ipynb. The gaussian and sobel filter are already defined in the starter code.

<ul>

 <li>Use what you’ve learned from the lectures and the recommended resources to remove the noise in the image seen in Figure 5a. Note that the noise is a periodic signal. Also, the result you should expect can be seen in Figure 5b</li>

</ul>

Include the filtered result in your report.

Implement this in the file task4c.py/task4c.ipynb.

<em>Hint: </em>Try to inspect the image in the frequency domain and see if you see any abnormal spikes that might be the noise.

(a)                                              (b)

Figure 5: (a) An image of a moon with periodic noise. (b) The image after applying filtering in the frequency domain

<ul>

 <li>Now we will create a function to automatically find the rotation of scanned documents, such that we can align the text along the horizontal axis.</li>

</ul>

You will use the frequency domain to extract a binary image which draws a rough line describing the rotation of each document. From this, we can use a <a href="https://scikit-image.org/docs/0.12.x/auto_examples/edges/plot_line_hough_transform.html">hough transform</a> to find a straight line intersecting most of the points in the binary image. When we have this line, we can easily find the rotation of the line and the document.

Your task is to generate the binary image by using the frequency spectrum. See Figure 6 which shows you what to expect. We’ve implemented most of the code for you in this task; you only need to alter the function create_binary_image in task4d.py/task4.ipynb.

Include the generated image in your report (similar to Figure 6).

<em>Hint: </em>You can use a thresholding function to threshold the magnitude of the frequency domain to find your binary image (it’s OK to hardcode the thresholding value).

<a href="#_ftnref1" name="_ftn1">[1]</a> You can check out <a href="https://cs231n.github.io/assets/nn3/opt1.gif">this cool gif</a> that visualizes that there is a significant difference in performance of different optimizers. For those specially interested, we recommend <a href="https://cs231n.github.io/neural-networks-3/">CS231n’s course post about optimizers.</a>

<a href="#_ftnref2" name="_ftn2">[2]</a> Note that in this example we are visualizing filters from an RGB image (therefore 3 input channels), not grayscale images from MNIST.

<a href="#_ftnref3" name="_ftn3">[3]</a> Remember that a complex exponent <em>e<sup>it </sup></em>can be rewritten in terms of imaginary sine part and a real cosine part:

<em>e<sup>it </sup></em>= cos(<em>t</em>) + <em>i</em>sin(<em>t</em>) (Euler’s formula), where <em>i</em><sup>2 </sup>= −1.