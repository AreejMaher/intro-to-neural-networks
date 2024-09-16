# Neural networks

## Neural networks approach the problem : 
*  The idea is to take a large number of handwritten digits, known as training examples and then develop a system which can learn from those training example.

* increasing the number of training examples, the network can learn more about handwriting, and so improve its accuracy.


## key ideas about neural networks: 
* including two important types of artificial neuron (the perceptron and the sigmoid neuron), and the standard learning algorithm for neural networks, known as stochastic gradient descent.

* Today, it's more common to use other models of artificial neurons than perceptron.

* the main neuron model used is one called the sigmoid neuron.

# perceptrons
## how do perceptrons work?
* A perceptron takes several binary inputs, x1,x2,…x1,x2,…, and produces a single binary output:

![how do perceptrons work](Capture.PNG)

* weights, w1,w2,…w1,w2,…, real numbers expressing the importance of the respective inputs to the output , as it increases it indicate that the output really cares about this input.

 The neuron's output, 0 or 1, is determined by whether the weighted sum ∑ is less than or greater than some threshold value .

 ![how do perceptrons work](Capture1.PNG)
 
* If the weight of this input is less than the threshold value then the output will not really care about this input.

- ex :
  * if w1 = 6 , w2 =2 , w3 =2 
    *  hreshold = 5 >> output will  depend on input 1 
    * threshold = 3 >> output will  depend on input 1 or both input 2 and 3 together 
* By varying the weights and the threshold , we can get different models of decision-making.

## network:
###  the first layer of perceptrons : 
* is making three very simple decisions, by weighing the input evidence. 
###  The second layer fo perceptrons :
* Each of those perceptrons is making a decision by weighing up the results from the first layer of decision-making. In this way a perceptron in the second layer can make a decision at a more complex and more abstract level than perceptrons in the first layer.

![how do perceptrons work](Capture2.PNG)
###  The Third layer fo perceptrons :
* even more complex decisions can be made.

In this way, a many-layer network of perceptrons can engage in sophisticated decision making.

## Let's simplify
can be written as dot product between w (weights) and x (inputs).

![how do perceptrons work](Capture3.PNG)

b(bais)≡−threshold

![how do perceptrons work](Capture4.PNG)

* as bias increase it become more easy to get output = 1 (less threshold).
* as bias decrease (very negative ) it become more difficult  to get output = 1 (more threshold).

## Another way perceptrons can be used is to compute the elementary logical functions :
* think of as underlying computation, functions such as AND, OR, and NAND.
- NAND gate example :
    * w1 = -2 , w2 = -2 , b =3
        * 00 :
            * -2(0)+-2(0)+3 = 3 (+ve) output = 1
        * 01 :
            * -2(0)+-2(1)+3 = 1 (+ve) output = 1
        * 10 :
            * -2(1)+-2(0)+3 = 1 (+ve) output = 1
        * 11
            * -2(1)+-2(1)+3 = -1 (-ve) output = 0
* NAND gate is univeral so we can build up any computation up out of NAND gates.
    * so perceptrons are also universal for computation

## perceptrons is simultaneously reassuring and disappointing
* It's reassuring because it tells us that networks of perceptrons can be as powerful as any other computing device
* it's also disappointing, because it makes it seem as though perceptrons are merely a new type of NAND gate.

## Suggests :
* devise learning algorithms which can automatically tune the weights and biases of a network of artificial neurons. This tuning happens in response to external stimuli, without direct intervention by a programmer


* we have a network of perceptrons and suppose that a small change in any weight (or bias) causes a small change in the output
![how do perceptrons work](Capture5.PNG)
* example :
    * network was mistakenly classifying an image as an "8" when it should be a "9" 
    * we repeatly changes the weights and biases over and over to produce better and better output. The network would be learning.
###  problem :
*  this isn't what happens when our network contains perceptrons .
    * a small change in the weights or bias of any single perceptron in the network can sometimes cause the output of that perceptron to completely flip, say from 0
 to  1
### overcome :
* a new type of artificial neuron called a sigmoid neuron

# Sigmoid neurons
## how do Sigmoid neurons work?
* Just like a perceptron, the sigmoid neuron has weights for each input and an overall bias and inputs But instead of being just 0 or 1 , these inputs can also take on any values between 0 and  1 
* But the output is not 0 or 1 Instead, it's σ(w⋅x+b)
    * where σ :
    * ![how do Sigmoid neurons work](Capture6.PNG)
    * where z (output of perceptrons ) :
    * ![how do Sigmoid neurons work](Capture8.PNG)
    * output :
    * ![how do Sigmoid neurons work ](Capture7.PNG)
* if z 
    * large and positive >> e<sup>-z</sup>≈0 >> σ(z)≈1 
    * very negative  >> e<sup>-z</sup>≈&infin; >> σ(z)≈0
* The sigmoid function smooths out the sharp transitions of a perceptron, making it easier to train neurons in a networkز
    * ![how do Sigmoid neurons work ](Capture9.PNG)
    * ![how do Sigmoid neurons work ](Capture10.PNG)
    * It allows small, controllable changes to the neuron’s output based on small changes in the weights and biases
    * exact form of the sigmoid is less important than its shape
        * ![how do Sigmoid neurons work ](Capture11.PNG)

## The architecture of neural networks
* ![how do Sigmoid neurons work ](Capture12.PNG)
    * can have one or more hidden layer.
* A natural way to design the network is to encode the intensities of the image pixels into the input neurons.
    * If the image is a 64 by 64 greyscale image, then we'd have 4,096=64×64 input neurons.
    * intensities scaled appropriately between 0 and  1.
    * The output layer will contain just a single neuron.
    * the output from one layer is used as input to the next layer.
    * there is no loops (feedforward neural networks) never fed back.
    * If we did have loops, we'd end up with situations where the input to the σ function depended on the output

## A simple network to classify handwritten digits
split problem into two sub-problems
* First, Segmentation: breaking an image containing many digits into a sequence of separate images, each containing a single digit.
    * ![how do Sigmoid neurons work ](Capture13.PNG)
    * Scoring depend on If the classifier is confident and accurate with the digit classification.
* second,Classification :to classify each individual digit (our focus).
    * To recognize individual digits we will use a three-layer neural network:
    * ![how do Sigmoid neurons work ](Capture14.PNG)
    * The input layer of the network contains neurons encoding the values of the input pixels 28 by 28 pixel = 784
    * The input pixels are greyscale
        * 0 = white 
        * 1 = black 
        * in between = shades of gray
    * The second layer of the network is a hidden layer
        * n layers
    * The output layer of the network 
        * 10 neurons
        * output depends on the neuron which has the highest activation value
        * if first output is highest >> indicate 0 and so on from ( 0 to 9 )
## hidden layer 
* first neuron detects whether or not an image like the following is present
    * ![how do Sigmoid neurons work ](Capture15.PNG)
    * It can do this by heavily weighting input pixels which overlap with the image, and only lightly weighting the other inputs.
* the second, third, and fourth neurons in the hidden layer detect whether or not the following images are present
    * ![how do Sigmoid neurons work ](Capture16.PNG)
* four images together make up the 0
    * ![how do Sigmoid neurons work ](Capture17.PNG)
    * we can conclude that the digit is a  0 when these 4 hidden neurons fires

## why not use 4 outputs 
* treating each neuron as taking on a binary value 2<sup>4</sup>=16 
* the first output neuron would be trying to decide what the most significant bit of the digit was. And there's no easy way to relate that most significant bit to simple shapes















