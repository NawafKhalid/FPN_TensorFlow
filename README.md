# FPN_TensorFlow

## What Is FPN (Feature Pyramid Network)?
Before we talk why FPN what is Feature Pyramid Network here is the paper for [FPN](https://arxiv.org/abs/1612.03144v2) and big thanks for [Mohammad](https://www.medrxiv.org/content/early/2020/06/12/2020.06.08.20121541) because I used some help for writing the Pyramid from his code code.

architecture: ![](https://cdn-images-1.medium.com/max/1000/1*D_EAjMnlR9v4LqHhEYZJLg.png)

Basically rather than using single feature map which is only the bottom-up pathway in FPN and produce the output in the last layer (i.e. AlexNet, ResNet,...). The architecture as shown in the figure staring with bottom-up pathway which is the feed forward computation of the backbone conv. Bottom-up pathway produce the usual feature map from CNN. For top-down pathway producehigher resolution feature map.  And the idea over all from FPN is to combine low resolution, semantically strong feature with high resolution, semantically week featurevia  top-down  pathway  and  lateral  connection.   Lateral  connection  is  developed  for building high-level semantic feature maps at all scales. 

## Why using FPN for image classification?


- Using FPN allow us to extract higher resolution features by upsampling spatially coarser, and this probably useful for complex features.  

- FPN is fast like single feature pyramid and Pyramidal feature hierarchy, but more accurate.


### What could be changed in the architecture?

- Backbone Network: The original paper they represents the results using ResNet as backbone network, and the process of the backbone convolutional architecture is independent so you can use any network as backbone network.
- Features channels output: Because in the original paper all levels of the pyramid use shared classifiers/regressors as in a traditional featurized image pyramid,
they fixed the feature dimension (numbers of channels, denoted as d) in all the feature maps.
- This final set of feature maps    

> Simplicity is central to our design and we have found that
> our model is robust to many design choices. We have experimented with more sophisticated blocks and observed
marginally better results. Designing better connection modules is not the focus of this paper, so we opt for the simple design described above.

Lastly, the main idea is to take all levels, but unfortunately it is not feasible.

> The principle advantage of featurizing each level of an image pyramid is that it produces
a multi-scale feature representation in which all levels are
semantically strong, including the high-resolution levels.
Nevertheless, featurizing each level of an image pyramid has obvious limitations. Inference time increases considerably, making this approach
impractical for real applications. Moreover, training deep networks end-to-end on an image pyramid is infeasible in
terms of memory, and so, if exploited, image pyramids are
used only at test time, which creates an
inconsistency between train/test-time inference.

### Installation
Every dependency should be included within the requirements file.
