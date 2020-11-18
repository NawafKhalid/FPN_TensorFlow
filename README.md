# FPN_TensorFlow

## What Is FPN (Feature Pyramid Network)?
Before we talk why FPN what is Feature Pyramid Network here is the paper for [FPN](https://arxiv.org/abs/1612.03144v2) and big thanks for [Mohammad](https://www.medrxiv.org/content/early/2020/06/12/2020.06.08.20121541) because I used some help for writing the Pyramid from his code code.

architecture: ![](https://cdn-images-1.medium.com/max/1000/1*D_EAjMnlR9v4LqHhEYZJLg.png)

Basically rather than using single feature map which is only the bottom-up pathway in FPN and produce the output in the last layer (i.e. AlexNet, ResNet,...). The architecture as shown in the figure staring with bottom-up pathway which is the feed forward computation of the backbone conv. Bottom-up pathway produce the usual feature map from CNN. For top-down pathway producehigher resolution feature map.  And the idea over all from FPN is to combine low resolution, semantically strong feature with high resolution, semantically week featurevia  top-down  pathway  and  lateral  connection.   Lateral  connection  is  developed  for building high-level semantic feature maps at all scales. 

## Why using FPN for image classification?


- Using FPN allow us to extract higher resolution features by upsampling spatially coarser, and this       probably useful for complex features.  

  - FPN is fast like single feature pyramid and Pyramidal feature hierarchy , but more accurate..


### What could be change in the architecture?

- Backbone Network: The original paper they represents the results using ResNet as backbone network, and the process of the backbone convolutional architecture is independent so you can use any network as backbone network.
- Features channels output: Because in the original paper all levels of the pyramid use shared classifiers/regressors as in a traditional featurized image pyramid,
they fixed the feature dimension (numbers of channels, denoted as d) in all the feature maps.

> Simplicity is central to our design and we have found that
> our model is robust to many design choices. We have experimented with more sophisticated blocks (e.g., >    using multilayer residual blocks [16] as the connections) and observed
marginally better results. Designing better connection modules is not the focus of this paper, so we opt for the simple design described above.


### Installation
Every dependency should be include within the requirement file.
### Plugins

Dillinger is currently extended with the following plugins. Instructions on how to use them in your own application are linked below.

| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |
| Medium | [plugins/medium/README.md][PlMe] |
| Google Analytics | [plugins/googleanalytics/README.md][PlGa] |


### Development

Want to contribute? Great!

Dillinger uses Gulp + Webpack for fast developing.
Make a change in your file and instantaneously see your updates!

Open your favorite Terminal and run these commands.

First Tab:
```sh
$ node app
```

Second Tab:
```sh
$ gulp watch
```

(optional) Third:
```sh
$ karma test
```
#### Building for source
For production release:
```sh
$ gulp build --prod
```
Generating pre-built zip archives for distribution:
```sh
$ gulp build dist --prod
```
### Docker
Dillinger is very easy to install and deploy in a Docker container.

By default, the Docker will expose port 8080, so change this within the Dockerfile if necessary. When ready, simply use the Dockerfile to build the image.

```sh
cd dillinger
docker build -t joemccann/dillinger:${package.json.version} .
```
This will create the dillinger image and pull in the necessary dependencies. Be sure to swap out `${package.json.version}` with the actual version of Dillinger.

Once done, run the Docker image and map the port to whatever you wish on your host. In this example, we simply map port 8000 of the host to port 8080 of the Docker (or whatever port was exposed in the Dockerfile):

```sh
docker run -d -p 8000:8080 --restart="always" <youruser>/dillinger:${package.json.version}
```

Verify the deployment by navigating to your server address in your preferred browser.

```sh
127.0.0.1:8000
```

#### Kubernetes + Google Cloud

See [KUBERNETES.md](https://github.com/joemccann/dillinger/blob/master/KUBERNETES.md)


### Todos

 - Write MORE Tests
 - Add Night Mode

License
----

MIT


**Free Software, Hell Yeah!**

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
