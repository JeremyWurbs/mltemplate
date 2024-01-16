# Docker

Docker is a tool used to isolate complete software builds along with their environment contexts for ease of testing
and deployment. Docker can package an application and its dependencies in a virtual container that can run on any Linux, 
Windows, or MacOS computer. 

A Docker package is called an *image* which, when run in the Docker daemon becomes known as a *container*. You create
Docker images by creating Dockerfiles, which are very similar to bash scripts. 

We offer a number of Dockerfiles here, which you may find useful.

## Install Docker

Use the official [Docker installation guide](https://docs.docker.com/engine/install/) to install Docker on your machine.

## Install Docker Compose

Docker compose is used for defining and running multi-container Docker applications. With Compose, you use a YAML file 
to configure your application's services. Then, with a single command, you create and start all the services from your 
configuration. 

Install Compose by following the official [Compose installation guide](https://docs.docker.com/compose/install/).

## GPU (Nvidia) Support (optional, but highly recommended)

If you would like your Docker containers to be able to use (Nvidia) gpus, you need to install the Nvidia Container
Toolkit. To do so, follow the instructions in the official
[Nvidia Container Toolkit Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### Test GPU Installation

At this point, a working setup can be tested by running a base CUDA container:

```commandline
sudo docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi
```

Which will download and run the `nvidia-smi` command in the listed image. If successful, you should see something 
similar to:

```text
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.125.06   Driver Version: 525.125.06   CUDA Version: 12.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  Off |
|  0%   51C    P8    46W / 480W |  14359MiB / 24564MiB |     11%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```

# Mltemplate

There are a number of starter images provided to package and deploy Mltemplate, provided for your convenience. All the 
Mltemplate-related images are given in the associated [Dockerfile](local/Dockerfile), using a multi-stage build. 
The individual stages can be built and cached separately, and are described below.

## GPU (Nvidia) Support

To build containers with GPU support, follow the instructions in the GPU Support section above, and then build an 
nvidia/cuda image from the Nvidia Dockerfile that matches your OS and CUDA version.

The Nvidia Dockerfile for Ubuntu 22.04 / CUDA 12.2.2 is included in this repository 
([docker/nvidia_12.2.2/Dockerfile](docker/nvidia_12.2.2/Dockerfile)). If you are using a different OS or CUDA version, 
download the appropriate Dockerfile from 
[Nvidia's official image list](https://gitlab.com/nvidia/container-images/cuda/-/tree/master/dist), place it in its own 
directory here (e.g. just like [nvidia_12.2.2](docker/nvidia_12.2.2)), and adapt the following instructions to match 
this version. Note that your CUDA version can be found by running `nvidia-smi` on the host machine (e.g. `12.2` in the 
above example).

Once you have the appropriate Dockerfile, build the image:

```commandline
docker build -t nvidia/cuda:mltemplate -f docker/nvidia_12.2.2/Dockerfile .
```

## Release Image

The release image is split into a multi-stage build, with the first stage installing all the dependencies required to
run Mltemplate as a base image, without installing Mltemplate itself. This stage should stay cached for subsequent 
rebuilds of the base image, and should only have to be rebuilt when the package dependencies change.

To build the release image, make sure the first line in the [associated Dockerfile](release/Dockerfile) 
matches your build environment. I.e. for an Ubuntu 22.04 CPU-only build, uncomment and use the `FROM ubuntu:22.04`; for 
a GPU build, use the image built above, i.e. uncomment and use the `FROM nvidia/cuda:mltemplate` line. 

Then build the image:

```commandline
docker build -t mltemplate:release -f docker/release/Dockerfile .
```

## Dev Image

The dev image additionally installs all the dev dependencies— for example, those necessary to run linters and unit 
tests— and is built on top of the release image.

```commandline 
docker build -t mltemplate:dev -f docker/dev/Dockerfile .
```

You may then run this image interactively, and run the unit tests, for example:

```commandline
docker run -it mltemplate:dev /bin/bash
rye run tests
```

The same may be accomplished by building the test stage:

```commandline
docker build -t mltemplate:test -f docker/test/Dockerfile .
docker run mltemplate:test
```

Which may be used to automate the testing of the dockerized application.

## Backend Image

Finally, the backend stage can be used to run all the required backend servers in a single container, similar to running 
it locally on your own machine. It may be built with:

```commandline
docker build -t mltemplate:backend -f docker/backend/Dockerfile .
```

And then subsequently run with:

```commandline
docker run mltemplate:backend
```

If you wish to run the servers with GPU support, pass in the `--gpus all` flag:

```commandline
docker run --gpus all mltemplate:backend
```

## Docker Compose 

As an easier, albeit slightly more advanced usage, it is advised to use Docker Compose. A 
[separate backend Dockerfile](compose/Dockerfile) is provided specifically for Compose, which separates out each server 
into its own stage. The associated [docker-compose.yml](docker-compose.yml) file uses each stage to define a separate 
container, which are run together with a single `docker compose up` command. 

Using Docker Compose, servers can be run independently. For example, the training server may be restarted or even shut 
off without affecting the other servers. Server resources may also be allocated on a per-server basis this way. More, 
having the servers split into their own containers makes it easier to scale the application, where each server will 
likely be running on its own machine and infrastructure.

To configure the deployment servers, adapt the [provided Docker Compose file](docker-compose.yml) as desired. Then run
the following from the docker directory to start the servers:

```commandline
docker-compose up
```

## Deleting Images

By default, built images are stored in `/var/lib/docker/overlay2`. Given that each image is likely ~10gb, it is a good 
idea to run `docker system prune -a` from time to time, which will delete all images not actively in a container.
