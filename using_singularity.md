## Using Singularity Container

### Singularity container definition file
```shell
Bootstrap: library
From: ubuntu:22.04

%post
    apt-get -y update
    apt-get -y install g++ python3 python3-dev make git 
```
definition file user guide: https://docs.sylabs.io/guides/3.5/user-guide/definition_files.html

### Building a singularity container
Build a container from the above definition file:
```shell
sudo singularity build --sandbox singularity_ubuntu_2204 singularity_ubuntu_2204.def
```

### Start the container
```shell
singularity run singularity_ubuntu_2204
```
tells singularity to run `singularity_ubuntu_2204` container.