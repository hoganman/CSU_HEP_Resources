# CSU_HEP_Resources

# ENS HPC 
The CSU Engineering Network Services (ENS) High Performance Computing (HPC) cluster is a suite of compute and GPU compute nodes. For a detailed description and more tutorials, please go [https://www.engr.colostate.edu/ens/info/researchcomputing/cluster/|here]

 ## Physics Owned Nodes 

The HEPPPA group has 6 computing nodes of the 45 total. Each is a 2 x Intel Xeon 8-core processors and 64 GB RAM. In addition, we have 22 TB of data storage for physics use only

 ## Ganglia Cluster Monitoring 

The HPC is monitored by ganglia. The software is used to view either live or recorded statistics covering metrics such as CPU load averages or network utilization for many nodes. The url is [http://ens-hpc.engr.colostate.edu/ganglia/]

 ## Network File System 

The cluster uses a network file system (NFS) allowing a user to access files over a computer network much like local storage is accessed. The host computer that redirects I/O is the "hpc-storage" node, which is NOT visible to non-admin users. The way to monitor I/O usage is to go to the [http://ens-hpc.engr.colostate.edu/ganglia/?c=ens-hpc&h=hpc-storage&m=load_one&r=hour&s=by%20name&hc=4&mc=2|ganglia hpc-storage page] and monitor the Load / Processes graph (top left). If the graph is over 12 (the number of cores), that means more the disks are too busy to accommodate all network requests and must wait for the disks to be ready for another query. 

While I have not found a single good way to avoid this bottleneck, the following suggestions can help:
 * Submit jobs with wait times so that only a certain number of jobs are running at one point
 * Insert into your jobs a sleep time. You get all the slots available and can use a smaller submit wait time as well.
 * Query the hpc-storage node using this python script, which takes the load average over the last minute. [attachment:gethpcstorage_usage.py]  

 ## Environment Setup 

 ```bash
export UNIVA=/usr/loca/univa
export UNIVAVERSION=8.5.5  # subject to change with updates

export LD_LIBRARY_PATH=/physics/INSTALLATION/lib:/physics/INSTALLATION/lib64:/usr/lib:/usr/lib64:/usr/local/lib/:/usr/local/lib64:/usr/local2/etc/scripts/linux/X11rdp/lib:$UNIVA/$UNIVAVERSION/lib/lx-amd64:$LD_LIBRARY_PATH

export C_INCLUDE_PATH=/physics/INSTALLATION/include:/usr/include:/usr/local/include/:/usr/include/libxml2:/usr/include/X11:/usr/local2/etc/scripts/linux/X11rdp/include:$UNIVA/$UNIVAVERSION/include:${C_INCLUDE_PATH}

export PATH=/physics/INSTALLATION/bin:$PATH/./:/bin:/usr/bin:/usr/sbin:/sbin:/usr/local/bin:/usr/local/Modules/3.2.10/bin:$UNIVA/$UNIVAVERSION/bin/lx-amd64:/usr/lib64/qt-3.3/bin:/usr/local/openmpi/1.6.4/x86_64/ib/gnu/bin

export MANPATH=/physics/INSTALLATION/man:$MANPATH

 ```

 ## Batch Queue 

The software the manages jobs is the Univa Grid Engine or UGE. The following commands will be most useful to the average user

 * qsub: Submits jobs to the queue
 * qdel: Deletes jobs on the queue. Important option is -u <username> where <username> is your username
 * qstat: Lists jobs on the queue. Important options are -u "<username>" where <username> any username and accepts wildcards using *, -f and -r for more detailed job info
 * qacct: Gives a detailed description of the job runtime, memory usable, and other metrics

Advanced commands include qmod and qrsub. You should also familiarize yourself with the queues and nodes on the system.

There also is a [attachment:UsersGuideGE.pdf|users guide document] which you may find useful.

 ## Queues 

A queue is a set list of compute or graphical processing unit (GPU) nodes assigned to a specific name. You can think of this like the difference between regular and express lines at the grocery store. Depending on the job specifications, your job is always assigned to a queue and a node. You can always specify both.

 ## Free Use Queues =
Below is a list of queues that you are welcome to submit freely too.


|  Name  |  Description  |  Nodes  (format # node# except for GPUs) |
| defaultfaculty.q | For long (>8hr) jobs | 1, 2, 3, ..., 18, 19, 40, 41, 42, 43, 44, 45 |
| long.q | Eventual replacement for defaultfaculty.q | 1, 2, 3, ..., 18, 19, 40, 41, 42, 43, 44, 45 |
| short.q | Short (<8hr) jobs | 27, 28, 29, 30 |
| gpu.q | Designed for GPU jobs | gpu1, gpu2, gpu3, gpu4 |

Use these lines for LONG jobs
 ```bash
#$ -l qname="defaultfaculty.q|long.q|physics.q"
#$ -l hostname="node1|node2|node3|node4|node5|node6|node7|node8|node9|node10|node11|node12|node13|node14|node15|node16|node17|node18|node19|node40|node41|node42|node43|node44|node45"
 ```


Use these lines for SHORT jobs
 ```bash
#$ -l qname="short.q|physics.q"
#$ -l hostname="node27|node28|node29|node30|node40|node41|node42|node43|node44|node45"
 ```


 ## Owned / Opportunistic Queues =

The model of the ENS-HPC cluster is that groups buy nodes into the cluster and they have priority on those nodes.  The batch queue system is setup to kill jobs if the owners want to use them . This in the past has not always worked, but you have been warned.

|  Name  |  Description  |  Nodes  (format # node# except for GPUs) |
| physics.q | Owned by this group | 40, 41, 42, 43 , 44, 45 |
| musky.q |  | 20, 21, 22, 23, 24, 25, 26, 32, 33, 34, ..., 38, 39 |
| jathar.q |  | 31 |
| snow.q | | gpu5, gpu6, gpu7, gpu8 |
| musky-gpu.q |  | gpu9, gpu10, gpu11, gpu12 |

 ## Nodes 

See [https://www.engr.colostate.edu/ens/info/researchcomputing/cluster/keckinfo.html|here] for the specific node resources like memory and CPUs.

 ## Compute/CPU Nodes By Ownership =

| Name | Queues | - | Name | Queues | - | Name |  Queues | - | Name | Queues | - | Name | Queues |
| ---- | ------ | - | ---- | ------ | - | ---- | ------- | - | ---- | ------ | - | ---- | ------ |  
|node1     | defaultfaculty.q, long.q |  | node11 | defaultfaculty.q, long.q |  | node20 | munsky.q | |        |         | | node40 | physics.q, defaultfaculty.q |
|node2     | defaultfaculty.q, long.q |  | node12 | defaultfaculty.q, long.q |  | node21 | munsky.q | |        |         | | node41 | physics.q, defaultfaculty.q |
|node3     | defaultfaculty.q, long.q |  | node13 | defaultfaculty.q, long.q |  | node22 | munsky.q | | node31 | munsky.q | | node42 | physics.q, defaultfaculty.q |
|node4     | defaultfaculty.q, long.q |  | node14 | defaultfaculty.q, long.q |  | node23 | munsky.q | | node32 | munsky.q | | node43 | physics.q, defaultfaculty.q |
|node5     | defaultfaculty.q, long.q |  | node15 | defaultfaculty.q, long.q |  | node24 | munsky.q | | node33 | munsky.q | | node44 | physics.q, defaultfaculty.q |
|node6     | defaultfaculty.q, long.q |  | node16 | defaultfaculty.q, long.q |  | node25 | munsky.q | | node34 | munsky.q | | node45 | physics.q, defaultfaculty.q |
|node7     | defaultfaculty.q, long.q |  | node17 | defaultfaculty.q, long.q |  | node25 | munsky.q | | node35 | munsky.q | | node27 | short.q |
|node8     | defaultfaculty.q, long.q |  | node18 | defaultfaculty.q, long.q |  | node26 | munsky.q | | node36 | munsky.q | | node28 |  short.q  |
|node9     | defaultfaculty.q, long.q |  | node19 | defaultfaculty.q, long.q |  |        |          | | node37 | munsky.q | | node29 |  short.q |
|node10    | defaultfaculty.q, long.q |  |        |                          |  |        |         | | node38 | munsky.q | | node30 |  short.q |
|          |                          |  |        |                          |  |        |         | | node39 | munsky.q | |       |           |

 ## Graphics/GPU Nodes By Ownership =

| Name | Queues | GPU's |
| ---- | ------ | ----- |
| gpu1     | gpu.q      | Tesla K40c +  3x  Geforce 780 |
| gpu2     | gpu.q      |  3x  Geforce 780 |
| gpu3     | gpu.q      |  3x  Geforce 780 |
| gpu4     | gpu.q      |  3x  Geforce 780 |
| gpu5     | snow.q, gpu.q |  3x  Geforce 780 |
| gpu6     | snow.q, gpu.q  |  3x  Geforce 780 |
| gpu7     | snow.q, gpu.q  |  3x  Geforce 780 |
| gpu8     | snow.q, gpu.q | Geforce Titan X +  3x  Geforce 780 |
| gpu9     | munsky.q, gpu.q |  4x  Geforce GTX 1080 |
| gpu10    | munsky.q, gpu.q |  4x  Geforce GTX 1080 |
| gpu11    | munsky.q, gpu.q |  4x  Geforce GTX 1080 |
| gpu12    | munsky.q, gpu.q |  4x  Geforce GTX 1080 |

 ## Batch Queue Setup on Compute Nodes 

To run jobs, it is most convenient to write a script with all the commands and options set inside it. Sample setups are below.

 ## Sample Script =

Create a file called script.sh with the following contents

 ```bash
#!/bin/bash

#merge stdout and stderr into one file
#$ -j yes
# define the working directory where the job runs, do not use this dir in general
#$ -wd /physics/home
# where to store stdout, do not use this dir in general
#$ -o /physics/home
# define which queues to work with exclusively. The "|" concatenates multiple queues
#$ -l qname="short.q|defaultfaculty.q"
# what is the length of the job in HH:MM:SS
#$ -l h_cpu=:1:
# how much memory will the job need, use M for megabytes and G for gigabytes
#$ -l h_data=500M
echo "Hello world!"
 ```
which is submitted using "qsub script.sh" (no quotes)

 ## Other Job Options =

The full set of options available on the queue can be searched using the man pages
 * man qsub : lists most options for jobs except job resources (see man complex)
 * man complex : lists all the options for job resource specifications like job time (see man sge_types for how to describe resources)
 * man sge_types : lists accepted attributes for job resources
 * man qstat : lists how to view currently running jobs

 ```bash
# specify specifically which nodes to use with "|" concatenating multiple nodes
#$ -l hostname="node1|node2"
# set the job priority to -10 (cannot go above 0 unless su)
#$ -p -10
 ```

 ## Parallel Processing =

If you need to use multithreading on your jobs with OpenMP for example, use these options

 ```bash
# specify the parallel/multithread processing environment as OpenMP using 16 threads
#$ -pe OpenMP 16
# export a environment variable that persists in the job
#$ -v OMP_NUM_THREADS=16
 ```

MPI is also supported.

 ## Physics Software 

Currently the necessary libraries for ROOT v5r34 and "grid" access under /physics/software

 ## Problems 


## GPU Queues ##
The ENS-HPC has a number of free-use and owned GPU nodes. The list of available GPU's is below with the most up-to-date list [https://www.engr.colostate.edu/ens/info/researchcomputing/cluster/keckinfo.html](here)

| Name  | Queues        | GPU's                            |
| ----- | ------------- | -------------------------------- |
| gpu1  | gpu.q         | Tesla K40c + 3x Geforce 780      |
| gpu2  | gpu.q         | 3x Geforce 780                   |
| gpu3  | gpu.q         | 3x Geforce 780                   |
| gpu4  | gpu.q         | 3x Geforce 780                   |
| gpu5  | snow.q, gpu.q | 3x Geforce 780                   |
| gpu6  | snow.q, gpu.q | 3x Geforce 780                   |
| gpu7  | snow.q, gpu.q | 3x Geforce 780                   |
| gpu8  | snow.q, gpu.q | Geforce Titan X + 3x Geforce 780 |
| gpu9  | munsky.q, gpu.q |4x Geforce GTX 1080             |
| gpu10 | munsky.q, gpu.q |4x Geforce GTX 1080             |
| gpu11 | munsky.q, gpu.q |4x Geforce GTX 1080             |
| gpu12 | munsky.q, gpu.q |4x Geforce GTX 1080             |

## Query GPUs ##

The supported GPU query tool is `nvidia-smi`

```bash
[mhogan@ens-hpc ~]$ ssh gpu1
Last login: Mon Jun 18 09:37:48 2018 from ens-hpc
[mhogan@gpu1 ~]$ nvidia-smi -L
GPU 0: GeForce GTX 780 (UUID: GPU-7bbd9f21-01f9-0943-fa5f-4c45e1d65838)  # id = 0
GPU 1: GeForce GTX 780 (UUID: GPU-520cc3dc-e919-6e9b-f8ee-c965aaddad88)  # id = 1
GPU 2: Tesla K40c (UUID: GPU-02b1af51-6cc4-303b-85d3-c0965accc6ee)       # id = 2
GPU 3: GeForce GTX 780 (UUID: GPU-e9daca7b-085d-0f9b-0d8e-e3ea1b59e426)  # id = 3
[mhogan@gpu1 ~]$ nvidia-smi
Tue Jun 19 11:54:09 2018       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 367.48                 Driver Version: 367.48                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 780     Off  | 0000:02:00.0     N/A |                  N/A |
| 28%   48C    P0    N/A /  N/A |      0MiB /  3020MiB |     N/A      Default |
+-------------------------------+----------------------+----------------------+
|   1  GeForce GTX 780     Off  | 0000:03:00.0     N/A |                  N/A |
| 28%   50C    P0    N/A /  N/A |      0MiB /  3020MiB |     N/A      Default |
+-------------------------------+----------------------+----------------------+
|   2  Tesla K40c          Off  | 0000:83:00.0     Off |                    0 |
| 24%   46C    P0    44W / 235W |      0MiB / 11439MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   3  GeForce GTX 780     Off  | 0000:84:00.0     N/A |                  N/A |
|  0%   46C    P0    N/A /  N/A |      0MiB /  3020MiB |     N/A      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|    0                  Not Supported                                         |
|    1                  Not Supported                                         |
|    3                  Not Supported                                         |
+-----------------------------------------------------------------------------+
```

## Masking GPUs ##

If you are only going for one specific gpu then I would try the environment variable CUDA_VISIBLE_DEVICES. See the article below

## CUDA_VISIBLE_DEVICES ##

Originally from [[http://www.acceleware.com/blog/cudavisibledevices-masking-gpus]]

CUDA_VISIBLE_DEVICES – Masking GPUs
Submitted by Chris Mason on Wed, 2013-02-20 10:51

Does your CUDA application need to target a specific GPU? If you are writing GPU enabled code, you would typically use a device query to select the desired GPUs. However, a quick and easy solution for testing is to use the environment variable CUDA_VISIBLE_DEVICES to restrict the devices that your CUDA application sees. This can be useful if you are attempting to share resources on a node or you want your GPU enabled executable to target a specific GPU.
Environment Variable Syntax 	Results

```bash
CUDA_VISIBLE_DEVICES=1 	        # Only device 1 will be seen
CUDA_VISIBLE_DEVICES=0,1 	# Devices 0 and 1 will be visible
CUDA_VISIBLE_DEVICES=”0,1” 	# Same as above, quotation marks are optional
CUDA_VISIBLE_DEVICES=0,2,3 	# Devices 0, 2, 3 will be visible; device 1 is masked
```


CUDA will enumerate the visible devices starting at zero. In the last case, devices 0, 2, 3 will appear as devices 0, 1, 2. If you change the order of the string to “2,3,0”, devices 2,3,0 will be enumerated as 0,1,2 respectively. If CUDA_VISIBLE_DEVICES is set to a device that does not exist, all devices will be masked. You can specify a mix of valid and invalid device numbers. All devices before the invalid value will be enumerated, while all devices after the invalid value will be masked.

To determine the device ID for the available hardware in your system, you can run NVIDIA’s deviceQuery executable included in the CUDA SDK. Happy programming!

Chris Mason

## Using CUDA

Here are a couple guides for [[attachment:gpu_computing_intro.pdf|GPU computing]] and [[attachment:cuda_intro.pdf|CUDA]]

### Example Written by Jackie Schwehr

Below is an example CUDA program written by Jackie Schwehr.

### Environment variables

```bash
module load cuda/7.5  # module add cuda/7.5 works too
module add compilers/gcc4.9.4
export PATH=/physics/INSTALLATION/bin:$PATH
export LD_LIBRARY_PATH=/physics/INSTALLATION/lib64:$LD_LIBRARY_PATH
```

### Hello World CUDA File

Download the CUDA file here

```C++
//////////////////////////////////////////////////////////////////////
//PURPOSE:
//
//    Prove that cuda works on the engineering cluster by 
//    running a hello world example.
//
// [      
// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA with an array
// of offsets. Then the offsets are added in parallel to produce the string "World!"
// By Ingemar Ragnemalm 2010
// ]     
//
//CALLING SEQUENCE:
//    ./cudaHelloWorld   
//
//INPUT PARAMETERS:
//    none   
//
//OUTPUTS:
//    to screen: "Hello Hello" if it fails, "Hello World!" if it succeeds
//
//EXAMPLES:
//    compile: nvcc -o cudaHelloWorld cudaHelloWorld.cu
//    execute: ./cudaHelloWorld
//
//MODIFICATIONS:
//    2018/06/15 - documented by jschwehr
//
//////////////////////////////////////////////////////////////////////
//
//   Additional notes for making this work! 
//   
//   First, you need cuda!  You can get this by loading the module with:
//   module add cuda/7.5
//
//   Next you need gcc - but none of the modules present work with cuda, 
//   so go ahead and adjust your path and ld_library_path to point to 
//   the physics install:
//   export PATH=/physics/INSTALLATION/bin:$PATH
//   export LD_LIBRARY_PATH=/physics/INSTALLATION/lib64:$LD_LIBRARY_PATH
//
//   Now you should be able to compile (with nvcc, as described above)
//   Note that you'll need to be on hpc or the sandbox node for this
//   
//   To run, you need to be on one of the gpu nodes.  You should be able
//   to ssh to them, but they could be full, run a 'top' to check:
//   ssh gpu9
//   top
//
//   On the gpu node, the executable you creading by compiling should work
//   just fine. (you won't be able to compile on the gpu node, so if you
//   want to change something, do it from the sandbox or ens-hpc node)
//
//   I also wrote an example python script to submit jobs to the gpu queue
//
//   ~ Jackie

//    Now a little about the code

#include <stdio.h>

// playing with a string of 16 characters, each of which will be 'acted' on in this example, so there are going to be 16 threads (or processes) that we want. 
const int N = 16; 
// blocks are groups of threads.  for cuda, these need to be less than 1024 threads.  for this example, we'll put all 16 threads into one block, so the block size is 16.
const int blocksize = 16; 

//  the function that is going to be run on the gpu nodes is this one - it appends the initial string with a new string built by incrementing the original string based on a given vector. 
__global__ 
void hello(char *a, int *b) 
{
	a[threadIdx.x] += b[threadIdx.x];
}
 
int main()
{

	// start by defining the initial character array
	char a[N] = "Hello \0\0\0\0\0\0";
	// define the array that will convert 'hello' into 'world'
	int b[N] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	// start with the initial 'hello' - if nothing else works, this is all that will be output, but hopefully there will be more...
	printf("%s", a);

	// cuda is going to need to know how much memory to use.
	// build the memory needed by getting the memory of one element and multiplying by how many units are needed. 
	char *ad;
	int *bd;
	const int csize = N*sizeof(char);
	const int isize = N*sizeof(int);
 
	// assign the memory
	cudaMalloc( (void**)&ad, csize ); 
	cudaMalloc( (void**)&bd, isize ); 
	cudaMemcpy( ad, a, csize, cudaMemcpyHostToDevice ); 
	cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice ); 

	// the dimension of the block is 16 x 1 - you can do other combinations if you want, but this is simple.
	dim3 dimBlock( blocksize, 1 );
	// blocks can be combined into grids, but for this example there is just one block in the grid, so the dimensions of the grid are 1 block x 1 block
	dim3 dimGrid( 1, 1 );

	// now do the thing!
	hello<<<dimGrid, dimBlock>>>(ad, bd);
	cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost ); 

	// clean up the memory after you are done
	cudaFree( ad );
	cudaFree( bd );

	// print the resulting string- should be 'World!'
	printf("%s\n", a);
	return EXIT_SUCCESS;
}
```

### Submit Job Script

The script file here does NOT contain the `-l cuda=1` option

```python
#!/usr/bin/python/
######################################################################
##PURPOSE:
##  Submit the cudaHelloWorld script to the gpu queue
##
##CALLING SEQUENCE:
##  submitCuda
##
##HARDCODED INPUT PARAMETERS:
##  sourceCode: path to script that loads the cuda module and adds gcc
##              to the path and ld_library_path
##  storepath: path to where to store output text file
##  runpath: path to where to store the submit scripts
##  codePath: path to where the exe file lives
##
##  different .exe file for normal/fsi/minoo because
##  of different reweight tree locations
##
##OUTPUTS:
##  submit script: storepath/submit/submit-dateTime-cudaHelloWorld.sh
##                 this is the script that is submitted to the queue
##  output file:  storepath/helloWorldOutput-dateTime.out
##                this is the output of the script,
##                hopefully: Hello World!
##
##EXAMPLES:
##  (on ens-hpc)
##  python submitCudaHelloWorld.py
##
##MODIFICATIONS:
##       2018/06/15 - created by jschwehr
##
######################################################################

import sys
import os
import string
import time

########### INPUTS ###################
## path to script that defines P0DANALYSISROOT
sourceCode = "source /home/other/jschwehr/multi/sourceThisForCuda.sh"
## path to where to store output
storepath = "/home/other/jschwehr/multi/gpuQueueTest"
## path to where to store the submit scripts, and where the code will run
runpath = storepath+"/submit/"
## path to where the exe file lives
codePath = "/home/other/jschwehr/multi/"
######################################

## check if paths exist, and create if they don't
if not os.path.exists(storepath):
    os.makedirs(storepath)
if not os.path.exists(runpath):
    os.makedirs(runpath)


## change directories
os.chdir(runpath)

## get a timestamp string to mark the submit script
dateTime = time.strftime("%m%d%H%M")

## open the script that will be submitted to the queue
subfilename = "submit-"+dateTime+"-cudaHelloWorld.sh"
qsubfile = open(subfilename,"w")

## define the queue parameters to use for the job
## this option specifies the specific gpu nodes
#qsubfile.write("#!/bin/bash\n##Job settings\n#$ -j yes\n#$ -p -10\n#$ -l h_data=1500m\n#$ -l h_cpu=172800\n#$ -l hostname=\"gpu1|gpu2|gpu3|gpu4|gpu5|gpu6|gpu7|gpu8|gpu9|gpu10|gpu11|gpu12|gpu13\"\n#$ -o "+runpath+"\n\n")
## this option specifies the gpu queue
qsubfile.write("#!/bin/bash\n##Job settings\n#$ -j yes\n#$ -p -10\n#$ -l h_data=1500m\n#$ -l h_cpu=172800\n#$ -l qname=\"gpu.q\"\n#$ -o "+runpath+"\n\n")

# source the setup script
qsubfile.write(sourceCode+"\n")

# change to the directory to store the output
qsubfile.write("cd "+storepath+"\n")

# run the code
qsubfile.write(codepath+"/cudaHelloWorld &> helloWorldOutput-"+dateTime+".out \n")

# close the submit script
qsubfile.close()

# submit the job to the queue
os.system("qsub "+subfilename)

print "done"
```
