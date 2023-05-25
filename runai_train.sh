#!/bin/bash
# Running
#	bash runai_interactive.sh
# will create a job yourname-inter which 
# * has "interactive priority"
# * uses 0.5 GPU (customizable)
# * starts a jupyter server at port 8888 with default password "hello"
# * runs for 8 hours
#
# Optionally you can give the name a suffix:
#	bash runai_interactive.sh 1
# will create yourname-inter1
#
# Before starting a new interactive job, delete the previous one:
#	runai delete yourname-inter

# Customize before using:
# * CLUSTER_USER and CLUSTER_USER_ID
# * MY_WORK_DIR
# * MY_GPU_AMOUNT - fraction of GPU memory to allocate. Our GPUs usually have 32GB, so 0.25 means 8GB and 0.5 means 16GB.
# * JUPYTER_CONFIG_DIR if you want to configure jupyter (for example change password)


CLUSTER_USER=sooh # find this by running `id -un` on iccvlabsrv
CLUSTER_USER_ID=255692 # find this by running `id -u` on iccvlabsrv
CLUSTER_GROUP_NAME=DHLAB-unit # find this by running `id -gn` on iccvlabsrv
CLUSTER_GROUP_ID=11703 # find this by running `id -g` on iccvlabsrv

MY_IMAGE="ic-registry.epfl.ch/dhlab/sooh_test"
# MY_IMAGE="ubuntu:22.04"
arg_job_name="$CLUSTER_USER-fine"

echo "Job [$arg_job_name]"

runai submit $arg_job_name \
	-i $MY_IMAGE \
	--gpu 4 \
	--pvc=runai-dhlab-sooh-data1:/data1 \
	-e USER=$CLUSTER_USER \
	-e USER_ID=$CLUSTER_USER_ID \
	-e CLUSTER_GROUP_NAME=$CLUSTER_GROUP_NAME \
	-e CLUSTER_GROUP_ID=$CLUSTER_GROUP_ID \
	--command -- entrypoint.sh \
	-- sleep infinity 

# --node-type G10 \
# check if succeeded
if [ $? -eq 0 ]; then
	runai describe job $arg_job_name

	echo ""
	echo "Connect - terminal:"
	echo "	runai bash $arg_job_name"	
fi

