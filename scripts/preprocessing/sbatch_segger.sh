#!/bin/bash

# script to submit slurm jobs to run segger on a set of samples
# usage: ./submit_segger_jobs.sh <sample_list.txt> <input_directory> <output_directory> <segger_parameters>
#
# This script submits jobs in batches of 4 GPUs at a time, with dependencies
# so that only 4 GPU jobs run simultaneously to avoid hogging the queue
# example command:
# ./submit_segger_jobs.sh sample_list.txt /data/input /data/output "--param1


INPUT_FOLDER=''

## normal command to get all samples in input folder
#mapfile -t SAMPLES < <(ls -1 $INPUT_FOLDER | grep output ) # get 10x output folders

## modify samples to run only those that need reprocessing
## array of sample names
SAMPLES=(

)

OUTPUT_FOLDER=''

# print sample list as a check
echo "Sample list:"
echo "${SAMPLES[@]}"
echo ""

# Maximum number of concurrent GPU jobs
MAX_CONCURRENT_GPUS=4

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_FOLDER

# Array to store job IDs
declare -a JOB_IDS=()

# Counter for tracking batch position
BATCH_COUNTER=0

# Read samples into array
#mapfile -t SAMPLES < "$SAMPLE_LIST"

echo "Total samples to process: ${#SAMPLES[@]}"
echo "Max concurrent GPU jobs: ${MAX_CONCURRENT_GPUS}"
echo ""

mkdir -p "${OUTPUT_FOLDER}/jobs"

for SAMPLE in "${SAMPLES[@]}"; do
    echo "Preparing job for sample: ${SAMPLE}"
    # make output directory for this sample if it doesn't exist
    mkdir -p "${OUTPUT_FOLDER}/${SAMPLE}"

    # Create a SLURM job script
    JOB_SCRIPT="${OUTPUT_FOLDER}/jobs/segger_job_${SAMPLE}.sh"
    echo "#!/bin/bash" > $JOB_SCRIPT
    echo "#SBATCH --job-name=segger_${SAMPLE}" >> $JOB_SCRIPT
    echo "#SBATCH --output=${OUTPUT_FOLDER}/${SAMPLE}/segger_${SAMPLE}.out" >> $JOB_SCRIPT
    echo "#SBATCH --error=${OUTPUT_FOLDER}/${SAMPLE}/segger_${SAMPLE}.err" >> $JOB_SCRIPT
    echo "#SBATCH -n 8 --ntasks-per-node=8 --mem-per-cpu=30G --time=24:00:00" >> $JOB_SCRIPT
    echo "#SBATCH --gres=gpu:1" >> $JOB_SCRIPT
    echo "" >> $JOB_SCRIPT
    echo "source activate segger" >> $JOB_SCRIPT
    ## normal segmentation command with defaults
    ### directory structure will differ for some samples. If the sample name begins with XeniumRanger then the input folder is $INPUT_FOLDER/outputs/${SAMPLE}/
    if [[ $SAMPLE == XeniumRanger* ]]; then
        echo "segger segment --max-nodes-per-tile 55000  -i $INPUT_FOLDER/${SAMPLE}/outputs/ -o $OUTPUT_FOLDER/${SAMPLE}/" >> $JOB_SCRIPT
    else
        echo "segger segment --max-nodes-per-tile 55000  -i $INPUT_FOLDER/${SAMPLE}/ -o $OUTPUT_FOLDER/${SAMPLE}/" >> $JOB_SCRIPT
    fi

    # Determine if this job needs a dependency
    if [ $BATCH_COUNTER -ge $MAX_CONCURRENT_GPUS ]; then
        # This job should depend on the job from the previous batch
        # Calculate which previous job to depend on
        DEPENDENCY_INDEX=$((BATCH_COUNTER - MAX_CONCURRENT_GPUS))
        DEPENDENCY_JOB_ID=${JOB_IDS[$DEPENDENCY_INDEX]}

        echo "Submitting job for sample: ${SAMPLE} (depends on job ${DEPENDENCY_JOB_ID})"
        JOB_OUTPUT=$(sbatch --dependency=afterany:${DEPENDENCY_JOB_ID} $JOB_SCRIPT)
    else
        # No dependency needed, submit immediately
        echo "Submitting job for sample: ${SAMPLE} (batch ${BATCH_COUNTER})"
        JOB_OUTPUT=$(sbatch $JOB_SCRIPT)
    fi

    # Extract job ID from sbatch output
    JOB_ID=$(echo $JOB_OUTPUT | awk '{print $4}')
    JOB_IDS+=($JOB_ID)

    # Increment batch counter
    BATCH_COUNTER=$((BATCH_COUNTER + 1))
done

echo ""
echo "Submitted ${#JOB_IDS[@]} jobs total"
echo "First ${MAX_CONCURRENT_GPUS} jobs will run immediately"
echo "Remaining jobs will start as previous jobs finish"
echo ""
echo "Monitor with: squeue -u \$USER -p gpu"
