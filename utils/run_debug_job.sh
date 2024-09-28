#!/bin/bash

cd ~/multi-gpu-path-tracer
mkdir -p tmp
output_file="tmp/debug-job-output.txt"
prev_output_file="tmp/debug-job-output-prev.txt"
touch "$output_file" "$prev_output_file" 

jobId=$(sbatch --output="$output_file" --error="$output_file" ./scripts/run_job.sh "$@" | awk '{print $4}')

show_new_lines() {
  diff "$prev_output_file" "$output_file" | grep ">" | cut -c 3-
}

get_job_status() {
  hpc-jobs | grep $jobId | awk '{print $4}'
}

cleanup() {
  scancel $jobId
  rm -f "$output_file" "$prev_output_file"
}

trap cleanup EXIT

has_notified_about_pending_state=false
while true; do
  if ! cmp -s "$output_file" "$prev_output_file"; then
    show_new_lines
    cp "$output_file" "$prev_output_file"
  fi

  job_status=$(get_job_status)

  if [ "$job_status" == "COMPLETED" ]; then
    echo "Job completed successfully"
    break
  elif [ "$job_status" == "FAILED" ]; then
    echo "Job failed"
    break
  elif [ "$job_status" == "PENDING" ]; then
    if [ "$has_notified_about_pending_state" = false ]; then
      echo "Job is pending..."
      has_notified_about_pending_state=true
    fi
  fi  

  sleep 0.3
done