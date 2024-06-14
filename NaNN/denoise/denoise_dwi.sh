#!/bin/bash

# Usage function to display help
usage() {
    echo "Usage: $0 -i <input_dataset> [-m <method>]"
    echo "  -i <input_dataset>    Specify the base name of the diffusion-weighted MRI dataset (without extensions)."
    echo "  -m <method>           Method used for noise estimation and denoising. Default is 'mppca'."
    echo "Example: $0 -i dwi -m mppca"
    exit 1
}

# Default method
method="mppca"

# Parse command-line options
while getopts "i:m:h" opt; do
    case ${opt} in
        i ) input_dataset="$OPTARG" ;;
        m ) method="$OPTARG" ;;
        h ) usage ;;
        \? ) usage ;;
    esac
done

# Check if input dataset is provided
if [ -z "$input_dataset" ]; then
    echo "Error: Input dataset not specified."
    usage
fi

# Validate method
if [ "$method" != "mppca" ]; then
    echo "Error: Unsupported method '$method'. Currently, only 'mppca' is supported."
    exit 1
fi

# Filename preparations
input_nii="${input_dataset}.nii.gz"
output_denoised="${input_dataset}_denoised.nii.gz"
output_noise="${input_dataset}_sigma.nii.gz"

# Check if the necessary files exist
if [ ! -f "$input_nii" ]; then
    echo "Error: File $input_nii does not exist."
    exit 1
fi

# Denoising process
echo "Starting denoising process using $method method..."
if [ "$method" == "mppca" ]; then
    dwidenoise "$input_nii" "$output_denoised" -noise "$output_noise"
    echo "Denoising completed. Output files are $output_denoised and $output_noise."
else
    echo "Unsupported method specified."
    exit 1
fi

echo "Processing completed successfully."

