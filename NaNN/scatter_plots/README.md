### README.md

- To create the scatter plots maps edit and run the scatter_plot.py script. You may need to symb link the D.nii.gz and adc.nii.gz input files first.

- E.g.:
ln -s ../adc/dwi_synth_adc.nii.gz
ln -s ../syn_data/D.nii.gz
ln -s ../syn_data/sigma_map.nii.gz sigma.nii.gz
python scatter_plot.py
