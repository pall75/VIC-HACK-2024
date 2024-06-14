### README.md

- To create adc maps edit and run the lmse_adc.py script. You may need to symb link the dwi/bval/bvec input files first. 

- E.g.:
ln -s ../syn_data/dwi_synth.nii.gz 
ln -s ../syn_data/dwi_synth.bval 
ln -s ../syn_data/dwi_synth.bvec 
python lmse_adc.py
