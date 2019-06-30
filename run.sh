# chmod 755 run.sh
# script -c ./run.sh /dev/null | tee output.log
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export OMP_NUM_THREADS=16
python meta_train.py --demo_dir="../../data/mil/data_mini/sim_push/" --state_path="../../data/mil/data/scale_and_bias_sim_push.pkl"
