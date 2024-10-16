# Revisiting-Delta-Parameter-Pruning-For-Fine-Tuned-Models

code coming soon


#### find q encoder analytically
python find_q_encoder.py --l_bound 0.1 --u_bound 0.5 --step_size 0.1  --device cuda:1 --analytical  --finetuned_model cola model  --dataset_name cola  --param 0.37

python find_q_encoder.py --l_bound 0.3 --u_bound 0.7 --step_size 0.1  --device cuda:1 --analytical  --finetuned_model sst2 model  --dataset_name sst2 --param 1.01

python find_q_encoder.py --l_bound 0.3 --u_bound 0.7 --step_size 0.1  --device cuda:1 --analytical  --finetuned_model mrpc model  --dataset_name mrpc --param 0.27

python find_q_encoder.py --l_bound 0.3 --u_bound 0.7 --step_size 0.1  --device cuda:1 --analytical  --finetuned_model stsb model  --dataset_name stsb --param 0.36