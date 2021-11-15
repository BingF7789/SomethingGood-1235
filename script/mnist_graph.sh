python main.py --dataset mnist --com_round 20  --shards 2 --lr 0.001 --logDir /home/fengchen/Data/log/,1217-021.txt --device cuda:1 --agg graph --serveralpha 1 --reg 1
python main.py --dataset mnist --com_round 20  --shards 5 --lr 0.001 --logDir /home/fengchen/Data/log/,1217-022.txt --device cuda:1 --agg graph --serveralpha 1 --reg 1
python main.py --dataset mnist --com_round 20  --shards 10 --lr 0.001 --logDir /home/fengchen/Data/log/,1217-023.txt --device cuda:1 --agg graph --serveralpha 1 --reg 1
python main.py --dataset mnist --com_round 20  --shards 20 --lr 0.001 --logDir /home/fengchen/Data/log/,1217-024.txt --device cuda:1 --agg graph --serveralpha 1 --reg 1
python main.py --dataset mnist --com_round 20  --shards 50 --lr 0.001 --logDir /home/fengchen/Data/log/,1217-025.txt --device cuda:1 --agg graph --serveralpha 1 --reg 1
