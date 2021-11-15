python main.py --dataset mnist --com_round 20  --shards 2 --lr 0.001 --logDir /home/fengchen/Data/log/,1217-001.txt --device cuda:1 --agg avg --serveralpha 1 --reg 0
python main.py --dataset mnist --com_round 20  --shards 5 --lr 0.001 --logDir /home/fengchen/Data/log/,1217-002.txt --device cuda:1 --agg avg --serveralpha 1 --reg 0
python main.py --dataset mnist --com_round 20  --shards 10 --lr 0.001 --logDir /home/fengchen/Data/log/,1217-003.txt --device cuda:1 --agg avg --serveralpha 1 --reg 0
python main.py --dataset mnist --com_round 20  --shards 20 --lr 0.001 --logDir /home/fengchen/Data/log/,1217-004.txt --device cuda:1 --agg avg --serveralpha 1 --reg 0
python main.py --dataset mnist --com_round 20  --shards 50 --lr 0.001 --logDir /home/fengchen/Data/log/,1217-005.txt --device cuda:1 --agg avg --serveralpha 1 --reg 0
