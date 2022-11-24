
linucb_parameter={'lambda':1.2,'sigma':0.05, 'alpha': 0.5}
arm_con_para={'lambda':1.2,'sigma':0.1, 'alpha': 0.5}
linucb_para={'lambda':0.15,'sigma':0.05, 'alpha': 0.1} 
# conucb_para={'lambda':0.5, 'sigma':0.05, 'tilde_lambda':1, 'alpha': 0.25, 'tilde_alpha': 0.25}
conucb_para={'lambda':0.5, 'sigma':0.05, 'tilde_lambda':1, 'alpha': 0.25, 'tilde_alpha': 0.25}
# linucb_force_exploration_para={'lambda':0.1,'sigma':0.05, 'alpha': 0.1}
linucb_force_exploration_para={'lambda':0.15,'sigma':0.05, 'alpha': 0.1}
linucb_conucb_para={'lambda':0.1, 'sigma':0.05, 'tilde_lambda':1, 'alpha': 0.1}
linucb_more_info={'lambda':0.2,'sigma':0.05, 'alpha': 0.1}
train_iter=0
test_iter=1000
armNoiseScale=0.1
suparmNoiseScale=0.1
batch_size=50
bt= lambda t: int(0.1*t)
#bt= lambda t: 10*int(0.02*t)
seeds_set=[2756048, 675510, 807110,2165051, 9492253, 927,218,495,515,452]
