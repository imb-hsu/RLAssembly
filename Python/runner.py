import main
from stable_baselines3 import PPO

if __name__ == "__main__":

    #Single Run

    main.forwards(PPO, 250000, 4000, learning_rate=0.0003, net_arch=[128 ,128],activation_fn="ReLU", env_num = 1)
    main.forwards(PPO, 250000, 4001, learning_rate=0.0003, net_arch=[128 ,128],activation_fn="ReLU", env_num = 1)
    main.forwards(PPO, 250000, 4002, learning_rate=0.0003, net_arch=[128 ,128],activation_fn="ReLU", env_num = 1)
    #main.forwards(PPO, 250000, 3000, learning_rate=0.0003, net_arch=[128 ,128],activation_fn="ReLU", env_num = 2)
    #main.forwards(PPO, 250000, 3001, learning_rate=0.0003, net_arch=[128 ,128],activation_fn="ReLU", env_num = 2)
    #main.forwards(PPO, 250000, 3002, learning_rate=0.0003, net_arch=[128 ,128],activation_fn="ReLU", env_num = 2)

    #Short Runs
    """
    # Net Archs for the runs
    net_arch1 = [64,64] 
    net_arch2 = dict(pi=[128, 128, 64], vf=[64, 64, 64])
    net_arch3 = [128,128,128]
    net_archs = [net_arch1, net_arch2, net_arch3] 

    # Learning rates
    learning_rate1 = 0.003
    learning_rate2 = 0.0003
    learning_rate3 = 0.00003
    learning_rates = [learning_rate1, learning_rate2, learning_rate3] 

    # Activation functions
    activation_fn1 = "Tanh"
    activation_fn2 = "ReLU"
    activation_fns =[activation_fn1, activation_fn2] 

    seed = 2000


    for net_arch in net_archs:
        for learning_rate in learning_rates:
            for activation_fn in activation_fns:
                main.forwards(PPO, 250000, seed, learning_rate=learning_rate, net_arch=net_arch,activation_fn=activation_fn)
                seed = seed+1
                print("================================================================================================")
                print("================================================================================================")
    """
    """
    # Long Runs

    # Net Archs for the runs
    net_arch1 = [128,128,128]
    net_arch2 = [256,256, 128,128]
    net_archs = [net_arch1, net_arch2] 


    # Activation functions
    activation_fn1 = "Tanh"
    activation_fn2 = "ReLU"
    activation_fns =[activation_fn1, activation_fn2] 

    seed = 2000


    for net_arch in net_archs:
        for activation_fn in activation_fns:
            main.forwards(PPO, 500000, seed, learning_rate=0.0003, net_arch=net_arch,activation_fn=activation_fn)
            seed = seed+1
            print("================================================================================================")
            print("================================================================================================")
    """