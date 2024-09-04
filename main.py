from Agent.env import CustomEnv
from Utils.get_data import get_data
env = CustomEnv()
while True:
    # Get action input from the user
    user_input = int(input("Enter action (0, 1, or 2): "))  # Assuming actions are integers
    assert user_input in [0, 1, 2], "Invalid action"  # Validate the input

    # Call step method on the environment instance
    next_state, reward, done = env.step(user_input)
    data = get_data('./Data')
    drop = ['timestamp_o', 'timestamp_cl', 'ignore']
    data.drop(columns=drop, inplace=True)
    state = data.iloc[200:200+42]
    print("columns:", len(state.columns))
    # Print information
    # print("Next state:", next_state.columns)
    # print("Next state:", len(next_state.columns))
    # print("Reward:", reward)
    # print("Done:", done)

    # Check if episode is finished
    if done:
        print("Episode finished")
        break  # End the loop