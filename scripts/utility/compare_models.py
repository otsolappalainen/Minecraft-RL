import torch as th
from stable_baselines3 import DQN
import os
from zipfile import ZipFile
import io

def load_model_state_dict(model_path, device='cpu'):
    """
    Loads the state_dict from a SB3 model zip file.
    """
    with ZipFile(model_path, 'r') as zip_ref:
        # SB3 models save the policy parameters in 'policy.pth'
        with zip_ref.open('policy.pth') as f:
            buffer = io.BytesIO(f.read())
            state_dict = th.load(buffer, map_location=device)
    return state_dict

def compare_models(model_path_before, model_path_after, device='cpu'):
    """
    Compares the state_dict of two SB3 DQN models.
    Prints out the parameters that have changed.
    """
    state_dict_before = load_model_state_dict(model_path_before, device)
    state_dict_after = load_model_state_dict(model_path_after, device)
    
    changed_params = []
    unchanged_params = []
    
    for key in state_dict_before:
        if key in state_dict_after:
            param_before = state_dict_before[key]
            param_after = state_dict_after[key]
            if not th.equal(param_before, param_after):
                changed_params.append(key)
            else:
                unchanged_params.append(key)
        else:
            print(f"Parameter {key} not found in the after model.")
    
    print(f"\nTotal Parameters Compared: {len(state_dict_before)}")
    print(f"Changed Parameters: {len(changed_params)}")
    print(f"Unchanged Parameters: {len(unchanged_params)}\n")
    
    if changed_params:
        print("Parameters that have changed:")
        for param in changed_params:
            print(f"  - {param}")
    else:
        print("No parameters have changed.")

    return changed_params, unchanged_params

def main_comparison():
    MODEL_PATH_DQN = r"E:\DQN_BC_MODELS\models_dqn_from_bc"

    # Paths to the models before and after training
    model_before = os.path.join(MODEL_PATH_DQN, "initial_model.zip")  # Replace with your initial model path
    model_after = os.path.join(MODEL_PATH_DQN, "final_model.zip")    # Replace with your final model path

    if not os.path.exists(model_before):
        print(f"Model before training not found at {model_before}")
        return
    if not os.path.exists(model_after):
        print(f"Model after training not found at {model_after}")
        return

    changed, unchanged = compare_models(model_before, model_after, device='cpu')

    if changed:
        print("\nGradients have likely flowed, as model parameters have been updated.")
    else:
        print("\nNo changes detected in model parameters. Gradients may not be flowing.")

if __name__ == "__main__":
    main_comparison()
