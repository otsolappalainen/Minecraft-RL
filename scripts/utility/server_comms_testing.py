import websocket
import json
import time

# Define the list of actions
ACTIONS = [
    "move_forward",
    "move_left",
    "move_right",
    "jump_walk_forward",
    "jump",
    "look_left",
    "look_right",
    "look_up",
    "look_down",
    "attack",
    "reset 2"
]

SERVER_URI = "ws://localhost:8080"

def send_action(ws, action):
    """
    Sends an action message through the WebSocket and prints the response excluding 'surrounding_blocks'.
    """
    message = {"action": action}
    try:
        # Send the action message as JSON
        ws.send(json.dumps(message))
        # Receive the response from the server
        response = ws.recv()
        data = json.loads(response)
        # Remove 'surrounding_blocks' from the data if it exists
        if 'surrounding_blocks' in data:
            del data['surrounding_blocks']
        # Pretty-print the remaining data
        print(json.dumps(data, indent=4))
    except websocket.WebSocketConnectionClosedException:
        print("Connection closed by the server.")
    except json.JSONDecodeError:
        print("Received non-JSON response:", response)
    except Exception as e:
        print(f"An error occurred: {e}")

def establish_connection():
    """
    Establishes a WebSocket connection to the server.
    Returns the WebSocket object if successful, else None.
    """
    try:
        ws = websocket.create_connection(SERVER_URI)
        print(f"Connected to {SERVER_URI}")
        return ws
    except Exception as e:
        print(f"Failed to connect to the server: {e}")
        return None

def close_connection(ws):
    """
    Closes the WebSocket connection gracefully.
    """
    try:
        ws.close()
        print("WebSocket connection closed.")
    except Exception as e:
        print(f"Error while closing the connection: {e}")

def main():
    """
    Main loop that continuously prompts the user for actions and sends them to the WebSocket server.
    Exits only when the user types 'exit'.
    """
    while True:
        print("\nAvailable actions:")
        for idx, action in enumerate(ACTIONS, 1):
            print(f"{idx}. {action}")
        
        action_input = input("Enter the action (name or number, or 'exit' to quit): ").strip()
        
        if action_input.lower() == 'exit':
            print("Exiting the script.")
            break

        # Determine the action based on user input
        if action_input.isdigit():
            action_idx = int(action_input) - 1
            if 0 <= action_idx < len(ACTIONS):
                action = ACTIONS[action_idx]
            else:
                print("Invalid action number. Please try again.")
                continue
        elif action_input in ACTIONS:
            action = action_input
        else:
            print("Invalid action. Please enter a valid action name or number.")
            continue
        
        # Prompt for the number of times to send the action
        try:
            times_input = input("How many times to send this action: ").strip()
            times = int(times_input)
            if times < 1:
                print("Number must be at least 1. Please try again.")
                continue
        except ValueError:
            print("Invalid number. Please enter an integer value.")
            continue
        
        # Establish WebSocket connection
        ws = establish_connection()
        if ws is None:
            # If connection failed, prompt for the next action
            continue
        
        # Send the action the specified number of times
        for i in range(1, times + 1):
            print(f"\nSending action {i}/{times}: {action}")
            send_action(ws, action)
            time.sleep(0.05)  # Wait for 40 milliseconds
        
        # Close the WebSocket connection after sending all messages
        close_connection(ws)

if __name__ == "__main__":
    main()