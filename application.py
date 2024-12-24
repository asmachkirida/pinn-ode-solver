import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch

# Add model directory to sys.path dynamically
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

# Import the model classes from their respective files
from first_ode import Network  # First-order model from first_ode.py
from second_ode import Network2  # Second-order model from second_ode.py

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Function to load the appropriate model
def load_model(model_name):
    model = None
    if model_name == "first_order":
        model = Network()  # Use the Network class from first_ode.py
        model.load_state_dict(torch.load('model_first_order.pth'))  # Load the first-order model
    elif model_name == "second_order":
        model = Network2()  # Assuming Network2 is defined in second_ode.py
        model.load_state_dict(torch.load('model_second_order.pth'))  # Load the second-order model
    model.eval()  # Set the model to evaluation mode
    return model

@app.route('/solve-ode', methods=['POST'])
def solve_ode():
    try:
        # Get JSON data from request
        data = request.get_json()
        ode_type = data.get('ode_type')
        x_start = float(data.get('x_start'))
        x_end = float(data.get('x_end'))

        # Validate inputs
        if ode_type not in ['first_order', 'second_order']:
            return jsonify({"error": "Invalid ODE type"}), 400
        if x_start >= x_end:
            return jsonify({"error": "Invalid range: x_start must be less than x_end"}), 400

        # Load the model based on the ODE type
        model = load_model(ode_type)

        # Generate x values
        x = torch.linspace(x_start, x_end, 100)[:, None]

        # Solve ODE
        with torch.no_grad():
            solution = model(x).cpu().numpy()

        # Return the solution
        return jsonify({"x": x.detach().cpu().numpy().tolist(), "solution": solution.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
