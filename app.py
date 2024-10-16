from flask import Flask, render_template, request, jsonify
import subprocess
import threading

app = Flask(__name__)

# Function to execute the Python script
def execute_python_script(script_filename):
    try:
        subprocess.run(['python', script_filename])
    except Exception as e:
        print(f'Error executing Python script {script_filename}: {e}')

# Route to serve heartposes.html
@app.route('/')
def index():
    return render_template('heartposes.html')

# Route to handle the AJAX request for executing Python script for a specific pose
@app.route('/execute_python_script/<int:pose_number>', methods=['GET'])
def execute_python_script_route(pose_number):
    # Define the script filename based on the pose number
    script_filename = f'pose{pose_number}.py'
    
    # Function to execute the Python script for the specific pose
    def execute_specific_python_script():
        try:
            execute_python_script(script_filename)
        except Exception as e:
            print(f'Error executing Python script {script_filename}: {e}')

    # Create a new thread for executing the Python script
    thread = threading.Thread(target=execute_specific_python_script)
    thread.start()
    return jsonify({'success': True})

if __name__ == '__main__':
    # Run the Flask app using the built-in development server
    app.run(debug=True)