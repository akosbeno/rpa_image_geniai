from typing import Annotated, List, Tuple
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from dotenv import load_dotenv
import os 
import matplotlib.pyplot as plt
import numpy as np
import time
import csv

# Set the backend to Agg
plt.switch_backend('Agg')

load_dotenv() 

__tavily_api = os.getenv("TAVILY_API_KEY")

tavily_tool = TavilySearchResults(max_results=5, tavily_api_key=__tavily_api)

# Warning: This executes code locally, which can be unsafe when not sandboxed

repl = PythonREPL()

@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    print(code)
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )

@tool
def plot_line_graph(
    senzor: Annotated[int, "The sensor value that triggers the light scene. The function will only run if senzor is 1."],
    r1: Annotated[int, "The red component of the LED strip color during ramp-up (0-255)."],
    g1: Annotated[int, "The green component of the LED strip color during ramp-up (0-255)."],
    b1: Annotated[int, "The blue component of the LED strip color during ramp-up (0-255)."],
    r2: Annotated[int, "The red component of the LED strip color during ramp-down (0-255)."],
    g2: Annotated[int, "The green component of the LED strip color during ramp-down (0-255)."],
    b2: Annotated[int, "The blue component of the LED strip color during ramp-down (0-255)."],
    ramp_up_time: Annotated[int, "Time in milliseconds to ramp up from 0% to 100% brightness."],
    constant_time: Annotated[int, "Time in milliseconds to maintain 100% brightness."],
    ramp_down_time: Annotated[int, "Time in milliseconds to ramp down from 100% to 0% brightness."],
    off_time: Annotated[int, "Time in milliseconds to maintain 0% brightness."],
    repetitions: Annotated[int, "Number of times to repeat the entire sequence."]
):
    """Controls the LED strip and generates a brightness graph over time with different colors for ramp-up and ramp-down phases."""
    if senzor != 1:
        print("Sensor is not active. Exiting.")
        return

    total_time = (ramp_up_time + constant_time + ramp_down_time + off_time) * repetitions
    time_points = np.linspace(0, total_time, num=1000)
    brightness = np.zeros_like(time_points)
    colors = np.zeros((1000, 3))

    single_cycle_time = ramp_up_time + constant_time + ramp_down_time + off_time
    for i in range(repetitions):
        start_time = i * single_cycle_time
        end_ramp_up = start_time + ramp_up_time
        end_constant = end_ramp_up + constant_time
        end_ramp_down = end_constant + ramp_down_time
        end_off = end_ramp_down + off_time

        ramp_up_indices = (time_points >= start_time) & (time_points < end_ramp_up)
        constant_indices = (time_points >= end_ramp_up) & (time_points < end_constant)
        ramp_down_indices = (time_points >= end_constant) & (time_points < end_ramp_down)
        off_indices = (time_points >= end_ramp_down) & (time_points < end_off)

        brightness[ramp_up_indices] = np.linspace(0, 1, num=np.sum(ramp_up_indices))
        brightness[constant_indices] = 1
        brightness[ramp_down_indices] = np.linspace(1, 0, num=np.sum(ramp_down_indices))
        brightness[off_indices] = 0

        colors[ramp_up_indices] = [r1 / 255, g1 / 255, b1 / 255]
        colors[ramp_down_indices] = [r2 / 255, g2 / 255, b2 / 255]

    plt.figure()
    for i in range(len(time_points) - 1):
        plt.plot(time_points[i:i+2], brightness[i:i+2] * 100, color=colors[i])
    plt.xlabel('Time (ms)')
    plt.ylabel('Brightness (%)')
    plt.title('LED Strip Brightness Over Time')
    plt.savefig('led_strip_brightness_colored.png')  # Save the plot to a file
    print("Plot saved as 'led_strip_brightness_colored.png'")
    
@tool
def generate_csv(
    senzor: Annotated[int, "The sensor value that triggers the light scene. The function will only run if senzor is 1."],
    ramp_up_time: Annotated[int, "Time in milliseconds to ramp up from 0% to 100% brightness."],
    constant_time: Annotated[int, "Time in milliseconds to maintain 100% brightness."],
    ramp_down_time: Annotated[int, "Time in milliseconds to ramp down from 100% to 0% brightness."],
    off_time: Annotated[int, "Time in milliseconds to maintain 0% brightness."],
    repetitions: Annotated[int, "Number of times to repeat the entire sequence."]
):
    """Generates a CSV file with the points that change the shape of the line for the LED sequence graph."""
    if senzor != 1:
        print("Sensor is not active. Exiting.")
        return

    total_time = (ramp_up_time + constant_time + ramp_down_time + off_time) * repetitions
    single_cycle_time = ramp_up_time + constant_time + ramp_down_time + off_time

    csv_data = []

    for i in range(repetitions):
        start_time = i * single_cycle_time
        end_ramp_up = start_time + ramp_up_time
        end_constant = end_ramp_up + constant_time
        end_ramp_down = end_constant + ramp_down_time
        end_off = end_ramp_down + off_time

        # Add points that change the shape of the line
        csv_data.append([start_time, 0])
        csv_data.append([end_ramp_up, 100])
        csv_data.append([end_constant, 100])
        csv_data.append([end_ramp_down, 0])
        csv_data.append([end_off, 0])

    output_path = r"C:\\Users\\akos.beno\\Desktop\\rpa outputs\\led_sequence_graph_points.csv"
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write metadata
        writer.writerow(["Graph Title", "Sample Graph"])
        writer.writerow(["Horizontal Axis Label", "X-axis"])
        writer.writerow(["Vertical Axis Label", "Y-axis"])
        writer.writerow(["Y-axis Scale", ""])
        writer.writerow([])  # Empty row
        # Write column headers
        writer.writerow(["X-axis", "Y-axis"])
        # Write data points
        writer.writerows(csv_data)

    print(f"CSV file saved as '{output_path}'")