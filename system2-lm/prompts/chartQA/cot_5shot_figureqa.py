PROMPT = """
Answer the following questions step by step.

Q: Does Dark Magenta have the minimum area under the curve?
A: Let's describe the figure.
The figure shows the data of: Cadet Blue (light blue) | Dark Gray (grey) | Burlywood (orange) | Dark Magenta (purple) | Violet (pink) | Dark Seafoam (green). The x-axis shows: 0 | 20 | 40 | 60 | 80 | 100.
Let's extract the data of 20.
The data is 20.3 in Cadet Blue, 75.7 in Dark Gray, 33.0 in Burlywood, 26.0 in Dark Magenta, 20.0 in Violet, 43.0 in Dark Seafoam.
Let's extract the data of 80.
The data is 27.0 in Cadet Blue, 73.3 in Dark Gray, 45.0 in Burlywood, 27.4 in Dark Magenta, 22.8 in Violet, 41.4 in Dark Seafoam.
On two x-axises, the values of Dark Magenta are not the minimum. So the answer is no.

Q: Is Dark Red the smoothest?
A: Let's describe the figure.
The figure shows the data of: Light Gold (yellow) | Dark Red (dard red) | Light Green (green). The x-axis shows: 0 | 20 | 40 | 60 | 80 | 100.
Let's extract the data of Light Gold.
The data is 74.08 in 0, 77.31 in 20, 77.19 in 40, 74.02 in 60, 74.04 in 80.
Let's extract the data of Dark Red.
The data is 74.14 in 0, 75.03 in 20, 76.88 in 40, 77.67 in 60, 78.20 in 80.
Let's extract the data of Light Green.
The data is 74.28 in 0, 74.62 in 20, 76.01 in 40, 76.03 in 60, 77.05 in 80.
Among all the groups, the values in Dark Red are the smoothest. So the answer is yes.

Q: Does Orange Red intersect Firebrick?
A: Let's describe the figure.
The figure shows the data of: Orange Red (red) | Firebrick (dard red). The x-axis shows: 0 | 20 | 40 | 60 | 80 | 100.
Let's extract the data of 0.
The data is 21.9 in Orange Red, 43.0 in Firebrick.
Let's extract the data of 100.
The data is 39.15 in Orange Red, 36.4 in Firebrick.
Orange Red is lower than Firebrick at xaxis 0 and higher than Firebrick at xaxis 100. Orange Red must intersect Firebrick. So the answer is yes.

Q: Is Dark Green the low median?
A: Let's describe the figure.
The figure shows the data of: xaxis label (light blue). The x-axis shows: Orange | Midnight Blue | Peru | Firebrick | Dark Green | Pale Green | Purple | Cornflower | Dark Turquoise.
Let's extract the data of xaxis label.
The data is 0.0 in Orange, 10.7 in Midnight Blue, 21.2 in Peru, 32.0 in Firebrick, 42.9 in Dark Green, 53.5 in Pale Green, 64.4 in Purple, 75.0 in Cornflower, 85.6 in Dark Turquoise.
The medians are 42.9 in Dark Green, 53.5 in Pale Green. Dark Green is lower. So the answer is yes.

Q: Does Chocolate have the highest value?
A: Let's describe the figure.
The figure shows the data of: Chocolate (brown) | Light Salmon (orange) | Medium Mint (light green). The x-axis shows: 0 | 20 | 40 | 60 | 80 | 100.
Let's extract the data of 0.
The data is 23.24 in Chocolate, 23.0 in Light Salmon, 28.0 in Medium Mint.
Let's extract the data of 100.
The data is 24.2 in Chocolate, 23.0 in Light Salmon, 24.3 in Medium Mint.
The highest value of Chocolate 24.2 is still lower than 28.0 in Medium Mint. So the answer is no.

Q: {question}
A:
""".strip()

