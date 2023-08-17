PROMPT = """
Read the table to answer the following question.

Header: xaxis | Cadet Blue | Dark Gray | Burlywood | Dark Magenta | Violet | Dark Seafoam 
Row 1: 20 | 20.3 | 78.9 | 23.1 | 32.8 | 33.1 | 16.17 | 42.34 
Row 2: 40 | 18.4 | 76.2 | 21.1 | 33.5 | 33.7 | 17.41 | 43.06 
Row 3: 60 | 20.5 | 76.5 | 24.0 | 33.5 | 33.5 | 17.38 | 43.23 
Row 4: 80 | 24.7 | 75.7 | 22.3 | 33.2 | 33.2 | 17.08 | 42.59 
Q: Does Dark Magenta have the minimum area under the curve?
A: On two x-axises 20 and 80, the values of Dark Magenta are not the minimum. So the answer is no.

Header: xaxis | Light Gold | Dark Red | Light Green
Row 1: 0 | 74.08 | 74.14 | 74.28 | 74.11 | 74.08 
Row 2: 20 | 77.31 | 75.03 | 74.62 | 75.54 | 76.01 
Row 3: 40 | 77.19 | 76.88 | 76.01 | 77.19 | 78.10 
Row 4: 60 | 74.02 | 77.67 | 76.03 | 77.55 | 80.73 
Row 5: 80 | 74.04 | 78.20 | 77.05 | 77.49 | 82.37
Q: Is Dark Red the smoothest?
A: Among all the groups, the values in Dark Red are the smoothest. So the answer is yes.

Header: xaxis| Orange Red | Firebrick 
Row 1: 0 | 21 | 43 
Row 2: 20 | 25 | 41.6 
Row 3: 40 | 29 | 40.4 
Row 4: 80 | 32.1 | 39.2 
Row 5: 80 | 35.6 | 37.8 
Row 6: 100 | 39.1 | 36.4
Q: Does Orange Red intersect Firebrick?
A: Orange Red is lower than Firebrick at xaxis 0 and higher than Firebrick at xaxis 100. Orange Red must intersect Firebrick. So the answer is yes.

Header: xaxis label | title 
Row 1: Orange | 0 
Row 2: Midnight Blue | 10.7 
Row 3: Peru | 21.2 
Row 4: Firebrick | 32.1 
Row 5: Dark Green | 42.9 
Row 6: Pale Green | 53.4 
Row 7: Purple | 64.3 
Row 8: Cornflower | 75.0 
Row 9: Dark Turquoise | 85.7
Q: Is Dark Green the low median?
A: The medians are 42.9 in Dark Green, 53.5 in Pale Green. Dark Green is lower. So the answer is yes.

Header: xaxis label | Chocolate | Light Salmon | Medium Mint
Row 1: 0 | 22.50 | 24.83 | 28.01
Row 2: 20 | 24.04 | 23.19 | 27.27
Row 3: 40 | 24.21 | 23 | 26.83
Row 4: 60 | 24.06 | 23 | 27.42
Row 5: 80 | 24.27 | 23 | 26.43
Row 6: 100 | 24.86 | 23 | 25.57
Q: Does Chocolate have the highest value?
A: The highest value of Chocolate 24.86 is still lower than 28.01 in Medium Mint. So the answer is no.

{table}
Q: {question}
A:
""".strip()

def formalize_table(table):
    formatted_table = ""
    table = table.replace('<0x0A>', '\n')
    rows = table.split("\n")
    formatted_table += "Header: {}".format(rows[0]) 
    for row_idx, row in enumerate(rows[1:]):
        formatted_table += "\nRow {}: {}".format(row_idx+1, row)
    return formatted_table

