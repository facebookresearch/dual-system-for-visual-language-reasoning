PROMPT = """
Read the table to answer the following question.

Header: Value | Rural poverty headcount ratio
Row 1: 1996 | 59.6
Row 2: 2001 | 52.1
Row 3: 2007 | 55.0
Q: In which year, was the value of the line maximum?
A: Let's extract the data of Rural poverty headcount ratio. The data is 59.6 in 1996, 52.1 in 2001, 55.0 in 2007. The maximum value is 59.6 in 1996. So the answer is 1996.
 
Header: Value | Central African Republic | Namibia | Yemen, Rep.
Row 1: 1990 | 168.8 | 68.3 | 120.7
Row 2: 2000 | 167.0 | 70.3 | 90.6
Row 3: 2010 | 142.9 | 49.2 | 50.2
Row 4: 2015 | 123.2 | 41.2 | 38.3
Q: What is the difference between the under 5 mortality rate in Namibia in 1990 and the under 5 mortality rate in Yemen, Rep. in 2000?
A: Let's extract the data of Namibia BY 1990. The data is 41.2. Let's extract the data of Yemen, Rep. BY 2000. The data is 50.2. The difference is 41.2-50.2=-9.0. So the answer is -9.0.

Header: Value | 2002 | 2003 | 2004
Row 1: Guyana | 33.11 | 23.06 | 44.83
Row 2: Honduras | 31.96 | 54.9 | 134.97
Row 3: Hong Kong | 728.4 | 359.8 | 390.8
Row 4: Hungary | 94.88 | 95.5 | 98.52
Q: In how many countrys, is the amount in kilograms of the orange bar greater than 51?
A: Let's extract the data of 2003. The data is 23.06 in Guyana, 54.9 in Honduras, 359.8 in Hong Kong, 95.5 in Hungary. The values that are greater than 51 are [54.9, 359.8, 95.5]. So the answer is 3.

Header: Value | Cameroon | Kuwait | Oman
Row 1: Grants | 14.26 | 58.71 | 35.03
Row 2: Taxes | 9.77 | 1.55 | 9.36
Q: What is the revenue of Taxes generated in Cameroon in the year of 2018?
A: Let's extract the data of Cameroon BY Taxes. The data is 9.77. The revenue of Taxes generated in Cameroon is 9.77. So the answer is 9.77.

Header: Value | Female | Male
Row 1: 2000 | 11.9 | 13.9
Row 2: 2006 | 6.6 | 7.7
Row 3: 2011 | 7.4 | 9.4
Q: Is the % of children under 5 of the Female dot in 2000 less than that in 2006?
A: Let's extract the data of Female BY 2000. The data is 11.9. Let's extract the data of Female BY 2006. The data is 6.6. 11.9 is not less than 6.6. So the answer is no.

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

