PROMPT = """
Read the table below to answer the following questions.

Header: Year | Democrats | Republicans | Independents
Row 1: 2004 | 68.1% | 45.0% | 53.0%
Row 2: 2006 | 58.0% | 42.0% | 53.0%
Row 3: 2007 | 59.0% | 38.0% | 45.0%
Row 4: 2009 | 72.0% | 49.0% | 60.0%
Row 5: 2011 | 71.0% | 51.2% | 58.0%
Row 6: 2012 | 70.0% | 48.0% | 53.0%
Row 7: 2013 | 72.0% | 41.0% | 60.0%

Q: In which year republicans have the lowest favor rate?
A: Let's find the column of republicans. Then let's extract the favor rates: [45.0, 42.0, 38.0, 49.0, 51.2, 48.0, 41.0]. The smallest number is 38.0, that's Row 3. Row 3 is year 2007. The answer is 2007.

Q: What is the sum of Democrats' favor rates of 2004, 2012, and 2013?
A: Let's find the rows of years 2004, 2012, and 2013. We find Row 1, 6, 7. The favor dates of Democrats on that 3 rows are 68.1, 70.0, and 72.0. 68.1+70.0+72=210.1. The answer is 210.1.

Q: By how many points do Independents surpass Republicans in the year of 2011?
A: Let's find the row with year = 2011. We find Row 5. We extract Independents and Republicans' numbers. They are 58.0 and 51.2. 58.0-51.2=6.8. The answer is 6.8.

Q: Which group has the overall worst performance?
A: Let's sample a couple of years. In Row 1, year 2004, we find Republicans having the lowest favor rate 45.0 (since 45.0<68.1, 45.0<53.0). In year 2006, Row 2, we find Republicans having the lowest favor rate 42.0 (42.0<58.0, 42.0<53.0). The trend continues to other years. The answer is Republicans.

Q: Which party has the second highest favor rates in 2007?
A: Let's find the row of year 2007, that's Row 3. Let's extract the numbers on Row 3: [59.0, 38.0, 45.0]. 45.0 is the second highest. 45.0 is the number of Independents. The answer is Independents.

Read the table below to answer the following questions.

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