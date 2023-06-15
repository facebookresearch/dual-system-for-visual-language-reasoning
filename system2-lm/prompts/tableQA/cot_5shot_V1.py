PROMPT = """
Read the table to answer the following question.

Header: Entity | 2008 | 2009 | 2010 | 2011 | 2012 | 2013 | 2014
Row 1: Oman | 183.88 | 233.80 | 210.69 | 195.26 | 196.32 | 154.21 | 153.22
Row 2: Samoa | 40.72 | 40.04 | 39.21 | 40.63 | 41.47 | 41.76 | 42.77
Q: In which year the private health expenditure per person in Oman is 210.69?
A: Let's find the row of Oman, that's Row 1. Let's find the number 210.69, that's for the year of 2010. The answer is 2010. 

Header: Year | NET Excellent/ good | NET Only fair/ poor
Row 1: Feb 2014 | 54 | 39
Row 2: June 2015 | 55 | 0
Row 3: Jan 2018 | 45 | 46
Row 4: Sept. 2018 | 331 | 62
Q: By how many points does NET Excellent/good surpass NET Only fair/poor in the year of 2014?
A: Let's find the row of 2014, that's Row 1. We extract NET Excellent/good's and NET Only fair/poor's numbers. They are 54.00 and 39.00. 54.00-39.00=15.00. The answer is 15.00.

Header: Characteristic | Share of respondents who believe Coronavirus is a threat
Row 1: United Kingdom | 60%
Row 2: Italy | 60%
Row 3: Germany | 38%
Row 4: Finland | 29%
Q: How many countries have a value below 40%?
A: Let's find the values that are below 40: [38, 29]. The answer is 2.

Header: Characteristic | Consoles | PC | Mobile and ancillary | Other
Row 1: 2020 | 2784 | 2056 | 2559 | 687
Row 2: 2019 | 1920 | 1718 | 2203 | 648
Row 3: 2018 | 2538 | 2180 | 2175 | 607
Row 4: 2017 | 2389 | 2042 | 2081 | 505
Row 5: 2016 | 2453 | 2124 | 1674 | 357
Row 6: 2015 | 2391 | 1499 | 418 | 356
Q: How much money did Activision Blizzard's console segment generate in annual revenues in 2020?
A: Let's find the row of 2020, that's Row 1. Let's find the number for Consoles, that's 2784. The answer is 2784.

Header: Characteristic | Share of respondents
Row 1: Decreased | 81%
Row 2: No impact | 16%
Row 3: Increased | 3%
Q: Is the sum of two smallest segments greater than the largest segment?
A: Let's extract the numbers of all the rows: [81, 16, 3]. The two smallest numbers are 16 and 3. 16+3=19, which is smaller than 81. The answer is no.

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

# 00268016000832, 4549, two_col_101055, two_col_42979, multi_col_1490, two_col_62894
# Header: Characteristic | population with access to improved sanitation
# Row 1: 2015 | 96%
# Row 2: 2014 | 96%
# Row 3: 2013 | 96%
# Row 4: 2012 | 96%
# Row 5: 2011 | 96%
# Row 6: 2010 | 95%
# Row 7: 2009 | 95%
# Row 8: 2008 | 95%
# Row 9: 2007 | 94%
# Row 10: 2006 | 94%
# Row 11: 2005 | 93%
# Q: What was the percentage of the Malaysian population that had access to improved sanitation in 2013?
# A: Let's find the row of 2013, that's Row 3. Let's extract the number in Row 3, that's 96. The answer is 96.

