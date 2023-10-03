PROMPT = """
Read the table to answer the following question.

Header: Title | size | clerk 
Row 1: nobody | 2 | 3 
Row 2: today | 4 | 8 
Row 3: phrase | 8 | 7 
Row 4: heart | 8 | 7
Q: Which group of bars contains the smallest valued individual bar in the whole chart?
A: The value 2.0 in the group of nobody is the smallest. So the answer is nobody.

Header: Value | hay | number
Row 1: supply | 9.0 | 1.0
Row 2: center | 1.0 | 4.0
Row 3: sweat | 2.5 | 4.5
Q: Which stack of bars has the smallest summed value?
A: The summed value of supply is 9.0+1.0=10.0. The summed value of center is 1.0+4.0=5.0. The summed value of sweat is 2.5+4.5=7.0. The smallest summed value is 5.0 in center. So the answer is center.

Header: Value | route | cash | king | taste
Row 1: oxygen | 4.0 | 1.0 | 0.0 | 4.0
Row 2: lamp | 4.0 | 4.0 | 9.0 | 4.0
Row 3: clock | 6.0 | 4.0 | 5.0 | 3.0
Q: How many groups of bars contain at least one bar with value smaller than 9?
A: The groups of oxygen, lamp and clock all contain at least one bar with value smaller than 9. So the answer is 3.

Header: Value | stand | title | crowd
Row 1: horse | 4.0 | 5.0 | 8.0
Row 2: supply | 5.0 | 1.0 | 5.0
Q: Which algorithm has the largest accuracy summed across all the datasets?
A: Let's extract the data of horse. The data is 4.0 in stand, 5.0 in title, 8.0 in crowd. Let's extract the data of supply. The data is 5.0 in stand, 1.0 in title, 5.0 in crowd. The summed value of horse is 4.0+5.0+8.0=17.0. The summed value of supply is 5.0+1.0+5.0=11.0. 17.0 in horse is the largest accuracy summed across all the datasets. So the answer is horse.

Header: Object | hung | pot
Row 1: device | 4.0 | 8.0
Row 2: turn | 8.0 | 8.0
Q: Is the object device in the category hung preferred by more people than the object turn in the category pot?
A: Let's extract the data of hung BY device. The data is 4.0. Let's extract the data of pot BY turn. The data is 8.0. 4.0 is less than 8.0. So the answer is no.

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

