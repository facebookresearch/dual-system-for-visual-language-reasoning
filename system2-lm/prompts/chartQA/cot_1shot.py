PROMPT = """
Answer the following questions step by step.

Q: In which year republicans have the lowest favor rate?
A: Let's describe the figure.
The figure shows the data of: Democrats (brown) | Republicans (dark blue) | Independents (green). The x-axis shows: 2004 | 2006 | 2007 | 2009 | 2011 | 2012 | 2013.
Let's extract the data of republicans. 
The data is 45.0 in 2004, 42.0 in 2006, 38.0 in 2007, 49.0 in 2009, 51.2 in 2011, 48.0 in 2012, 41.0 in 2013.
The smallest number is 38.0, that's in year 2007. So the answer is 2007.

Q: What is the sum of Democrats' favor rates of 2004, 2012, and 2013?
A: Let's describe the figure.
The figure shows the data of: Democrats (brown) | Republicans (dark blue) | Independents (green). The x-axis shows: 2004 | 2006 | 2007 | 2009 | 2011 | 2012 | 2013.
Let's extract the data of Democrats BY 2004.
The data is 68.1.
Let's extract the data of Democrats BY 2012.
The data is 70.0.
Let's extract the data of Democrats BY 2013.
The data is 72.0.
68.1+70.0+72=210.1. So the answer is 210.1.

Q: By how many points do Independents surpass Republicans in the year of 2011?
A: Let's describe the figure.
The figure shows the data of: Democrats (brown) | Republicans (dark blue) | Independents (green). The x-axis shows: 2004 | 2006 | 2007 | 2009 | 2011 | 2012 | 2013.
Let's extract the data of Independents BY 2011.
The data is 58.0.
Let's extract the data of Republicans BY 2011.
The data is 51.2.
58.0-51.2=6.8. So the answer is 6.8.

Q: Which group has the overall worst performance?
A: Let's describe the figure.
The figure shows the data of: Democrats (brown) | Republicans (dark blue) | Independents (green). The x-axis shows: 2004 | 2006 | 2007 | 2009 | 2011 | 2012 | 2013.
Let's extract the data of Democrats.
The data is 68.1 in 2004, 58.0 in 2006, 59.0 in 2007, 72.0 in 2009, 71.0 in 2011, 70.0 in 2012, 72.0 in 2013.
Let's extract the data of Republicans.
The data is 45.0 in 2004, 42.0 in 2006, 38.0 in 2007, 49.0 in 2009, 51.2 in 2011, 48.0 in 2012, 41.0 in 2013.
Let's extract the data of Independents.
The data is 53.0 in 2004, 53.0 in 2006, 45.0 in 2007, 60.0 in 2009, 58.0 in 2011, 53.0 in 2012, 60.0 in 2013.
In year 2004, we find Republicans having the lowest favor rate 45.0 (since 45.0<68.1, 45.0<53.0). In year 2006, we find Republicans having the lowest favor rate 42.0 (42.0<58.0, 42.0<53.0). The trend continues to other years. So the answer is Republicans.

Q: Which party has the second highest favor rates in 2007?
A: Let's describe the figure.
The figure shows the data of: Democrats (brown) | Republicans (dark blue) | Independents (green). The x-axis shows: 2004 | 2006 | 2007 | 2009 | 2011 | 2012 | 2013.
Let's extract the data of 2007.
The data is 59.0 in Democrats, 38.0 in Republicans, 45.0 in Independents.
45.0 is the second highest. 45.0 is the number of Independents. So the answer is Independents

Q: {question}
A:
""".strip()