
PROMPT = """
Q: Which property do these three objects have in common?
Answer Choices:
(a) slippery
(b) sweet
(c) salty
A: Let's find out what is the first object.
Potato chips.
Let's find out what is the second object.
Ocean water.
Let's find out what is the third object.
Pretzel.
Potato chips, ocean water and pretzel are all salty. So the answer is salty.

Q: Will these magnets attract or repel each other?
Answer Choices:
(a) attract
(b) repel
A: Let's find out which pole of the first magnet is closest to the other.
North.
Let's find out which pole of the second magnet is closest to the other.
North.
Poles that are the same repel. So the answer is repel.

Q: Compare the average kinetic energies of the particles in each sample. Which sample has the higher temperature?
Answer Choices:
(a) sample A
(b) neither; the samples have the same temperature
(c) sample B
A: Let's find out what is the average speed of the particles in the first sample.
1,300 m/s.
Let's find out what is the mass of each particle in the first sample.
34 u.
Let's find out what is the average speed of the particles in the second sample.
1,300 m/s.
Let's find out what is the mass of each particle in the second sample.
16 u.
The particles in both samples have the same average speed, but each particle in sample A has more mass than each particle in sample B. Thus, sample A must have the higher temperature. So the answer is sample A. 

Q: Which continent is highlighted?
Answer Choices:
(a) Africa
(b) Asia
(c) Europe
A: Let's find out what color is highlighted.
Dark green.
Let's find out which continent is reprensented by dark green.
Europe.
The continent in dark green is Europe. So the answer is Europe.

Q: Which animal's feet are also adapted for sticking to smooth surfaces?
Answer Choices:
(a) American alligator
(b) tiger-striped leaf frog
A: Let's find out whose feet are shown in the figure.
Frog.
Most kinds of frogs' feet are adapted for sticking to smooth surfaces, including tiger-striped leaf frog. So the answer is tiger-striped leaf frog.

Q: {question}
Answer Choices:
{choices}
A:
""".strip()

