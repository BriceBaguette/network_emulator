# network_emulator

To run on test topology: 

python main.py .\topology\node.json .\topology\link.json

The graph of the test topology is the next one: 

[[     0      1      1      1      0]
 [     1      0      200000 0      1]
 [     1      200000 0      0      1]
 [     1      0      0      0      1]
 [     0      1      1      1     0]]