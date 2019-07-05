# IP-ESN
ESN with IP rules and new weight initilaization, which is proposed by Liu Zongying.

The code of basic ESN algorithm is developed by paper named 'Real-Time Paralled Processing of Grammatical Structure in the Fronto-Striatal System: A Recurrent Network Simulation Study Using Reservoir Computing', which is based on the grammatical structure experiments and proposed by Xavier Hinaut. 

The basic ESN code is from the website: https://sites.google.com/site/xavierhinaut/downloads

Install IP-ESN

(1). Install python and Oger.

(2). Based on the description in the link(https://journals.plos.org/plosone/article/file?id=info%3Adoi/10.1371/journal.pone.0052946.s006&type=supplementary) to install ESN.

(3). After installed ESN in the step (2), copy IP-ESN.py, IP-search.py, and weight_function.py to '.../scripts_plos/'.

(4). Find out the location of Oger library that has already installed in your computer, and then go to direction 'Oger/nodes/reservoir_nodes'.

(5). Copy all content in 'new_reservoir_nodes.py' and replace the content of location in step (4).

(6). run IP-ESN.py
