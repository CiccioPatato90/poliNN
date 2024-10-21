to run the project:
    1. take the kmeans.csv and put in a res folder
    2. run process_replace.py to load the db (res/records.db)
        2.1 this will also create a copy of db in csv -> processed_data.csv
        2.2 run engine.py and let the "magic" happen

- Main branch is behind, current project is on branch "simple_net"
- Run git checkout simple_net and do the steps above
- main_2.L.py has a script that currently run in production doing k-means clustering. The aim is to replace it with the nn
