# parse an atop file to a csv
Considering atop recorded everything, you can do
```bash
atopsar -r data/recording.atop -w -m -d -p -c -P > data/recording.csv
```
Then you can read that csv using the python scripts :
```bash
python3 scripts/my_script.py
```