# parse an atop file to a csv
Considering atop recorded everything, you can do
```bash
atopsar -r path/to/your/log.atop -w -m -d -p -c -P > path/to/your/output_recording.csv
```
Then you can read that csv using the command below. First make sure you post-processed GPU and delay values as shown under the examples located in the data folder (.txt format).
```bash
python3 scripts/plot_atopsar_csv_results.py path/to/your/output_recording.csv
```