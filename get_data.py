import json
import pandas as pd

df = pd.DataFrame()
file_path = 'metadata/train_metadata.json'
df['labels'] = None
df['values'] = None
df['title'] = None
df['value_heading'] = None
with open(file_path) as file:
    data = json.load(file)

if data:
    i = 0
    while(len(df) < 1):
        first_element = data[i]
        i += 1
        table_element = first_element['table']
        if(len(table_element) > 2):
            continue
        texts = first_element['texts']
        #iterate through the texts until you find an element with a key 'text_function' having value 'title'
        for text in texts:
            if text['text_function'] == 'title':
                title = text['text']
            if text['text_function'] == 'value_heading':
                value_heading = text['text']
        formatted_output = json.dumps(first_element, indent=4)
        print(formatted_output)
        table_output = json.dumps(table_element, indent=4)
        labels = table_element[0]
        values = table_element[1]
        df.loc[len(df)] = [labels, values, title, value_heading]
        if(len(df) % 100 == 0):
            print(len(df))
    df.to_csv('metadata/train_metadata.csv', index=False)
else:
    print("The JSON file is empty.")