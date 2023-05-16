import pandas as pd

data_df = pd.read_csv('C:\\Users\\johnh\\OneDrive\\Documents\\공부\\2023년\\인공지능프로젝트\\기말프로젝트\\data_base.csv', names=['id', 'state', 'sunlight'], header=0)
print(data_df.head())
us_state = {'MN':'minnesota', 'ND':'north dakota', 'SD':'south dakota', 'AR':'arkansas', 'CA':'california', 'LA':'louisiana', 'IL':'illinois', 'IA':'iowa', 'NE':'nebraska'}
data_to_use = {}


# with open('sunlight.txt', 'r', encoding='utf-8') as fr:
#     while True:
#         line = fr.readline()
#         if not line:
#             break
#         line_data = [ele.replace('\n', '') for ele in line.split(' ') if ele]
#         line_state = line_data[3]
#         if line_state in us_state.keys():
#             with open(f"{line_state}.txt", 'a', encoding='utf-8') as fw:
#                 fw.write(line)        


#for city in us_state.keys():
city_df = pd.DataFrame(index=range(88))
with open(f"MN.txt", 'r', encoding='utf-8') as fr:
    base_city = 'DULUTH'
    city_data = [0]*88
    while True:
        line = fr.readline()
        if not line:
            city_df[f"{base_city}"]=city_data
            print(city_df.head(20))
            break
        line_city = line[20:33].strip()
        line_year = line[37:41]
        line_data = line[43:]
        if base_city != line_city:
            city_df[f"{base_city}"]=city_data
            print(city_df.head(20))
            base_city = line_city
            city_data = [0]*88
        if line_year>=str(1900):
            city_data[int(line_year)-1900] = tuple([ele.replace('\n', '') for ele in line_data.split(' ') if ele])
        #print(f"city : {line_city} :: year : {line_year} :: data : {line_data}")
        
            