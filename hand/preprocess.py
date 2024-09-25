new = []

#d/m/y

with open('datasets/minimum-temp.csv', 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
        line = line.strip().split(',')
        date = line[0].split('/')
        new_date = f'{date[1]}/{date[0]}/{date[2]}'
        new.append(f'{new_date},{line[1]}\n')

with open(f'new_datasets/minimum_temp.csv', 'w') as f:
    for line in new:
        f.write(line)
