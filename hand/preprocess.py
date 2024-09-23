new = []

#d/m/y

with open('datasets/yahoo_stock.csv', 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
        line = line.strip().split(',')
        date = line[0].split('-')
        new_date = f'{date[2]}/{date[1]}/{date[0]}'
        new.append(f'{new_date},{line[4]}\n')

with open(f'new_datasets/yahoo_stock.csv', 'w') as f:
    for line in new:
        f.write(line)
