pairs = [(4, 'Donut'), (1, 'Banana'), (3, 'Apple'), (2, 'Citron')]
pairs.sort(key=lambda pair: pair[1])
print(pairs)
pairs.sort(key=lambda pair:pair[0])
print(pairs)