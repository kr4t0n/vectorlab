import vectorlab as vl

urls = [
    f'www.test.com/samples/{i}'
    for i in range(20)
]

freq_tree = vl.stats.FreqTree(
    split_token='/',
    wild_token='*',
    freq_threshold=1 / 20,
    failed_safe='invalid'
)

transformed_urls = freq_tree.fit_transform(
    urls
)

print(transformed_urls)
